# Loom

Loom provides two layers for working with graph-based intermediate
representations (IRs):

1. **IR** — typed metadata, schema validation, and graph-invariant checks.
2. **Engine** — AQL pattern matching, single-pass rewriting, and fixed-point
   rule execution.

Loom operates on [`HGraph`](https://github.com/heterograph/heterograph), whose
vertices and edges can carry property dictionaries.

## 1. IR

The IR layer defines the metadata allowed on a graph, its vertices, and its
edges. Schemas use Pydantic models, while ordinary Python functions provide
structural graph checks.

### Defining a schema

```python
from pydantic import BaseModel, Field
from loom import IRSchema, Issue, Severity


class Tensor(BaseModel):
    kind: str = Field(alias="_type")
    name: str
    shape: list[int]


class Operation(BaseModel):
    kind: str = Field(alias="_type")
    name: str


class DataEdge(BaseModel):
    kind: str = Field(alias="_type")
    index: int


class GraphProps(BaseModel):
    kind: Literal["example-ir"] = Field(alias="_type")
    name: str = "example"


def require_operation_inputs(graph):
    issues = []
    for vertex in graph.vertices:
        if graph.pmap[vertex].get("_type") != "operation":
            continue
        if graph.num_in_vx(vertex) == 0:
            issues.append(Issue(
                Severity.WARNING,
                f"vertex {vertex}",
                "operation has no inputs",
            ))
    return issues


schema = IRSchema(
    name="example-ir",
    graph=GraphProps,
    vertex={
        "tensor": Tensor,
        "operation": Operation,
    },
    edge={"data": DataEdge},
    rules=[require_operation_inputs],
)
```

Every vertex and edge schema must include `_type`. The `_type` value selects
which Pydantic model validates that item. Extra properties are rejected.

### Creating typed graphs

```python
graph = schema.new_graph()

input_tensor = schema.new_vertex(
    graph,
    "tensor",
    name="input",
    shape=[1, 16],
)
layer = schema.new_vertex(
    graph,
    "operation",
    name="dense",
)

schema.new_edge(
    graph,
    input_tensor,
    layer,
    "data",
    index=0,
)
```

Use `update_vertex()` and `update_edge()` to change metadata while validating
the complete updated object:

```python
schema.update_vertex(graph, layer, name="dense_1")
schema.update_edge(graph, (input_tensor, layer), index=1)
```

Unknown types and invalid properties raise immediately when creating or
updating an item:

```python
schema.new_vertex(graph, "operation", name=123)  # validation error
schema.new_vertex(graph, "unknown", name="bad")  # KeyError
```

### Validating a graph

```python
report = schema.validate(graph)

print(report.ok)       # True when there are no errors
print(report.errors)   # error Issue objects
print(report)          # human-readable summary
```

Validation checks:

- graph-level metadata and graph `_type`;
- vertex metadata and vertex `_type`;
- edge metadata and edge `_type`;
- every registered structural rule.

An `Issue` has a severity, location, and message. Warnings are reported but do
not make `ValidationReport.ok` false; errors do.

### Styling

`IRSchema.new_graph()` calls the overridable `style(graph)` method. The
default style uses a left-to-right layout and labels vertices with `_type`.
Schemas can override it for IR-specific colours and labels:

```python
class StyledIR(IRSchema):
    def style(self, graph):
        super().style(graph)
        graph.vstyle["shape"] = "Mrecord"
        graph.vstyle["fillcolor"] = "lightblue"
```

## 2. Engine

The engine has three related pieces:

- `Processor` performs matching and one rewrite pass.
- `Rule` and `RuleFn` package graph transformations.
- `Runner` executes phased rules and repeats NORMAL rules until a fixed point.

### AQL

The engine uses AQL (Acyclic/Attributed Query Language) to describe query
graphs. AQL is used for both `select` patterns and `rewrite` RHS patterns.

The basic grammar is:

```text
identifier ::= letters, digits, and underscores, starting with a letter
node       ::= identifier | identifier { arguments }
edge       ::= => | ={ arguments }>
graph      ::= node | graph edge graph | (graph | graph)
```

Whitespace is ignored. Node and edge identifiers are pattern-local names; they
are not required to be integers or to match host vertex IDs.

#### Nodes and directed edges

```python
"a"
"a => b"
"a => b => c"
```

`=>` creates a directed edge. The last form describes the chain:

```text
a → b → c
```

Repeated identifiers refer to the same pattern node, so a pattern can express
branching or joining:

```python
"input => left; input => right"
```

When several independent fragments are needed, `QGraph` accepts semicolon-
separated AQL expressions. This is especially useful for rewrite annotations
or disconnected RHS components.

#### Node and edge attributes

Attributes use braces. They can be positional values or key-value pairs:

```python
"tensor {input, shape: 16}"
"tensor ={index: 0, channel: \"data\"}> operation {dense}"
```

Supported literal values include identifiers, quoted strings, integers,
floating-point numbers, and `True`/`False`. Positional and keyword attributes
are retained on the parsed query graph. Matching policies can use them to
annotate or constrain matches.

The rewrite engine uses node attributes for rewire directives, for example:

```python
"a => x {rewire:b} => c"
```

Here `rewire:b` is interpreted by the rewrite processor; it is not a generic
host-graph property.

#### Groups and fan-in/fan-out

Parentheses with `|` combine graph fragments:

```python
"q => (a | b | c) => d"
```

This describes:

```text
q → a → d
q → b → d
q → c → d
```

Groups are useful for expressing one-to-many and many-to-one structures while
keeping the pattern compact.

#### Empty patterns and repeated nodes

The literal `0` represents an empty query graph:

```python
"0"
```

An identifier appearing more than once is one node, not multiple nodes:

```python
"a => b; a => c"
```

describes one `a` with two outgoing edges. In a rewrite, identifiers present
on both sides refer to preserved host vertices; identifiers appearing only on
the RHS create new host vertices.

### Matching

`Processor.run()` accepts an AQL `select` expression. Processor snapshots are
enabled by default; pass `snapshot=False` to disable them and call
`snapshot_view()` to display captured original/rewrite graphs. The simple edge operator
`=>` describes a directed edge:

```python
from heterograph import HGraph
from loom.engine import Processor

graph = HGraph()
graph.add_vx(3)
graph.add_edge(0, 1)
graph.add_edge(1, 2)

p = Processor()
result = p.run(graph, select="a => b => c")

print(result["matches"])
# [{'a': 0, 'b': 1, 'c': 2}]
```

Bindings map pattern IDs to host vertex IDs. A `where` callback can apply a
semantic filter after structural matching:

```python
result = p.run(
    graph,
    select="a => b",
    where=lambda g, a, b: g.num_out_vx(a) == 1,
)
```

The default `IsoMatchPolicy` uses induced subgraph isomorphism. Matching is
separate from rewriting: a query may return overlapping matches, but a rewrite
requires all selected matches to be disjoint. Overlapping rewrite matches raise
`RuntimeError`.

### Basic rewriting

```python
graph = HGraph()
graph.add_vx(3)
graph.add_edge(0, 1)
graph.add_edge(1, 2)

p.run(
    graph,
    select="a => b => c",
    rewrite="a => c",
)
```

The rewrite compares the LHS (`select`) with the RHS (`rewrite`):

```text
preserved = LHS ∩ RHS
deleted   = LHS - RHS
created   = RHS - LHS
```

For the example:

```text
preserved: a, c
deleted:   b
created:   none
```

The edge differences are applied as follows:

- RHS-only edges are added.
- LHS-only edges between preserved vertices are removed.
- Deleted vertices and their remaining incident edges are removed.
- Preserved vertices remain the same host vertices.

Rewriting is single-pass and in-place. Matches are found once at the start of
`run()`; newly created structure is not matched again during that call.

### Replacing a node and inheriting its edges

Rewire annotations are placed on the RHS receiver. The referenced ID is the
LHS node that will be deleted:

```python
graph = HGraph()
graph.add_vx(5)
for source, target in [(0, 1), (1, 2), (3, 1), (1, 4)]:
    graph.add_edge(source, target)

p.run(
    graph,
    select="a => b => c",
    rewrite="a => x {rewire:b} => c",
)
```

This means:

- `b` is deleted because it is absent from the RHS;
- `x` is created;
- `x` inherits external incoming edges of `b`;
- `x` inherits external outgoing edges of `b`.

The resulting edges are:

```text
0 → x → 2
3 → x
x → 4
```

The directional forms are:

```text
x {rewire:b}       # incoming and outgoing edges
x {rewire_in:b}    # incoming edges only
x {rewire_out:b}   # outgoing edges only
```

Only edges crossing the match boundary are redirected. Internal edges must be
declared explicitly by the RHS. For example:

```python
# Valid: the implied internal edge a -> x is explicit.
rewrite = "a => x {rewire_in:b}"

# Invalid: rewiring a -> b would imply a -> x, but the RHS omits it.
rewrite = "a; x {rewire_in:b}"
```

Invalid rewire annotations raise `RuntimeError`. The receiver must exist in the
RHS, and each referenced source must be an LHS-only (deleted) node.

### Finalize callbacks

`finalize` runs after RHS nodes, RHS edges, and external rewiring have been
applied, but before LHS cleanup:

```python
def finalize(graph, a, c, x):
    graph.pmap[x]["lowered_from"] = "b"
    return True


result = p.run(
    graph,
    select="a => b => c",
    rewrite="a => x {rewire:b} => c",
    finalize=finalize,
)
```

The callback must return an actual Python `bool`:

- `True` marks the graph as modified;
- `False` means the callback made no metadata change;
- any other return type raises `RuntimeError`.

`finalize` may update metadata and may perform other graph updates, but it must
not remove vertices or edges that the current rewrite still needs to clean up.
The processor checks those planned cleanup targets before removing them.

### Snapshot visualization

Snapshots are enabled by default on `Processor` and are rendered into an
internal Heterograph `WebView`:

```python
p = Processor(snapshot=True)
p.run(graph, select="a => b", rewrite="a => x")
p.snapshot_view()
```

For each run that specifies `rewrite`, the processor records:

```text
original #n
rewrite #n
```

The rewrite snapshot is recorded even when `modified` is false. Matching-only
runs do not add snapshots. Use `Processor(snapshot=False)` to disable the
viewer and avoid rendering snapshots.

### Rules and fixed-point execution

For reusable transformations, wrap a function in `RuleFn`:

```python
from loom.engine import RuleFn, Runner


def remove_unconnected(graph):
    changed = False
    for vertex in list(graph.vertices):
        if graph.num_in_vx(vertex) == 0 and graph.num_out_vx(vertex) == 0:
            graph.rm_vx(vertex)
            changed = True
    return changed


rule = RuleFn(
    remove_unconnected,
    name="remove-unconnected",
    description="Remove isolated vertices",
)

runner = Runner(snapshot=False)
runner.add_rule(rule)
result = runner.execute(graph, max_iterations=20, verbose=True)
```

`Runner` has three phases. `PRE` rules run once before the loop, `NORMAL`
rules run repeatedly until a full pass makes no changes, and `POST` rules run
once afterward. `NORMAL` is the default:

```python
from loom.engine import RulePhase

runner.add_rule(check_source, phase=RulePhase.PRE)
runner.add_rule(lower_rule)  # NORMAL
runner.add_rule(check_target, phase=RulePhase.POST)
```

`CheckSchemaRule` is a non-mutating rule that raises `RuntimeError` when a
graph fails schema validation. This makes source-to-target lowering explicit:

```python
from loom.ir import CheckSchemaRule

runner.add_rule(CheckSchemaRule(NN_IR), phase=RulePhase.PRE)
runner.add_rule(LowerDenseRule())
runner.add_rule(CheckSchemaRule(OP_IR), phase=RulePhase.POST)
runner.execute(graph)
```

The lowering rule must set `graph.pmap["_type"] = OP_IR.name` before the POST
check. A rule must return an actual Python `bool`; `Runner` rejects other
return types. It stops when a full NORMAL pass makes no changes or when
`max_iterations` is reached.

Runner snapshots are enabled by default. The initial graph and every graph
state produced by a modifying rule are recorded and can be viewed with:

```python
runner = Runner()
runner.add_rule(rule)
runner.execute(graph)
runner.snapshot_view()
```

Use `Runner(snapshot=False)` to disable snapshots.

The result has this shape:

```python
{
    "iterations": 3,
    "modified": True,
}
```

`Processor` is the AQL-oriented API for local matching and rewriting; `Runner`
is the phased fixed-point driver for composing reusable rules.
