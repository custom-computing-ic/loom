# Loom

Loom is a Python framework for building graph-based compiler flows that
are easy to extend, test, and verify. It is designed to support human-built and AI-assisted compilers.

## Why Loom

A compiler usually has many separate concerns: import a model, canonicalize an
IR, lower operations, optimize, schedule, estimate resources, emit an
artifact, and check the result. When these concerns live in one large compiler
function, each change becomes difficult to isolate, reuse, or verify.

Loom gives each concern a clear home. Schemas define the legal
representations. Actions perform small pieces of work. Tasks compose Actions
to solve one compiler concern. Pipelines compose Tasks into an application
workflow. Contracts decide whether a workflow result is acceptable.

## How the concepts fit together

The diagram below shows the execution path. An IR is an `HGraph`; it is
validated by a Schema and may be transformed by Processor-backed Actions.
Tasks and Pipelines decide when those operations run. Contracts sit outside
the compiler work and judge the result of a selected Pipeline workflow.

```text
Workflow execution

Model or source artifact
          │
          ▼
Pipeline workflow ──► Task ──► Action ──► Processor (optional) ──► HGraph / IR
       ▲                  │         │                                      ▲
       │                  │         └──── may read, analyse, or modify ───┘
       │                  ▼
       │             TaskResult ◄── ActionResult
       │                  │
       └────────── PipelineResult

IRSchema ─────────────────────────────── validates ───────────────────────► HGraph / IR

Verification

Verifier ── invokes each ──► Contract ── runs named workflow ──► Pipeline workflow
                               ▲                                      │
                               └──── evaluates PipelineResult ◄────────┘
                               │
                               ▼
                         ContractResult ──► Verifier aggregates all results
```

The layers have deliberately different responsibilities:

```text
Schema  → defines a legal graph IR
Processor → matches and rewrites graph structure
Action  → performs one reusable unit of work
Task    → orchestrates Actions for one compiler concern
Pipeline → composes Tasks into a workflow
Contract → verifies one Pipeline workflow
Verifier → runs every Contract and aggregates feedback
```

`Processor` is optional: an Action can perform analysis or generate an
artifact without changing a graph. Similarly, a Task is free to use custom
Python control flow instead of a fixed-point loop.

## Human-built and AI-assisted compilers

Loom supports a normal, entirely human-built compiler. Developers define the
schemas, implement Actions and Tasks, compose Pipelines, and use Contracts as
automated regression and acceptance checks.

```text
Developer
  ├── defines Schemas
  ├── implements Actions and Tasks
  ├── composes Pipelines
  └── writes Contracts
            │
            ▼
    Pipeline executes compiler work
            │
            ▼
    Contracts accept or reject the result
```

The same separation supports AI-assisted development without treating generated
code as trusted. Humans retain ownership of the specification and verification
machinery. An agent is assigned one bounded Task or Action; trusted code places
that candidate in a Pipeline workflow. Every Contract runs, and the `Verifier`
returns all failures and metrics as feedback for the next revision.

```text
Human defines: Schemas, trusted Pipeline, Contracts
                         │
                         ▼
Agent builds one candidate Action or Task
                         │
                         ▼
Trusted Pipeline workflow runs the candidate with trusted work
                         │
                         ▼
Verifier runs every Contract
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
  all Contracts pass           failures and metrics
          │                             │
          ▼                             └──► Agent revises candidate
   accept candidate
```

The agent may understand the task interface and contract expectations, but it
does not own the trusted Pipeline or decide whether it succeeded. This makes
the unit of generation small while allowing verification to span downstream
tasks, simulation, or artifact checks.

## Installation

Loom requires Python 3.10+ and uses
[Heterograph](https://github.com/heterograph/heterograph) for graph storage,
query parsing, and visualization.

For the included examples, create the supplied Conda environment:

```bash
conda env create -f examples/environment.yml
conda activate loom-examples
```

The environment installs Loom in editable mode. The Keras lowering example
also requires TensorFlow; the schema and processor examples do not.

## The graph IR model

An IR is a `heterograph.HGraph` whose graph, vertices, and edges hold property
dictionaries. Every typed vertex and edge uses `_type` to choose its schema.

```text
HGraph
  graph properties:  {"_type": "NN-IR", ...}
  vertex properties: {"_type": "Dense", ...}
  edge properties:   {"_type": "data", ...}
```

Schemas describe what is legal. Actions and Tasks perform work. Pipelines
compose configured Tasks with ordinary Python. Contracts judge a Pipeline
workflow rather than trusting the Task that produced it.

## Define and validate an IR

`IRSchema` combines Pydantic metadata validation with graph-level structural
checks. The schema is normally human-defined: it is the trusted boundary that
distinguishes valid from invalid representations.

```python
from typing import Literal

from pydantic import BaseModel, Field
from heterograph import HGraph
from loom.ir import IRSchema, Issue, Severity


class Tensor(BaseModel):
    kind: Literal["tensor"] = Field(alias="_type")
    name: str
    shape: list[int]


class Operation(BaseModel):
    kind: Literal["operation"] = Field(alias="_type")
    name: str


class DataEdge(BaseModel):
    kind: Literal["data"] = Field(alias="_type")
    index: int = Field(ge=0)


def require_operation_inputs(graph: HGraph) -> list[Issue]:
    issues = []
    for vertex in graph.vertices:
        if graph.pmap[vertex].get("_type") == "operation" and graph.num_in_vx(vertex) == 0:
            issues.append(Issue(
                Severity.ERROR,
                f"vertex {vertex}",
                "operation requires an input",
            ))
    return issues


IR = IRSchema(
    name="example-ir",
    vertex={"tensor": Tensor, "operation": Operation},
    edge={"data": DataEdge},
    rules=[require_operation_inputs],
)
```

Use the schema to create typed graphs. Creation and updates validate the full
property dictionary immediately; unknown types and extra fields are rejected.

```python
graph = IR.new_graph()
x = IR.new_vertex(graph, "tensor", name="input", shape=[1, 16])
op = IR.new_vertex(graph, "operation", name="dense")
IR.new_edge(graph, x, op, "data", index=0)

IR.update_vertex(graph, op, name="dense_1")
report = IR.validate(graph)
assert report.ok
```

`validate()` returns a `ValidationReport` rather than raising. It contains
every `Issue`; warnings do not make `report.ok` false, while errors do.

```python
for issue in report.issues:
    print(issue)
```

The validation pass always checks the graph `_type`, typed vertex and edge
metadata, and the schema's structural `rules`. If a graph must be checked in a
Task, use `CheckSchemaAction(schema)`: it raises `RuntimeError` on an invalid
graph and otherwise returns an unchanged `ActionResult`.

### Styling

`IRSchema.new_graph()` calls `style(graph)`. Override it in a schema subclass
to control Heterograph visualization without mixing presentation concerns into
validation:

```python
class StyledIR(IRSchema):
    def style(self, graph):
        super().style(graph)
        graph.vstyle["shape"] = "Mrecord"
        graph.vstyle["fillcolor"] = "lightblue"
```

## Match and rewrite graphs with `Processor`

`Processor` is Loom's low-level mechanism for matching AQL patterns and
rewriting an `HGraph` in place. It deliberately represents mechanism rather
than compiler intent; production transformations usually wrap it in an
`Action`.

```python
from heterograph import HGraph
from loom.engine import Processor

graph = HGraph()
graph.add_vx(3)
graph.add_edge(0, 1)
graph.add_edge(1, 2)

processor = Processor(snapshot=False)
result = processor.run(
    graph,
    select="a => b => c",
    rewrite="a => c",
)

assert result["modified"]
assert set(graph.edges) == {(0, 2)}
```

`run()` returns a dictionary with:

- `matches`: pattern-ID to host-vertex-ID bindings;
- `pattern`: the parsed Heterograph query graph;
- `modified`: whether the rewrite changed the host graph.

### AQL patterns

Identifiers are local names in a pattern. `=>` is a directed edge; repeated
identifiers refer to the same node.

```python
"a => b => c"          # chain
"input => left; input => right"  # branching fragments
"q => (a | b | c) => d" # fan-out and fan-in
"0"                    # empty pattern
```

Node and edge attributes use braces and are available to custom match
policies:

```python
"tensor {input, shape: 16}"
"tensor ={index: 0}> operation {dense}"
```

Matching may use a semantic `where` filter after structural matching:

```python
processor.run(
    graph,
    select="a => b",
    where=lambda g, a, b: g.num_out_vx(a) == 1,
)
```

By default `Processor` uses `IsoMatchPolicy`, which performs induced subgraph
isomorphism. `DfsMatchPolicy` is also available for DFS-oriented matching. For
either policy, `max_n` limits matches and `deduplicate` controls whether
bindings covering the same host vertices are collapsed.

### Rewrite semantics

The processor compares the left-hand `select` pattern and the right-hand
`rewrite` pattern:

```text
preserved vertices = LHS ∩ RHS
deleted vertices   = LHS − RHS
created vertices   = RHS − LHS
```

RHS-only edges are added. LHS-only edges between preserved vertices are
removed. Deleted vertices and their incident edges are removed. Rewrites are
single-pass: newly created structure is not rematched in the same call.

All rewrite matches must be vertex-disjoint. An overlap is rejected with
`RuntimeError`; use `max_n=1` and a `FixedPointTask` when adjacent patterns
naturally overlap.

### Rewiring boundary edges

An RHS node can inherit external edges from an LHS-only node being deleted:

```python
processor.run(
    graph,
    select="a => b => c",
    rewrite="a => x {rewire:b} => c",
)
```

The created `x` inherits both incoming and outgoing edges of `b` that cross
the matched subgraph boundary. Directional forms are also available:

```text
x {rewire:b}       incoming and outgoing external edges
x {rewire_in:b}    incoming external edges only
x {rewire_out:b}   outgoing external edges only
```

Internal graph changes must be explicit on the RHS. Loom rejects a rewire that
would silently create an undeclared edge between preserved and replacement
nodes. Rewired edges copy the original edge property dictionary.

### Finalize metadata

Use `finalize` after new RHS nodes, edges, and boundary rewiring have been
created, but before obsolete vertices and edges are removed:

```python
def finalize(graph, a, b, c, x):
    graph.pmap[x]["lowered_from"] = "b"
    return True


processor.run(
    graph,
    select="a => b => c",
    rewrite="a => x {rewire:b} => c",
    finalize=finalize,
)
```

The callback receives every bound pattern ID and must return an actual Python
`bool`. It may update metadata, but it must not remove vertices or edges that
the active rewrite still needs to clean up.

### Snapshots

`Processor(snapshot=True)` is the default. Each rewrite captures `original #n`
and `rewrite #n` in a Heterograph WebView:

```python
processor = Processor()
processor.run(graph, select="a => b", rewrite="a => x")
processor.snapshot_view()
```

Use `snapshot=False` in non-interactive or performance-sensitive code.

## Actions

An `Action` is a reusable, stateless unit of compiler work. It may mutate the
graph, analyse it, produce an artifact, or combine these. Configuration is set
in its constructor; per-execution state lives in local variables, the graph,
or its result.

```python
from dataclasses import dataclass

from loom.engine import Action, ActionResult


@dataclass(frozen=True)
class CountOutput:
    operations: int


class CountOperations(Action):
    def __init__(self, *, operation_type: str = "operation"):
        super().__init__(name="count-operations")
        self.operation_type = operation_type

    def apply(self, graph) -> ActionResult:
        count = sum(
            graph.pmap[vertex].get("_type") == self.operation_type
            for vertex in graph.vertices
        )
        return ActionResult(output=CountOutput(operations=count))
```

Every Action returns:

```python
ActionResult(
    output=...,    # optional analysis result or artifact
    modified=...,  # actual bool; whether the graph changed
)
```

`modified` is separate from `output` so fixed-point orchestration can be
reliable even when an Action returns analysis data. For a small function, use
`ActionFn` instead of a subclass:

```python
from loom.engine import ActionFn, ActionResult

annotate = ActionFn(
    lambda graph: ActionResult(output=len(graph.vertices)),
    name="count-vertices",
)
```

## Tasks

A `Task` owns one bounded compiler concern and decides how its Actions are
orchestrated. It is also stateless: constructor arguments are its configuration
and `execute()` returns the run result.

```python
from loom.engine import Task, TaskResult


class CanonicalizeTask(Task):
    def __init__(self, cleanup):
        super().__init__(name="canonicalize")
        self.cleanup = cleanup

    def execute(self, graph) -> TaskResult:
        action_result = self.cleanup.apply(graph)
        return TaskResult(
            output=action_result.output,
            modified=action_result.modified,
        )
```

`TaskResult` has the same standard shape as `ActionResult`:

```python
TaskResult(output=..., modified=False)
```

Custom orchestration is the default. Use Python directly for conditions,
feedback loops, searches, and any domain-specific control flow.

### Fixed-point tasks

`FixedPointTask` is a reusable Task template for the common pattern of running
setup Actions once, iterative Actions until a stable pass, then final checks.

```python
from loom.engine import FixedPointTask
from loom.ir import CheckSchemaAction

lowering = FixedPointTask(
    name="lower-network",
    pre=[CheckSchemaAction(SourceIR)],
    iterative=[lower_one_operation, cleanup],
    post=[CheckSchemaAction(TargetIR)],
    max_iterations=100,
    snapshot=False,
)

result = lowering.execute(graph, verbose=True)
```

The three lists are:

- `pre`: runs once before iteration;
- `iterative`: repeats until a complete pass has no modified Action;
- `post`: runs once after iteration stops.

`max_iterations` is optional; it is a safety cap when provided. The Task
result's `output` is a dictionary containing `iterations` and
`action_results`, where each entry is `(phase, ActionResult)`. The task-level
`modified` flag is true if any Action modified the graph.

Snapshots are enabled by default. `snapshot_view()` displays the initial graph
and every state following a modifying Action. Pass `snapshot=False` to disable
this behavior.

## Pipelines

A `Pipeline` composes Tasks into a larger compiler or verification flow. Like a
Task, it is intentionally thin: its `execute()` method is custom Python, not a
second control-flow language.

```python
from loom.engine import Pipeline, PipelineResult


class CompilerPipeline(Pipeline):
    def __init__(self, *, import_task, folding_task, lowering_task):
        super().__init__(name="compile")
        self.import_task = import_task
        self.folding_task = folding_task
        self.lowering_task = lowering_task

    def execute(self, model, *, workflow="compile") -> PipelineResult:
        if workflow == "compile":
            graph = self.import_task.execute(model).output
            self.folding_task.execute(graph)
            result = self.lowering_task.execute(graph)
            return PipelineResult(output=graph, modified=result.modified)

        if workflow == "lower-only":
            graph = model  # caller provides an already imported graph
            result = self.lowering_task.execute(graph)
            return PipelineResult(output=graph, modified=result.modified)

        raise ValueError(f"unknown workflow {workflow!r}")
```

This structure supports multiple explicit workflows, feedback, and starting at
any Task. Pipeline configuration belongs in its constructor; the caller
selects a workflow with `execute(input, workflow="...")`.

Every Pipeline returns:

```python
PipelineResult(output=..., modified=False)
```

## Contracts and verification

A `Contract` is a human-defined acceptance check for one named Pipeline
workflow. It does not implement compiler work. Instead, it executes the
configured workflow and evaluates its `PipelineResult`.

This is the intended separation for agent-assisted development:

```text
agent-owned Task + trusted Tasks
             ↓
       trusted Pipeline workflow
             ↓
            Contract
             ↓
ContractResult: pass/fail, feedback, metrics
```

The candidate Task is configured into the Pipeline by trusted orchestration;
the Contract only selects a workflow and judges its outcome.

```python
from loom.engine import Contract, ContractResult, PipelineResult


class TargetSchemaContract(Contract):
    def __init__(self, *, pipeline, schema):
        super().__init__(
            name="valid-target-ir",
            pipeline=pipeline,
            workflow="lower",
        )
        self.schema = schema

    def evaluate(self, pipeline_result: PipelineResult) -> ContractResult:
        report = self.schema.validate(pipeline_result.output)
        return ContractResult(
            passed=report.ok,
            output=pipeline_result.output,
            failures=[str(issue) for issue in report.errors],
            metrics={"error_count": len(report.errors)},
        )
```

`ContractResult` has four fields:

```python
ContractResult(
    passed=True,       # actual bool
    output=...,        # optional workflow result or diagnostic artifact
    failures=[...],    # structured or textual feedback
    metrics={...},     # measurements such as area, latency, or error counts
)
```

Use `Verifier` to run every Contract and retain all feedback rather than
stopping at the first normal failure:

```python
from loom.engine import Verifier

verifier = Verifier([source_contract, target_contract, semantic_contract])
results = verifier.verify(model)

if Verifier.passed(results):
    print("candidate accepted")
else:
    for name, result in results.items():
        for failure in result.failures:
            print(f"{name}: {failure}")
```

The verifier catches an exception raised by one Contract, converts it into a
failed `ContractResult`, and continues with the other Contracts. This gives a
development loop the complete set of actionable failures for the next revision.

## End-to-end lowering example

`examples/lowering` demonstrates the full model with a Keras network:

```text
Keras model
   ↓ import workflow
NN-IR
   ↓ lower workflow: FixedPointTask
OP-IR
   ↓ schema Contracts
Verifier result
```

`LowerDenseAction` uses `Processor` to replace each `Dense` node with
`MatMul → BiasAdd → Activation`. The `FixedPointTask` lowers one match per
iteration because adjacent Dense patterns overlap. The `LoweringPipeline`
provides two workflows:

- `import`: build and return the NN-IR;
- `lower`: import, validate NN-IR, lower it, then validate OP-IR.

Two `SchemaContract` instances verify the source and target workflows, and a
`Verifier` aggregates both results.

Run it from the repository root:

```bash
python examples/lowering/main.py
```

Other runnable examples:

```bash
python examples/validation/main.py
python examples/processor_tests/main.py
```

## Public API

```python
from loom.ir import (
    CheckSchemaAction,
    IRSchema,
    Issue,
    Severity,
    ValidationReport,
)

from loom.engine import (
    Action,
    ActionFn,
    ActionResult,
    Contract,
    ContractResult,
    DfsMatchPolicy,
    FixedPointTask,
    IsoMatchPolicy,
    Pipeline,
    PipelineResult,
    Processor,
    Task,
    TaskResult,
    Verifier,
)
```

`loom` also re-exports the basic IR schema types (`IRSchema`, `Issue`,
`Severity`, and `ValidationReport`) for small programs.
