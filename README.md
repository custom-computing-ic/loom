# GraphProcessor

`GraphProcessor` implements **pattern matching** and **single-pass graph
rewriting** on top of `HGraph` using AQL (Heterograph's graph query
language).

It supports:

-   Subgraph matching via:
    -   `iso` (graph-tool subgraph isomorphism)
-   Optional deduplication of matches
-   Single-pass, disjoint rewrites
-   Controlled rewiring of external edges
-   Post-processing callbacks

`GraphProcessor` operates directly on `HGraph`, which is a high-level
wrapper around `graph-tool` with persistent vertex IDs and property
maps.

------------------------------------------------------------------------

# Quick Start

``` python
from heterograph import HGraph
from heterograph_ex import GraphProcessor

g = HGraph()

# Build a simple graph
a, b, c = g.add_vx(3)
g.add_edge(a, b)
g.add_edge(b, c)

gp = GraphProcessor()

result = gp.run(g, find="x => y")

print(result["matches"])
```

------------------------------------------------------------------------

# Overview

``` python
gp = GraphProcessor(deduplicate=True)

result = gp.run(
    g,
    find="AQL pattern",
    where=optional_filter,
    rewrite="AQL replacement",
    post=optional_callback
)
```

Return value:

``` python
{
    "matches": [...],
    "modified": bool
}
```

------------------------------------------------------------------------

# 1. Pattern Matching

## AQL `find` Pattern

Example:

``` python
find = "a => b => c"
```

Matches any chain of three connected vertices.

Bindings map **pattern IDs → host vertex IDs**:

``` python
[
  {"a": 0, "b": 1, "c": 2},
]
```

------------------------------------------------------------------------

## Matching Algorithm

-   Exact subgraph match
-   Induced matching (any two nodes in match )
-   Recommended for rewriting

------------------------------------------------------------------------

## `where` Filter

Optional semantic filter:

``` python
def where(g, a, b):
    return g.pmap[a]["type"] == "Conv" and g.num_out_vx(b) == 1
```

If omitted → all structural matches are accepted.

------------------------------------------------------------------------

# 2. Deduplication

When `deduplicate=True`, matches with the same **set of host vertices**
are merged.

Deduplication key:

``` python
frozenset(match.values())
```

Disable if needed:

``` python
GraphProcessor(deduplicate=False)
```

------------------------------------------------------------------------

# 3. Rewrite Semantics

Rewriting is **single-pass and in-place**.

Rules:

-   Matches must be **disjoint**
-   Overlapping matches raise an error
-   Rewrite is applied once (not fixed-point)

------------------------------------------------------------------------

## Basic Rewrite

``` python
find = "a => b => c"
rewrite = "a => c"
```

Semantics:

-   `b` deleted
-   LHS-only edges removed
-   RHS-only edges added
-   Shared nodes preserved

------------------------------------------------------------------------

## LHS / RHS Sets

-   Preserved = L ∩ R
-   Deleted = L − R
-   Created = R − L

Edges:

-   RHS − LHS → added
-   LHS − RHS → removed
-   Shared → preserved

------------------------------------------------------------------------

# 4. Rewiring External Edges

Syntax inside rewrite pattern:

    x {rewire:y}
    x {rewire_in:y}
    x {rewire_out:y}

Assume `x` deleted, `y` preserved.

-   `rewire` → redirect both directions
-   `rewire_in` → incoming only
-   `rewire_out` → outgoing only

------------------------------------------------------------------------

## Example: Bypass Node

    find:    a => b => c
    rewrite: a => c
             b {rewire:c}

External edges to/from `b` are redirected to `c`.

------------------------------------------------------------------------

## Internal Edge Safety

If rewiring implies a new internal edge between preserved nodes, that
edge must exist in RHS or rewrite fails.

------------------------------------------------------------------------

# 5. Post Callback

``` python
def post(g, a, c):
    g.pmap[a]["optimized"] = True
    return True
```

Called after structural additions but before deletions.

Return `True` if metadata changed.

------------------------------------------------------------------------

# 6. Rewrite Order

For each match:

1.  Validate match
2.  Create new nodes
3.  Add RHS-only edges
4.  Perform rewiring
5.  Call `post`
6.  Remove LHS-only edges
7.  Remove deleted nodes

------------------------------------------------------------------------

# 7. Fixed-Point Driver

``` python
gp = GraphProcessor(mode="iso")

while True:
    result = gp.run(g, find=..., rewrite=...)
    if not result["modified"]:
        break
```

------------------------------------------------------------------------

# Summary

`GraphProcessor` provides:

-   AQL-based pattern matching
-   Safe, disjoint rewriting
-   Controlled rewiring
-   Deterministic single-pass semantics

Suitable for graph lowering, optimisation passes, and rule-based
transformations.