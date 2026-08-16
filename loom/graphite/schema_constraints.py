"""Validation rules shared by every IR schema."""
from __future__ import annotations

from heterograph import HGraph

from .schema import Issue, Severity


def require_type(graph: HGraph) -> list[Issue]:
    """Require every vertex and edge property map to have ``_type``."""
    issues: list[Issue] = []
    for vertex in graph.vertices:
        props = graph.pmap[vertex]
        if not isinstance(props, dict) or "_type" not in props:
            issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                "missing _type"))
    for edge in graph.edges:
        props = graph.pmap[edge]
        if not isinstance(props, dict) or "_type" not in props:
            issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                "missing _type"))
    return issues


def require_graph_type(graph: HGraph, schema_name: str) -> list[Issue]:
    """Require graph-level ``_type`` to identify the active IR schema."""
    actual = graph.pmap.get("_type")
    if actual != schema_name:
        return [Issue(Severity.ERROR, "graph._type",
                      f"expected {schema_name!r}, got {actual!r}")]
    return []


SCHEMA_CONSTRAINTS = [require_graph_type, require_type]
