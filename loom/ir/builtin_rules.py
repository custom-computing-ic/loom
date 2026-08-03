"""Validation rules shared by every IR schema."""
from __future__ import annotations

from heterograph import HGraph

from .ir_schema import Issue, Severity


def require_type(graph: HGraph, namespace: str) -> list[Issue]:
    """Require every namespaced vertex and edge property map to have ``_type``."""
    issues: list[Issue] = []
    for vertex in graph.vertices:
        props = graph.pmap[vertex].get(namespace, {})
        if not isinstance(props, dict) or "_type" not in props:
            issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                f"missing {namespace}._type"))
    for edge in graph.edges:
        props = graph.pmap[edge].get(namespace, {})
        if not isinstance(props, dict) or "_type" not in props:
            issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                f"missing {namespace}._type"))
    return issues


def require_ir_anchor(graph: HGraph, schema_name: str, namespace: str) -> list[Issue]:
    """Require graph-level metadata to identify the active IR schema."""
    anchor = graph.pmap.get("_ir")
    if not isinstance(anchor, dict):
        return [Issue(Severity.ERROR, "graph", "missing _ir schema anchor")]
    issues: list[Issue] = []
    if anchor.get("schema") != schema_name:
        issues.append(Issue(Severity.ERROR, "graph._ir.schema",
                            f"expected {schema_name!r}, got {anchor.get('schema')!r}"))
    if anchor.get("namespace") != namespace:
        issues.append(Issue(Severity.ERROR, "graph._ir.namespace",
                            f"expected {namespace!r}, got {anchor.get('namespace')!r}"))
    return issues


BUILTIN_RULES = [require_ir_anchor, require_type]
