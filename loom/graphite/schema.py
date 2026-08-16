"""Schema validation framework for Heterograph-based intermediate representations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Type

from pydantic import BaseModel, ValidationError
from heterograph import HGraph


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class Issue:
    severity: Severity
    location: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.location}: {self.message}"


@dataclass
class ValidationReport:
    schema: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == Severity.ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        if not self.issues:
            return f"OK  {self.schema}: 0 issues"
        errors = len(self.errors)
        warnings = len(self.issues) - errors
        head = f"FAIL {self.schema}: {errors} error(s), {warnings} warning(s)"
        return "\n".join([head] + [f"  {issue}" for issue in self.issues])


Rule = Callable[[HGraph], list[Issue]]


@dataclass
class GraphSchema:
    """Declarative metadata schema, validation, construction, and styling."""
    name: str
    graph: Type[BaseModel] | None = None
    vertex: dict[str, Type[BaseModel]] = field(default_factory=dict)
    edge: dict[str, Type[BaseModel]] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)

    def style(self, graph: HGraph) -> None:
        """Apply the default visualization style for this IR.

        Subclasses may override this method to provide IR-specific labels,
        colours, or layout options. Styling is deliberately separate from
        validation so visualization metadata never affects IR correctness.
        """
        graph.style = {
            "rankdir": "LR",
            "label": self.name,
            "labelloc": "t",
        }
        graph.vstyle = {
            "label": lambda g, vertex: g.pmap[vertex].get("_type", "?"),
        }

    def new_graph(self) -> HGraph:
        """Create an empty graph identified by this schema and style it."""
        graph = HGraph()
        graph.pmap["_type"] = self.name
        self.style(graph)
        return graph

    def new_vertex(self, graph: HGraph, _type: str, **props: object) -> int:
        """Validate and add a typed vertex, returning its host ID."""
        data = self._vertex_data(_type, props)
        vertex = graph.add_vx()
        graph.pmap[vertex] = data
        return vertex

    def new_edge(self, graph: HGraph, source: int, target: int,
                 _type: str, **props: object) -> tuple[int, int]:
        """Validate and add a typed directed edge."""
        data = self._edge_data(_type, props)
        graph.add_edge(source, target)
        edge = (source, target)
        graph.pmap[edge] = data
        return edge

    def update_vertex(self, graph: HGraph, vertex: int, **changes: object) -> None:
        """Validate and replace selected properties on an existing vertex."""
        current = dict(graph.pmap[vertex])
        current.update(changes)
        graph.pmap[vertex] = self._vertex_data(current["_type"], current)

    def update_edge(self, graph: HGraph, edge: tuple[int, int], **changes: object) -> None:
        """Validate and replace selected properties on an existing edge."""
        current = dict(graph.pmap[edge])
        current.update(changes)
        graph.pmap[edge] = self._edge_data(current["_type"], current)

    def _vertex_data(self, _type: str, props: dict[str, object]) -> dict:
        """Validate properties for one vertex type and serialize aliases."""
        if _type not in self.vertex:
            raise KeyError(f"unknown vertex schema {_type!r}")
        values = dict(props)
        values["_type"] = _type
        return self.vertex[_type].model_validate(
            values, extra="forbid"
        ).model_dump(by_alias=True)

    def _edge_data(self, _type: str, props: dict[str, object]) -> dict:
        """Validate properties for one edge type and serialize aliases."""
        if _type not in self.edge:
            raise KeyError(f"unknown edge schema {_type!r}")
        values = dict(props)
        values["_type"] = _type
        return self.edge[_type].model_validate(
            values, extra="forbid"
        ).model_dump(by_alias=True)

    def validate(self, graph: HGraph) -> ValidationReport:
        """Return a report covering graph, item metadata, and structural rules."""
        issues: list[Issue] = []

        from .schema_constraints import require_graph_type, require_type
        issues += require_graph_type(graph, self.name)
        issues += require_type(graph)

        if self.graph is not None:
            issues += _check_model(
                self.graph,
                graph.pmap,
                "graph",
            )

        for vertex in graph.vertices:
            data = graph.pmap[vertex]
            if not self.vertex:
                continue
            if not isinstance(data, dict):
                issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                    "vertex properties must be a dict"))
                continue
            kind = data.get("_type")
            if kind is None:
                issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                    "missing _type"))
                continue
            model = self.vertex.get(kind)
            if model is None:
                issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                    f"unknown _type {kind!r} in schema {self.name}"))
                continue
            issues += _check_model(model, data, f"vertex {vertex}")

        if self.edge:
            for edge in graph.edges:
                data = graph.pmap[edge]
                if not isinstance(data, dict):
                    issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                        "edge properties must be a dict"))
                    continue
                kind = data.get("_type")
                if kind is None:
                    issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                        "missing _type"))
                    continue
                model = self.edge.get(kind)
                if model is None:
                    issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                        f"unknown _type {kind!r} in schema {self.name}"))
                    continue
                issues += _check_model(model, data, f"edge {edge}")

        for rule in self.rules:
            issues += rule(graph)
        return ValidationReport(self.name, issues)


def _check_model(model: Type[BaseModel], data: object, location: str) -> list[Issue]:
    if not isinstance(data, dict):
        return [Issue(Severity.ERROR, location,
                      f"properties must be a dict, got {type(data).__name__}")]
    try:
        model.model_validate(data, extra="forbid")
    except ValidationError as error:
        return [Issue(Severity.ERROR, location,
                      f"{'.'.join(str(p) for p in item['loc'])}: {item['msg']}")
                for item in error.errors()]
    return []
