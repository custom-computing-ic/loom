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
class IRSchema:
    """Declarative schema and graph-invariant checks for one IR."""
    name: str
    namespace: str
    graph_schema: Type[BaseModel] | None = None
    vertex_schema: dict[str, Type[BaseModel]] = field(default_factory=dict)
    edge_schema: dict[str, Type[BaseModel]] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)

    def new_graph(self) -> HGraph:
        graph = HGraph()
        graph.pmap["_ir"] = {
            "schema": self.name,
            "namespace": self.namespace,
        }
        graph.pmap[self.namespace] = {}
        return graph

    def new_vertex(self, graph: HGraph, kind: str, **props: object) -> int:
        data = self._vertex_data(kind, props)
        vertex = graph.add_vx()
        graph.pmap[vertex] = {self.namespace: data}
        return vertex

    def new_edge(self, graph: HGraph, source: int, target: int,
                 kind: str, **props: object) -> tuple[int, int]:
        data = self._edge_data(kind, props)
        graph.add_edge(source, target)
        edge = (source, target)
        graph.pmap[edge] = {self.namespace: data}
        return edge

    def update_vertex(self, graph: HGraph, vertex: int, **changes: object) -> None:
        current = dict(graph.pmap[vertex].get(self.namespace, {}))
        current.update(changes)
        graph.pmap[vertex][self.namespace] = self._vertex_data(current["_type"], current)

    def update_edge(self, graph: HGraph, edge: tuple[int, int], **changes: object) -> None:
        current = dict(graph.pmap[edge].get(self.namespace, {}))
        current.update(changes)
        graph.pmap[edge][self.namespace] = self._edge_data(current["_type"], current)

    def _vertex_data(self, kind: str, props: dict[str, object]) -> dict:
        if kind not in self.vertex_schema:
            raise KeyError(f"unknown vertex schema {kind!r}")
        values = dict(props)
        values["_type"] = kind
        return self.vertex_schema[kind](**values).model_dump(by_alias=True)

    def _edge_data(self, kind: str, props: dict[str, object]) -> dict:
        if kind not in self.edge_schema:
            raise KeyError(f"unknown edge schema {kind!r}")
        values = dict(props)
        values["_type"] = kind
        return self.edge_schema[kind](**values).model_dump(by_alias=True)

    def validate(self, graph: HGraph) -> ValidationReport:
        """Validate graph, vertex, and edge metadata plus structural rules."""
        issues: list[Issue] = []

        from .builtin_rules import require_ir_anchor, require_type
        issues += require_ir_anchor(graph, self.name, self.namespace)
        issues += require_type(graph, self.namespace)

        if self.graph_schema is not None:
            issues += _check_model(
                self.graph_schema,
                graph.pmap.get(self.namespace, {}),
                "graph",
                self.namespace,
            )

        for vertex in graph.vertices:
            data = graph.pmap[vertex]
            data = data.get(self.namespace, {}) if isinstance(data, dict) else data
            if not self.vertex_schema:
                continue
            if not isinstance(data, dict):
                issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                    f"{self.namespace}.* must be a dict"))
                continue
            kind = data.get("_type")
            if kind is None:
                issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                    f"missing {self.namespace}._type"))
                continue
            model = self.vertex_schema.get(kind)
            if model is None:
                issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                    f"unknown _type {kind!r} in schema {self.name}"))
                continue
            issues += _check_model(model, data, f"vertex {vertex}", self.namespace)

        if self.edge_schema:
            for edge in graph.edges:
                data = graph.pmap[edge]
                data = data.get(self.namespace, {}) if isinstance(data, dict) else data
                if not isinstance(data, dict):
                    issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                        f"{self.namespace}.* must be a dict"))
                    continue
                kind = data.get("_type")
                if kind is None:
                    issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                        f"missing {self.namespace}._type"))
                    continue
                model = self.edge_schema.get(kind)
                if model is None:
                    issues.append(Issue(Severity.ERROR, f"edge {edge}",
                                        f"unknown _type {kind!r} in schema {self.name}"))
                    continue
                issues += _check_model(model, data, f"edge {edge}", self.namespace)

        for rule in self.rules:
            issues += rule(graph)
        return ValidationReport(self.name, issues)


def _check_model(model: Type[BaseModel], data: object, location: str, namespace: str) -> list[Issue]:
    if not isinstance(data, dict):
        return [Issue(Severity.ERROR, location,
                      f"{namespace}.* must be a dict, got {type(data).__name__}")]
    try:
        model(**data)
    except ValidationError as error:
        return [Issue(Severity.ERROR, location,
                      f"{namespace}.{'.'.join(str(p) for p in item['loc'])}: {item['msg']}")
                for item in error.errors()]
    return []
