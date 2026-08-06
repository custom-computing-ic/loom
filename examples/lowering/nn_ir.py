"""Schema for the layer-level neural-network IR used by the example."""

"""NN-IR schema for the layer-level Keras lowering example."""

from typing import Literal

from heterograph import HGraph
from pydantic import BaseModel, Field

from loom.ir import IRSchema, Issue, Severity


class TensorProps(BaseModel):
    """Common tensor metadata used by inputs, parameters, and outputs."""
    type_: Literal["Tensor"] = Field(alias="_type")
    shape: list[int | None] | None = None
    dtype: str | None = None
    name: str | None = None


class InputProps(TensorProps):
    type_: Literal["Input"] = Field(alias="_type")


class DenseProps(BaseModel):
    """Metadata for a dense layer before primitive lowering."""
    type_: Literal["Dense"] = Field(alias="_type")
    name: str | None = None
    shape: list[int | None] | None = None
    dtype: str | None = None
    in_features: int = Field(gt=0)
    out_features: int = Field(gt=0)
    activation: str | None = None


class ParameterProps(TensorProps):
    type_: Literal["Parameter"] = Field(alias="_type")
    name: str | None = None


class OutputProps(TensorProps):
    type_: Literal["Output"] = Field(alias="_type")


class DataEdgeProps(BaseModel):
    type_: Literal["data"] = Field(alias="_type")
    index: int = Field(ge=0)


class ParameterEdgeProps(BaseModel):
    type_: Literal["parameter"] = Field(alias="_type")
    index: int = Field(ge=1)


def rule_acyclic(graph: HGraph) -> list[Issue]:
    color = {v: 0 for v in graph.vertices}

    def visit(v: int) -> bool:
        color[v] = 1
        for successor in graph.out_vx(v):
            if color[successor] == 1 or (color[successor] == 0 and visit(successor)):
                return True
        color[v] = 2
        return False

    return ([Issue(Severity.ERROR, "graph", "cycle detected")]
            if any(color[v] == 0 and visit(v) for v in graph.vertices) else [])


class NNIRSchema(IRSchema):
    """NN-IR schema with role-specific graph visualization."""
    def style(self, graph: HGraph) -> None:
        palette = {
            "Input": ("#43A047", "white"),
            "Output": ("#E53935", "white"),
            "Parameter": ("#FFB300", "black"),
            "Dense": ("#1E88E5", "white"),
        }

        def label(g: HGraph, vertex: int) -> str:
            props = g.pmap[vertex]
            name = props.get("name") or props["_type"]
            shape = props.get("shape")
            dtype = props.get("dtype")
            details = [str(name), f"\\<{props['_type']}\\>" ]
            if shape is not None:
                details.append(str(tuple(shape)))
            if dtype:
                details.append(str(dtype))
            return "{" + "|".join(details) + "}"

        def fill(g: HGraph, vertex: int) -> str:
            return palette.get(g.pmap[vertex]["_type"], ("#1E88E5", "white"))[0]

        def font(g: HGraph, vertex: int) -> str:
            return palette.get(g.pmap[vertex]["_type"], ("#1E88E5", "white"))[1]

        edge_palette = {
            "data": "#424242",
            "parameter": "#7B1FA2",
        }

        graph.style = {
            "rankdir": "LR",
            "fontname": "Helvetica",
            "fontsize": "14",
            "labelloc": "t",
            "label": graph.pmap.get("name", self.name),
        }
        graph.vstyle = {
            "shape": "Mrecord",
            "style": "filled,bold",
            "fontname": "Helvetica",
            "fontsize": "9",
            "penwidth": "1.3",
            "label": label,
            "fillcolor": fill,
            "fontcolor": font,
        }
        graph.estyle = {
            "arrowhead": "open",
            "fontname": "Helvetica",
            "fontsize": "8",
            "penwidth": "1.2",
            "label": lambda g, edge: str(g.pmap[edge].get("index", "")),
            "color": lambda g, edge: edge_palette.get(
                g.pmap[edge].get("_type", "data"), "#616161"
            ),
            "fontcolor": lambda g, edge: edge_palette.get(
                g.pmap[edge].get("_type", "data"), "#616161"
            ),
        }


NN_IR = NNIRSchema(
    name="NN-IR",
    vertex={
        "Input": InputProps,
        "Dense": DenseProps,
        "Parameter": ParameterProps,
        "Output": OutputProps,
    },
    edge={"data": DataEdgeProps, "parameter": ParameterEdgeProps},
    rules=[rule_acyclic],
)
