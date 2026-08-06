"""Schema for the primitive-operation IR produced by lowering."""

"""OP-IR schema for primitive operations produced by lowering."""

from typing import Literal

from heterograph import HGraph
from pydantic import BaseModel, Field

from loom.ir import IRSchema, Issue, Severity


class OpProps(BaseModel):
    """Metadata shared by primitive operation vertices."""
    type_: Literal["MatMul", "BiasAdd", "Activation"] = Field(alias="_type")
    name: str | None = None
    shape: list[int | None] | None = None
    dtype: str | None = None
    activation: str | None = None


class ValueProps(BaseModel):
    """Metadata for input, constant, and output values."""
    type_: Literal["Input", "Weight", "Bias", "Output"] = Field(alias="_type")
    name: str | None = None
    shape: list[int | None] | None = None
    dtype: str | None = None


class OperandEdgeProps(BaseModel):
    type_: Literal["data", "parameter"] = Field(alias="_type")
    index: int = Field(ge=0)


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


OP_IR = IRSchema(
    name="OP-IR",
    vertex={
        "Input": ValueProps,
        "Weight": ValueProps,
        "Bias": ValueProps,
        "MatMul": OpProps,
        "BiasAdd": OpProps,
        "Activation": OpProps,
        "Output": ValueProps,
    },
    edge={"data": OperandEdgeProps, "parameter": OperandEdgeProps},
    rules=[rule_acyclic],
)
