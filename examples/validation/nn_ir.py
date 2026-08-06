"""Validation schema for the example NN-IR."""
from typing import Literal

from pydantic import BaseModel, Field
from heterograph import HGraph
from loom.ir import IRSchema, Issue, Severity

class QInt(BaseModel):
    signed: bool
    int_bits: int
    frac_bits: int


class InputProps(BaseModel):
    type_: Literal["Input"] = Field(alias="_type")
    shape: list[int] = Field(min_length=1)
    qint: QInt


class EmbeddingProps(BaseModel):
    type_: Literal["Embedding"] = Field(alias="_type")
    in_features: int = Field(gt=0)
    out_features: int = Field(gt=0)
    has_bias: bool = True
    qint: QInt


class DenseProps(BaseModel):
    type_: Literal["Dense"] = Field(alias="_type")
    in_features: int = Field(gt=0)
    out_features: int = Field(gt=0)
    has_bias: bool = True
    qint: QInt


class AddProps(BaseModel):
    type_: Literal["Add"] = Field(alias="_type")
    qint: QInt


class SumPoolProps(BaseModel):
    type_: Literal["SumPool"] = Field(alias="_type")
    axis: int = 0
    scale_pow2: int = 0
    qint: QInt


class BatchNormProps(BaseModel):
    type_: Literal["BatchNorm"] = Field(alias="_type")
    num_features: int = Field(gt=0)
    epsilon: float = 1e-5
    qint: QInt


class ReLUProps(BaseModel):
    type_: Literal["ReLU"] = Field(alias="_type")
    qint: QInt


class OutputProps(BaseModel):
    type_: Literal["Output"] = Field(alias="_type")
    qint: QInt


class DataEdgeProps(BaseModel):
    type_: Literal["data"] = Field(alias="_type")


ARITY = {
    "Input": (0, 1), "Embedding": (1, 1), "Dense": (1, 1),
    "Add": (2, 1), "SumPool": (1, 1), "BatchNorm": (1, 1),
    "ReLU": (1, 1), "Output": (1, 0),
}


def rule_arity(graph: HGraph) -> list[Issue]:
    issues = []
    for vertex in graph.vertices:
        op = graph.pmap[vertex].get("_type")
        if op not in ARITY:
            continue
        expected_in, expected_out = ARITY[op]
        actual_in, actual_out = len(graph.in_vx(vertex)), len(graph.out_vx(vertex))
        if actual_in != expected_in:
            issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                f"{op} requires {expected_in} inputs, has {actual_in}"))
        if (op == "Output" and actual_out != 0) or (op != "Output" and actual_out < expected_out):
            issues.append(Issue(Severity.ERROR, f"vertex {vertex}",
                                f"{op} has invalid consumer count: {actual_out}"))
    return issues


def rule_acyclic(graph: HGraph) -> list[Issue]:
    color = {vertex: 0 for vertex in graph.vertices}

    def visit(vertex: int) -> bool:
        color[vertex] = 1
        for successor in graph.out_vx(vertex):
            if color[successor] == 1 or (color[successor] == 0 and visit(successor)):
                return True
        color[vertex] = 2
        return False

    if any(color[v] == 0 and visit(v) for v in graph.vertices):
        return [Issue(Severity.ERROR, "graph", "cycle detected")]
    return []


def rule_single_output(graph: HGraph) -> list[Issue]:
    outputs = [v for v in graph.vertices
               if graph.pmap[v].get("_type") == "Output"]
    if len(outputs) == 1:
        return []
    message = "no Output vertex" if not outputs else f"expected 1 Output, found {len(outputs)}"
    return [Issue(Severity.ERROR, "graph", message)]


NN_IR = IRSchema(
    name="NN-IR",
    vertex={
        "Input": InputProps, "Embedding": EmbeddingProps, "Dense": DenseProps,
        "Add": AddProps, "SumPool": SumPoolProps, "BatchNorm": BatchNormProps,
        "ReLU": ReLUProps, "Output": OutputProps,
    },
    edge={"data": DataEdgeProps},
    rules=[rule_arity, rule_acyclic, rule_single_output],
)
