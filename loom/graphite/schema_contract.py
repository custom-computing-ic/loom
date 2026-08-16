"""Contracts for checking graphs against graph schemas."""

from ..core import Contract, ContractResult, Result
from .schema import GraphSchema


class GraphSchemaContract(Contract):
    """Accept results whose output is valid for one :class:`GraphSchema`."""

    def __init__(self, schema: GraphSchema, *, name: str | None = None):
        super().__init__(name=name or f"valid-{schema.name}")
        self.schema = schema

    def evaluate(self, result: Result) -> ContractResult:
        report = self.schema.validate(result.output)
        return ContractResult(
            passed=report.ok,
            output=result.output,
            failures=[str(issue) for issue in report.errors],
            metrics={"errors": len(report.errors), "issues": len(report.issues)},
        )
