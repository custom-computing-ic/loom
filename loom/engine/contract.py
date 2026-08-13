"""Contracts for verifying pipeline workflows."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from .pipeline import Pipeline, PipelineResult


@dataclass(frozen=True)
class ContractResult:
    """Result of verifying one pipeline workflow."""

    passed: bool
    output: Any = None
    failures: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError(
                "ContractResult.passed must be an actual bool, "
                f"got {type(self.passed).__name__}."
            )


class Contract(ABC):
    """Human-defined acceptance criteria for one pipeline workflow."""

    def __init__(
        self,
        *,
        name: str,
        pipeline: Pipeline,
        workflow: str = "default",
    ):
        if not name:
            raise ValueError("Contract must have a name.")
        if not isinstance(pipeline, Pipeline):
            raise TypeError("Contract requires a Pipeline instance.")
        self.name = name
        self.pipeline = pipeline
        self.workflow = workflow

    def verify(self, input: Any) -> ContractResult:
        """Run the configured workflow and evaluate its result."""
        pipeline_result = self.pipeline.execute(input, workflow=self.workflow)
        if not isinstance(pipeline_result, PipelineResult):
            raise RuntimeError(
                f"pipeline returned {type(pipeline_result).__name__}; "
                "Pipeline.execute() must return PipelineResult"
            )
        return self.evaluate(pipeline_result)

    @abstractmethod
    def evaluate(self, pipeline_result: PipelineResult) -> ContractResult:
        """Evaluate the result produced by the configured workflow."""
        pass


class Verifier:
    """Run every Contract and aggregate the individual results."""

    def __init__(self, contracts: Iterable[Contract]):
        self.contracts = list(contracts)
        if not all(isinstance(contract, Contract) for contract in self.contracts):
            raise TypeError("contracts must contain only Contract instances.")

    def verify(self, input: Any) -> dict[str, ContractResult]:
        """Run all contracts, retaining failures from each contract."""
        results: dict[str, ContractResult] = {}
        for contract in self.contracts:
            try:
                results[contract.name] = contract.verify(input)
            except Exception as error:
                results[contract.name] = ContractResult(
                    passed=False,
                    failures=[f"{type(error).__name__}: {error}"],
                )
        return results

    @staticmethod
    def passed(results: dict[str, ContractResult]) -> bool:
        """Return whether every Contract passed."""
        return all(result.passed for result in results.values())
