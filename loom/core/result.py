"""Shared result types for Loom execution boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Result:
    """Domain-neutral result produced by a Task or Pipeline."""

    output: Any = None
    modified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.modified) is not bool:
            raise TypeError(
                "Result.modified must be an actual bool, "
                f"got {type(self.modified).__name__}."
            )


@dataclass(frozen=True)
class TaskResult(Result):
    """Result produced by one Task execution."""


@dataclass(frozen=True)
class PipelineResult(Result):
    """Result produced by one Pipeline execution."""

    task_results: dict[str, TaskResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all(isinstance(result, TaskResult)
                   for result in self.task_results.values()):
            raise TypeError("PipelineResult.task_results must contain TaskResult values.")
