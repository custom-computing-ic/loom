"""Pipeline abstractions for composing Tasks with custom Python logic."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PipelineResult:
    """Result returned by a Pipeline."""

    output: Any = None
    modified: bool = False

    def __post_init__(self) -> None:
        if type(self.modified) is not bool:
            raise TypeError(
                "PipelineResult.modified must be an actual bool, "
                f"got {type(self.modified).__name__}."
            )


class Pipeline(ABC):
    """Abstract, stateless orchestration of Tasks."""

    name: str
    description: Optional[str] = None

    def __init__(self, *, name: str, description: str | None = None):
        if not name:
            raise ValueError("Pipeline must have a name.")
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, input: Any, *, workflow: str = "default") -> PipelineResult:
        """Run the pipeline's custom workflow."""
        pass

    def __call__(self, input: Any, *, workflow: str = "default") -> PipelineResult:
        return self.execute(input, workflow=workflow)
