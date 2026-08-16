"""Task abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from .result import TaskResult

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Task(ABC):
    """Abstract, stateless unit of domain work."""

    name: str
    description: Optional[str] = None

    def __init__(self, pipeline: "Pipeline", *, name: str | None = None,
                 description: str | None = None):
        resolved_name = name or self.__class__.__name__
        if not isinstance(resolved_name, str) or not resolved_name.strip():
            raise ValueError("Task name must be a non-empty string.")
        self.name = resolved_name.strip().lower()
        self.description = description
        self.pipeline = pipeline
        pipeline._register_task(self)

    @abstractmethod
    def execute(self, input: Any) -> TaskResult:
        """Run the task's execution strategy."""
        pass

    def __call__(self, input: Any) -> TaskResult:
        return self.execute(input)
