"""Pipeline abstractions for composing Tasks with custom Python logic."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from .result import PipelineResult, TaskResult
from .task import Task


class Pipeline(ABC):
    """Abstract, stateless orchestration of Tasks."""

    name: str
    description: Optional[str] = None

    def __init__(self, *, name: str, description: str | None = None,
                 max_iterations: int = 1000,
                 task_factories: dict[str, Callable[["Pipeline"], Task]] | None = None):
        if not name:
            raise ValueError("Pipeline must have a name.")
        self.name = name
        self.description = description
        self._tasks: dict[str, Task] = {}
        self.task_factories = dict(self.builtin_task_factories())
        if task_factories:
            self.task_factories.update(task_factories)
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) \
                or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        self.max_iterations = max_iterations

    def builtin_task_factories(self) -> dict[str, Callable[["Pipeline"], Task]]:
        """Return the pipeline's default task factories."""
        return {}

    def register(self, name: str) -> Task:
        """Construct and register the task selected by ``name``."""
        if name not in self.task_factories:
            raise KeyError(f"No task factory registered for {name!r}.")
        task = self.task_factories[name](self)
        if not isinstance(task, Task):
            raise TypeError(
                f"Task factory {name!r} returned {type(task).__name__}; "
                "task factories must return Task instances."
            )
        if task.name != name.strip().lower():
            raise ValueError(
                f"Task factory {name!r} created task named {task.name!r}."
            )
        return task

    def _register_task(self, task: Task) -> None:
        """Register one task instance, rejecting duplicate task names."""
        if not isinstance(task, Task):
            raise TypeError("pipeline tasks must be Task instances.")
        if task.name in self._tasks:
            raise ValueError(
                f"Pipeline {self.name!r} already has a task named {task.name!r}."
            )
        self._tasks[task.name] = task

    @abstractmethod
    def execute(self, input: Any) -> PipelineResult:
        """Run the pipeline's custom orchestration."""
        pass

    def __call__(self, input: Any) -> PipelineResult:
        return self.execute(input)

    def repeat(self, task: Task, value: Any, *,
               max_iterations: int | None = None) -> TaskResult:
        """Run one Task repeatedly until it reports no modification."""
        if not isinstance(task, Task):
            raise TypeError("repeat() requires a Task instance.")
        limit = self.max_iterations if max_iterations is None else max_iterations
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        results: list[TaskResult] = []
        ever_modified = False
        for iteration in range(1, limit + 1):
            result = task.execute(value)
            if not isinstance(result, TaskResult):
                raise RuntimeError(
                    f"task {task.name} returned {type(result).__name__}; "
                    "Task.execute() must return TaskResult"
                )
            results.append(result)
            ever_modified |= result.modified
            if not result.modified:
                return TaskResult(
                    output=result.output,
                    modified=ever_modified,
                    metadata={
                        **result.metadata,
                        "iterations": iteration,
                        "converged": True,
                        "iteration_results": results,
                    },
                )

        return TaskResult(
            output=results[-1].output,
            modified=ever_modified,
            metadata={
                **results[-1].metadata,
                "iterations": limit,
                "converged": False,
                "iteration_results": results,
            },
        )
