"""Agent-facing pipeline execution boundaries.

This module deliberately contains no model-provider integration. An adapter
such as Pydantic AI can implement :class:`Agent` and use :class:`AgentContext`
as its dependency object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Type

from ..core import Pipeline, PipelineResult, Task

TaskFactory = Callable[[Pipeline], Task]
TaskFactories = Mapping[str, TaskFactory]
class PipelineRunner:
    """Capability for executing a human-owned pipeline.

    The pipeline factory and input factory remain private. Callers can submit
    implementations only for the explicitly editable task names.
    """

    def __init__(self, pipeline_class: Type[Pipeline], *,
                 editable_tasks: set[str] | frozenset[str],
                 pipeline_kwargs: Mapping[str, Any] | None = None,
                 draft: Mapping[str, str | Path] | None = None,
                 input_factory: Callable[[Any], Any] | None = None):
        if not isinstance(pipeline_class, type) or not issubclass(pipeline_class, Pipeline):
            raise TypeError("pipeline_class must be a Pipeline subclass.")
        self._pipeline_class = pipeline_class
        self._pipeline_kwargs = dict(pipeline_kwargs or {})
        self._editable_tasks = frozenset(editable_tasks)
        paths = {
            name: Path(path) for name, path in (draft or {}).items()
        }
        unknown_paths = set(paths) - self._editable_tasks
        if unknown_paths:
            raise ValueError(
                f"Task paths are not editable: {sorted(unknown_paths)!r}"
            )
        self._draft = MappingProxyType(paths)
        self._input_factory = input_factory or (lambda value: value)

    @property
    def editable_tasks(self) -> frozenset[str]:
        return self._editable_tasks

    @property
    def draft(self) -> Mapping[str, Path]:
        return self._draft

    def run(self, input: Any, task_factories: TaskFactories) -> PipelineResult:
        """Execute with candidate implementations for editable task slots."""
        unknown = set(task_factories) - self._editable_tasks
        if unknown:
            raise ValueError(f"Tasks are not editable: {sorted(unknown)!r}")
        pipeline = self._pipeline_class(
            task_factories=task_factories,
            **self._pipeline_kwargs,
        )
        result = pipeline.execute(self._input_factory(input))
        if not isinstance(result, PipelineResult):
            raise TypeError("pipeline must return PipelineResult")
        return result


@dataclass(frozen=True)
class AgentContext:
    """Dependencies exposed to an agent implementation."""

    runner: PipelineRunner
    input: Any

    @property
    def editable_tasks(self) -> frozenset[str]:
        return self.runner.editable_tasks

    @property
    def draft(self) -> Mapping[str, Path]:
        return self.runner.draft

    def run(self, task_factories: TaskFactories) -> PipelineResult:
        return self.runner.run(self.input, task_factories)


class Agent(Protocol):
    """Protocol suitable for a Pydantic AI-backed task reviser."""

    def initial_tasks(self, context: AgentContext) -> TaskFactories:
        ...

    def revise_tasks(self, context: AgentContext, *,
                     failure: Exception) -> TaskFactories:
        ...


class AgentLoop:
    """Run candidate task implementations until the pipeline succeeds."""

    def __init__(self, runner: PipelineRunner, *, max_attempts: int = 10):
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) \
                or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")
        self.runner = runner
        self.max_attempts = max_attempts

    def run(self, input: Any, agent: Agent) -> PipelineResult:
        context = AgentContext(self.runner, input)
        task_factories = agent.initial_tasks(context)
        last_failure: Exception | None = None

        for _ in range(self.max_attempts):
            try:
                return context.run(task_factories)
            except Exception as failure:
                # ContractException carries the PipelineResult and contract
                # results; other exceptions are task/runtime errors.
                last_failure = failure
                task_factories = agent.revise_tasks(
                    context, failure=failure
                )

        raise RuntimeError(
            f"agent loop exceeded max_attempts ({self.max_attempts}); "
            f"last failure: {last_failure}"
        ) from last_failure
