"""Protected execution boundary for agent-controlled task implementations."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Type

from ..core import Pipeline, PipelineResult, Task

TaskFactory = Callable[[Pipeline], Task]
TaskFactories = Mapping[str, TaskFactory]


class PipelineRunner:
    """Execute a human-owned pipeline with selected editable task slots."""

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
        paths = {name: Path(path) for name, path in (draft or {}).items()}
        unknown_paths = set(paths) - self._editable_tasks
        if unknown_paths:
            raise ValueError(f"Task paths are not editable: {sorted(unknown_paths)!r}")
        self._draft = MappingProxyType(paths)
        self._input_factory = input_factory or (lambda value: value)

    @property
    def editable_tasks(self) -> frozenset[str]:
        return self._editable_tasks

    @property
    def draft(self) -> Mapping[str, Path]:
        return self._draft

    def run(self, input: Any, task_factories: TaskFactories) -> PipelineResult:
        unknown = set(task_factories) - self._editable_tasks
        if unknown:
            raise ValueError(f"Tasks are not editable: {sorted(unknown)!r}")
        pipeline = self._pipeline_class(task_factories=task_factories, **self._pipeline_kwargs)
        result = pipeline.execute(self._input_factory(input))
        if not isinstance(result, PipelineResult):
            raise TypeError("pipeline must return PipelineResult")
        return result
