"""Model-backed generation of editable Loom task modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from pydantic import BaseModel, Field

from ..core import ContractException, Pipeline, Task
from .agent import AgentContext
from .runner import TaskFactories, TaskFactory
from .provider import Provider


class TaskSource(BaseModel):
    """One complete task module produced by a model."""

    class_name: str = Field(description="Name of the Task subclass in source.")
    source: str = Field(description="Complete Python module, without Markdown fences.")


class TaskGen:
    """Generate selected task implementations through a generic provider."""

    def __init__(self, *, provider: Provider, tasks: Iterable[str]):
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider instance.")
        self.provider = provider
        self.tasks = frozenset(tasks)
        if not self.tasks:
            raise ValueError("tasks must contain at least one task name.")

    def initial_tasks(self, context: AgentContext) -> TaskFactories:
        return self.build_tasks(context)

    def revise_tasks(self, context: AgentContext, *,
                     failure: Exception) -> TaskFactories:
        return self.build_tasks(context, failure=failure)

    def build_tasks(self, context: AgentContext, *,
                    failure: Exception | None = None) -> TaskFactories:
        selected = self._selected_paths(context)
        factories: dict[str, TaskFactory] = {}
        for name, path in selected.items():
            current_source = path.read_text(encoding="utf-8") if path.exists() else ""
            proposal = self.provider.generate(
                self._prompt(name, current_source, failure), TaskSource
            )
            if not isinstance(proposal, TaskSource):
                raise TypeError("provider returned an invalid task source.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(proposal.source, encoding="utf-8")
            factories[name] = self._factory(path, proposal.class_name)
        return factories

    def _selected_paths(self, context: AgentContext) -> dict[str, Path]:
        unknown = self.tasks - context.editable_tasks
        if unknown:
            raise ValueError(f"Tasks are not editable: {sorted(unknown)!r}")
        missing = self.tasks - set(context.draft)
        if missing:
            raise KeyError(f"Tasks have no draft path: {sorted(missing)!r}")
        return {name: context.draft[name] for name in self.tasks}

    @staticmethod
    def _prompt(name: str, source: str, failure: Exception | None) -> str:
        feedback = "No execution feedback is available yet."
        if isinstance(failure, ContractException):
            feedback = "Contract failures:\n" + "\n".join(
                f"- {contract}: {result.failures}"
                for contract, result in failure.failures.items()
            )
        elif failure is not None:
            feedback = f"Task/runtime exception: {type(failure).__name__}: {failure}"
        return (
            f"Editable task name: {name}\n\n"
            f"Current draft module:\n```python\n{source}\n```\n\n"
            f"Execution feedback:\n{feedback}\n\n"
            "Return a complete replacement module. Do not use Markdown fences "
            "inside the source field."
        )

    @staticmethod
    def _factory(path: Path, class_name: str) -> TaskFactory:
        def factory(pipeline: Pipeline) -> Task:
            module_name = f"loom_draft_{path.stem}_{uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load task module from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            task_type = getattr(module, class_name)
            task = task_type(pipeline)
            if not isinstance(task, Task):
                raise TypeError(
                    f"{class_name!r} from {path} did not create a Task instance."
                )
            return task
        return factory
