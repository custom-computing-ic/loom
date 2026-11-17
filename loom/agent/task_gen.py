"""Model-backed generation of editable Loom task modules."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..core import Pipeline, Task
from .agent import AgentContext
from .runner import TaskFactories, TaskFactory
from .provider import Provider


class TaskSource(BaseModel):
    """One complete task module produced by a model."""

    class_name: str = Field(description="Name of the Task subclass in source.")
    source: str = Field(description="Complete Python module, without Markdown fences.")


class TaskGen:
    """Generate one task implementation through a generic provider."""

    def __init__(self, *, provider: Provider, task: str, instructions: str,
                 mode_resume: bool = False):
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider instance.")
        self.provider = provider
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string.")
        self.task = task.strip()
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a non-empty string.")
        self.instructions = instructions.strip()
        if not isinstance(mode_resume, bool):
            raise TypeError("mode_resume must be a boolean.")
        self.mode_resume = mode_resume

    def initial_tasks(self, context: AgentContext) -> TaskFactories:
        path = self._selected_path(context)
        if self.mode_resume:
            return {self.task: self._factory(
                path, self._task_class_name(path.read_text(encoding="utf-8"), path)
            )}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return self.build_tasks(context, source="")

    def revise_tasks(self, context: AgentContext, *,
                     failure: Exception) -> TaskFactories:
        return self.build_tasks(context, failure=failure)

    def build_tasks(self, context: AgentContext, *,
                    failure: Exception | None = None,
                    source: str | None = None) -> TaskFactories:
        path = self._selected_path(context)
        current_source = (
            path.read_text(encoding="utf-8") if source is None and path.exists()
            else source or ""
        )
        proposal = self.provider.submit(
            self._prompt(self.task, current_source, failure), TaskSource
        )
        if not isinstance(proposal, TaskSource):
            raise TypeError("provider returned an invalid task source.")
        compile(proposal.source, str(path), "exec")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(proposal.source, encoding="utf-8")
        return {self.task: self._factory(path, proposal.class_name)}

    @staticmethod
    def _task_class_name(source: str, path: Path) -> str:
        """Find the direct Task subclass in an existing draft module."""
        module = ast.parse(source, filename=str(path))
        classes = [
            node.name
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and any(
                (isinstance(base, ast.Name) and base.id == "Task")
                or (isinstance(base, ast.Attribute) and base.attr == "Task")
                for base in node.bases
            )
        ]
        if len(classes) != 1:
            raise ValueError(
                f"Expected exactly one direct Task subclass in {path}; "
                f"found {classes!r}."
            )
        return classes[0]

    def _selected_path(self, context: AgentContext) -> Path:
        if self.task not in context.editable_tasks:
            raise ValueError(f"Task is not editable: {self.task!r}")
        try:
            return context.draft[self.task]
        except KeyError as error:
            raise KeyError(f"Task has no draft path: {self.task!r}") from error


    @staticmethod
    def _contract_failures(failure: Exception | None) -> Mapping[str, Any] | None:
        """Return per-contract failures when ``failure`` carries them.

        Contract types live outside Loom, so the shape is duck-typed rather
        than imported: ``loom.agent`` stays usable without a contract package
        installed.
        """
        failures = getattr(failure, "failures", None)
        return failures if isinstance(failures, Mapping) and failures else None


    def _prompt(self, name: str, source: str,
                failure: Exception | None) -> str:
        feedback = "No execution feedback is available yet."
        contract_failures = self._contract_failures(failure)
        if contract_failures is not None:
            feedback = "Contract failures:\n" + "\n".join(
                f"- {contract}: {getattr(result, 'failures', result)}"
                for contract, result in contract_failures.items()
            )
        elif failure is not None:
            feedback = f"Task/runtime exception: {type(failure).__name__}: {failure}"
        return (
            f"Task instructions:\n{self.instructions}\n\n"
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
