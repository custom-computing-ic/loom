"""Agent-facing pipeline execution boundaries.

This module deliberately contains no model-provider integration. An adapter
such as Pydantic AI can implement :class:`Agent` and use :class:`AgentContext`
as its dependency object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Callable, Mapping, Protocol

from ..core import Pipeline, PipelineResult, Task
from .runner import PipelineRunner

TaskFactory = Callable[[Pipeline], Task]
TaskFactories = Mapping[str, TaskFactory]
logger = logging.getLogger(__name__)


def _failure_feedback(failure: Exception) -> str:
    """Format contract failures as the feedback supplied to a revising agent."""
    failures = getattr(failure, "failures", None)
    if isinstance(failures, Mapping):
        return "Contract failures:\n" + "\n".join(
            f"- {contract}: {result.failures}"
            for contract, result in failures.items()
        )
    return f"Task/runtime exception: {type(failure).__name__}: {failure}"


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
        last_failure: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if last_failure is None:
                    logger.info("Agent: generating initial task candidate.")
                    task_factories = agent.initial_tasks(context)
                else:
                    logger.info("Agent feedback:\n%s",
                                _failure_feedback(last_failure))
                    logger.info("Agent: generating revised task candidate.")
                    task_factories = agent.revise_tasks(
                        context, failure=last_failure
                    )
                logger.info("Agent: running candidate (attempt %d/%d).",
                            attempt, self.max_attempts)
                result = context.run(task_factories)
                logger.info("Agent: candidate succeeded on attempt %d.", attempt)
                return result
            except Exception as failure:
                # ContractException carries the PipelineResult and contract
                # results; other exceptions are task/runtime errors.
                last_failure = failure
                logger.warning(
                    "Agent: attempt %d failed (%s: %s).",
                    attempt, type(failure).__name__, failure,
                )
                if attempt == self.max_attempts:
                    break

        raise RuntimeError(
            f"agent loop exceeded max_attempts ({self.max_attempts}); "
            f"last failure: {last_failure}"
        ) from last_failure
