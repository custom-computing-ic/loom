"""Domain-neutral task orchestration."""

from .result import PipelineResult, Result, TaskResult
from .task import Task
from .pipeline import Pipeline

__all__ = [
    "Result",
    "TaskResult",
    "PipelineResult",
    "Task",
    "Pipeline",
]
