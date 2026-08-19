"""Loom: domain-neutral task orchestration."""

from .core import (
    Pipeline,
    PipelineResult,
    Result,
    Task,
    TaskResult,
)

__all__ = [
    "Result",
    "Task",
    "TaskResult",
    "Pipeline",
    "PipelineResult",
]
