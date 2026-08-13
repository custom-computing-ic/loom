"""Graph matching, rewriting, and fixed-point action execution."""

from .processor import Processor
from .match_policy import DfsMatchPolicy, IsoMatchPolicy
from .action import Action, ActionFn, ActionResult
from .task import FixedPointTask, Task, TaskResult
from .pipeline import Pipeline, PipelineResult
from .contract import Contract, ContractResult, Verifier

__all__ = [
    "Processor",
    "DfsMatchPolicy",
    "IsoMatchPolicy",
    "Action",
    "ActionFn",
    "ActionResult",
    "Task",
    "TaskResult",
    "FixedPointTask",
    "Pipeline",
    "PipelineResult",
    "Contract",
    "ContractResult",
    "Verifier",
]
