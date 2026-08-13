"""Task abstractions and reusable execution templates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .action import Action, ActionResult


@dataclass(frozen=True)
class TaskResult:
    """Result returned by a Task."""

    output: Any = None
    modified: bool = False

    def __post_init__(self) -> None:
        if type(self.modified) is not bool:
            raise TypeError(
                "TaskResult.modified must be an actual bool, "
                f"got {type(self.modified).__name__}."
            )


class Task(ABC):
    """Abstract, stateless unit of compiler orchestration."""

    name: str
    description: Optional[str] = None

    def __init__(self, *, name: str, description: str | None = None):
        if not name:
            raise ValueError("Task must have a name.")
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, graph) -> TaskResult:
        """Run the task's execution strategy."""
        pass

    def __call__(self, graph) -> TaskResult:
        return self.execute(graph)


class FixedPointTask(Task):
    """Execute pre, iterative, and post actions to a fixed point."""

    def __init__(
        self,
        *,
        name: str = "fixed-point",
        description: str | None = None,
        pre: Iterable[Action] = (),
        iterative: Iterable[Action] = (),
        post: Iterable[Action] = (),
        max_iterations: int | None = None,
        snapshot: bool = True,
    ):
        super().__init__(name=name, description=description)
        self.pre = self._validate_actions(pre, "pre")
        self.iterative = self._validate_actions(iterative, "iterative")
        self.post = self._validate_actions(post, "post")
        if max_iterations is not None and max_iterations < 0:
            raise ValueError("max_iterations must be non-negative or None.")
        self.max_iterations = max_iterations
        self.snapshot = snapshot
        self.webview = None
        self._snapshot_count = 0
        if snapshot:
            from heterograph import WebView
            self.webview = WebView()

    @staticmethod
    def _validate_actions(actions: Iterable[Action], phase: str) -> list[Action]:
        actions = list(actions)
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError(f"{phase} must contain only Action instances.")
        return actions

    def _snapshot(self, graph, title: str) -> None:
        if self.webview is not None:
            self.webview.add_graph(graph, title=title)
            self._snapshot_count += 1

    def snapshot_view(self):
        if self.webview is None:
            raise RuntimeError("snapshot_view() requires snapshot=True.")
        return self.webview.run()

    def _apply_actions(self, actions: list[Action], graph, phase: str,
                       verbose: bool) -> tuple[bool, list[ActionResult]]:
        modified = False
        results = []
        for action in actions:
            result = action.apply(graph)
            if not isinstance(result, ActionResult):
                raise RuntimeError(
                    f"action {action.name} returned {type(result).__name__}; "
                    "Action.apply() must return ActionResult"
                )
            if result.modified:
                modified = True
                if verbose:
                    description = "" if action.description is None else f" ({action.description})"
                    print(f"  - Action {action.name}{description} modified graph")
                self._snapshot(graph, f"{phase} / {action.name}")
            results.append(result)
        return modified, results

    def execute(self, graph, *, max_iterations: int | None = None,
                verbose: bool = False) -> TaskResult:
        """Execute all phases and repeat iterative actions until stable."""
        limit = self.max_iterations if max_iterations is None else max_iterations
        if limit is not None and limit < 0:
            raise ValueError("max_iterations must be non-negative or None.")

        iteration = 0
        ever_modified = False
        action_results = []
        self._snapshot(graph, "initial")

        modified, results = self._apply_actions(self.pre, graph, "pre", verbose)
        ever_modified |= modified
        action_results.extend(("pre", result) for result in results)

        while True:
            if limit is not None and iteration >= limit:
                break
            iteration += 1
            modified_this_round, results = self._apply_actions(
                self.iterative, graph, "iterative", verbose
            )
            ever_modified |= modified_this_round
            action_results.extend(("iterative", result) for result in results)
            if not modified_this_round:
                break

        modified, results = self._apply_actions(self.post, graph, "post", verbose)
        ever_modified |= modified
        action_results.extend(("post", result) for result in results)
        return TaskResult(
            output={"iterations": iteration, "action_results": action_results},
            modified=ever_modified,
        )
