"""Reusable, stateless units of compiler work."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ActionResult:
    """Result returned by an Action."""

    output: Any = None
    modified: bool = False

    def __post_init__(self) -> None:
        if type(self.modified) is not bool:
            raise TypeError(
                "ActionResult.modified must be an actual bool, "
                f"got {type(self.modified).__name__}."
            )


class Action(ABC):
    """Abstract, stateless unit of compiler work."""

    name: str
    description: Optional[str] = None

    def __init__(self, *, name: str, description: str | None = None):
        if not name:
            raise ValueError("Action must have a name.")
        self.name = name
        self.description = description

    @abstractmethod
    def apply(self, g) -> ActionResult:
        """Apply the action and return its result."""
        pass


class ActionFn(Action):
    """Adapt a callable with signature ``fn(graph) -> ActionResult``."""

    def __init__(
        self,
        fn: Callable[[Any], ActionResult],
        *,
        name: str,
        description: str | None = None,
    ):
        super().__init__(name=name, description=description)
        if not callable(fn):
            raise TypeError("ActionFn requires a callable.")
        self.fn = fn

    def apply(self, g) -> ActionResult:
        result = self.fn(g)
        if not isinstance(result, ActionResult):
            raise TypeError(
                "ActionFn callable must return ActionResult, "
                f"got {type(result).__name__}."
            )
        return result
