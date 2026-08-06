from abc import ABC, abstractmethod
from typing import Callable, Optional, Any

class Rule(ABC):
    """Abstract graph transformation whose ``apply`` method returns ``bool``."""
    """
    Abstract rewrite rule.

    A rule must implement:
        apply(g) -> bool

    Returns:
        True  if the graph was modified
        False otherwise
    """

    # debug metadata
    name: str
    description: Optional[str] = None

    def __init__(self, *, name: str, description: str | None = None):
        if not name:
            raise ValueError("Rule must have a name.")
        self.name = name        
        self.description = description    

    @abstractmethod
    def apply(self, g) -> bool:
        pass



class RuleFn(Rule):
    """Adapt a callable with signature ``fn(graph) -> bool`` into a Rule."""
    """
    Wraps a function into a Rule.

    The function must have signature:

        fn(g) -> bool
    """

    def __init__(
        self,
        fn: Callable[[Any], bool],
        *,
        name: str,
        description: str | None = None,
    ):
        super().__init__(name=name, description=description)
        if not callable(fn):
            raise TypeError("RuleFn requires a callable.")
        self.fn = fn

    def apply(self, g) -> bool:
        return bool(self.fn(g))
