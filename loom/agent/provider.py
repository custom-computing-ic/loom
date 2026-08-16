"""Provider abstractions for model-backed Loom agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel


class Provider(ABC):
    """Generic interface for communicating with a language model."""

    @abstractmethod
    def generate(self, prompt: str, output_type: Type[BaseModel]) -> BaseModel:
        """Generate one structured response for ``prompt``."""
        pass


class PydanticAIProvider(Provider):
    """Provider backed by Pydantic AI.

    Model identifiers and provider-specific routing are kept here so callers
    such as :class:`~loom.agent.task_gen.TaskGen` need only use ``generate``.
    """

    def __init__(self, *, provider: str, model: str):
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")
        self.provider = provider.strip().lower()
        self.model = model.strip()

    @property
    def model_name(self) -> str:
        """Return the Pydantic AI model identifier for this configuration."""
        prefixes = {
            "openai": "openai-responses",
            "claude": "anthropic",
            "anthropic": "anthropic",
        }
        try:
            prefix = prefixes[self.provider]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Pydantic AI provider {self.provider!r}."
            ) from error
        return f"{prefix}:{self.model}"

    def generate(self, prompt: str, output_type: Type[BaseModel]) -> BaseModel:
        try:
            from pydantic_ai import Agent
        except ImportError as error:
            raise ImportError(
                "PydanticAIProvider requires Pydantic AI. Install it with "
                "`pip install loom[agent]`."
            ) from error

        agent = Agent(
            self.model_name,
            output_type=output_type,
            system_prompt=(
                "You write one complete Python Task module for Loom. "
                "Return only the structured response. The task must use its "
                "supplied pipeline constructor argument, preserve its assigned "
                "task name, and modify no files itself."
            ),
        )
        return agent.run_sync(prompt).output
