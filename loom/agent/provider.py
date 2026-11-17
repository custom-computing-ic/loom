"""Provider interfaces and model-specific Loom adapters."""

from __future__ import annotations

import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel

OutputT = TypeVar("OutputT", bound=BaseModel)


class Provider(ABC):
    """Reusable model client with structured submission support."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self._model = self._create_model()

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the reusable Pydantic AI model/client."""
        raise NotImplementedError

    def submit(self, prompt: str, output_type: type[OutputT]) -> OutputT:
        """Submit a prompt and parse the response as ``output_type``."""
        try:
            from pydantic_ai import Agent
        except ImportError as error:
            raise ImportError(
                "Provider requires Pydantic AI. Install it with "
                "`pip install loom[agent]`."
            ) from error
        agent = Agent(
            self._model,
            output_type=output_type,
            system_prompt=(
                "You write one complete Python Task module for Loom. "
                "Return only the structured response. The task must use its "
                "supplied pipeline constructor argument, preserve its assigned "
                "task name, and modify no files itself."
            ),
        )
        return agent.run_sync(prompt).output


class AzureProvider(Provider):
    """Azure OpenAI provider.

    Required config keys: ``deployment_name``, ``api_key``, and ``base_url`` and
     ``api_version``
    """

    def _create_model(self) -> Any:
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        deployment_name = self._required("deployment_name")
        api_key = self._required("api_key")
        _base_url = self.config.get("base_url")
        base_url = osp.join(_base_url, deployment_name)
        api_version = self._required("api_version")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key="unused",
            default_headers={"api-key": api_key},
            default_query={
                "api-version": api_version
            },
        )
        return OpenAIChatModel(
            deployment_name,
            provider=OpenAIProvider(openai_client=client),
        )

    def _required(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AzureProvider requires config key {key!r}.")
        return value.strip()


class ClaudeProvider(Provider):
    """Anthropic Claude provider.

    Required config keys: ``model`` and ``api_key``.
    """

    def _create_model(self) -> Any:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        model = self.config.get("model")
        api_key = self.config.get("api_key")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("ClaudeProvider requires config key 'model'.")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("ClaudeProvider requires config key 'api_key'.")
        return AnthropicModel(
            model.strip(),
            provider=AnthropicProvider(api_key=api_key.strip()),
        )


class LlamaCppProvider(Provider):
    """Local llama.cpp server provider.

    Required config keys: ``host`` and ``port``. The server must expose its
    OpenAI-compatible API under ``/v1``.
    """

    def _create_model(self) -> Any:
        from pydantic_ai.models.openai import AsyncOpenAI, OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        host = self.config.get("host")
        port = self.config.get("port")
        if not isinstance(host, str) or not host.strip():
            raise ValueError("LlamaCppProvider requires config key 'host'.")
        if not isinstance(port, int) or isinstance(port, bool) or port <= 0:
            raise ValueError("LlamaCppProvider requires a positive integer 'port'.")
        client = AsyncOpenAI(
            base_url=f"http://{host.strip()}:{port}/v1",
            api_key="unused",
        )
        return OpenAIChatModel(
            "local",
            provider=OpenAIProvider(openai_client=client),
        )
