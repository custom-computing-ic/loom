"""Smoke-test the configured OpenAI and Anthropic provider credentials.

Run after sourcing the project .env:

    python examples/agent/test_providers.py

The script makes one small structured-output request to each provider and
prints only success or the error type/message; it never prints credentials.
"""

from pathlib import Path
import os
import sys

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parents[2]))

from loom.agent import PydanticAIProvider


class ProviderReply(BaseModel):
    message: str


def test_provider(name: str, provider: PydanticAIProvider) -> None:
    try:
        reply = provider.generate(
            "Reply with a short greeting.", ProviderReply
        )
        print(f"{name}: OK ({reply.message})")
    except Exception as error:
        print(f"{name}: FAILED ({type(error).__name__}: {error})")


if __name__ == "__main__":
    test_provider(
        "OpenAI",
        PydanticAIProvider(
            provider="openai",
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
        ),
    )
    test_provider(
        "Anthropic",
        PydanticAIProvider(provider="claude", model="claude-3-5-haiku-latest"),
    )
