"""Provider adapters, one per wire protocol."""

from __future__ import annotations

from typing import Final

from .._spec import WireProtocol
from ._base import Adapter, Answers, GenerateRequest, ProviderError, Reply
from .anthropic import AnthropicAdapter
from .openai_compat import OpenAICompatAdapter
from .openai_decisions import OpenAIDecisionsAdapter
from .systemone import SystemOneAdapter

ADAPTERS: Final[dict[WireProtocol, Adapter]] = {
    "openai": OpenAICompatAdapter(),
    "anthropic": AnthropicAdapter(),
    "systemone": SystemOneAdapter(),
    "openai-decisions": OpenAIDecisionsAdapter(),
}
"""The adapter of each protocol. ``think_reason_learn.testing`` adds ``"fake"``."""

__all__ = [
    "ADAPTERS",
    "Adapter",
    "Answers",
    "GenerateRequest",
    "ProviderError",
    "Reply",
]
