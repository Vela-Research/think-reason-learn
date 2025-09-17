"""Unified interface for using LLMs from OpenAI, Google, Anthropic, and XAI."""

from ._schemas import (
    NotGiven,
    NOT_GIVEN,
    LLMChoice,
    LLMChoiceDict,
    OpenAIChoice,
    GoogleChoice,
    AnthropicChoice,
    XAIChoice,
    LLMChoiceModel,
    LLMProvider,
    LLMResponse,
    LLMChatModel,
    OpenAIChatModel,
    GoogleChatModel,
    AnthropicChatModel,
    xAIChatModel,
    TokenCount,
)
from ._ask import LLM


llm = LLM()
"""An instance of the singleton LLM class."""

__all__ = [
    "NotGiven",
    "NOT_GIVEN",
    "LLM",
    "LLMChoice",
    "OpenAIChoice",
    "GoogleChoice",
    "AnthropicChoice",
    "XAIChoice",
    "LLMChoiceModel",
    "LLMChoiceDict",
    "LLMResponse",
    "LLMChatModel",
    "LLMProvider",
    "OpenAIChatModel",
    "GoogleChatModel",
    "AnthropicChatModel",
    "xAIChatModel",
    "TokenCount",
    "llm",
]
