"""Unified interface for using LLMs from OpenAI, Google, Anthropic, and XAI."""

from ._schemas import (
    NotGiven,
    NOT_GIVEN,
    LLMChoice,
    LLMChoiceDict,
    OpenAIChoice,
    OpenAIChoiceDict,
    GoogleChoice,
    GoogleChoiceDict,
    AnthropicChoice,
    AnthropicChoiceDict,
    XAIChoice,
    XAIChoiceDict,
    LLMChoiceModel,
    LLMProvider,
    LLMResponse,
    LLMChatModel,
    OpenAIChatModel,
    GoogleChatModel,
    AnthropicChatModel,
    xAIChatModel,
    TokenCounter,
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
    "OpenAIChoiceDict",
    "GoogleChoice",
    "GoogleChoiceDict",
    "AnthropicChoice",
    "AnthropicChoiceDict",
    "XAIChoice",
    "XAIChoiceDict",
    "TokenCount",
    "TokenCounter",
    "LLMChoiceModel",
    "LLMChoiceDict",
    "LLMResponse",
    "LLMChatModel",
    "LLMProvider",
    "OpenAIChatModel",
    "GoogleChatModel",
    "AnthropicChatModel",
    "xAIChatModel",
    "TokenCounter",
    "llm",
]
