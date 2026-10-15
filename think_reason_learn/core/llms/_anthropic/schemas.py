from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel
from anthropic.types import Model
from anthropic import Omit as NotGiven, omit as NOT_GIVEN


AnthropicChatModel: TypeAlias = Model


class AnthropicChoice(BaseModel):
    """An LLM from Anthropic."""

    provider: Literal["anthropic"] = "anthropic"
    model: AnthropicChatModel


class AnthropicChoiceDict(TypedDict):
    """An LLM from Anthropic."""

    provider: Literal["anthropic"]
    model: AnthropicChatModel


__all__ = [
    "NOT_GIVEN",
    "AnthropicChatModel",
    "AnthropicChoice",
    "AnthropicChoiceDict",
    "NotGiven",
]
