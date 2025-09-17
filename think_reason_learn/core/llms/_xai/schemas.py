from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel


xAIChatModel: TypeAlias = (
    str | Literal["grok-3", "grok-3-mini", "grok-code-fast-1", "grok-4"]
)


class XAIChoice(BaseModel):
    """An LLM from XAI."""

    provider: Literal["xai"] = "xai"
    model: xAIChatModel


class XAIChoiceDict(TypedDict):
    """An LLM from XAI."""

    provider: Literal["xai"]
    model: xAIChatModel
