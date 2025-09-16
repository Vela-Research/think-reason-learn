from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel


xAIChatModel: TypeAlias = Literal["grok-3", "grok-3-mini", "grok-code-fast-1", "grok-4"]


class XAIChoice(BaseModel):
    provider: Literal["xai"] = "xai"
    model: xAIChatModel


class XAIChoiceDict(TypedDict):
    provider: Literal["xai"]
    model: xAIChatModel
