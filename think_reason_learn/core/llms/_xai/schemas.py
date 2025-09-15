from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel


xAIChatModel: TypeAlias = Literal["grok-3", "grok-3-mini", "grok-code-fast-1", "grok-4"]


class XAIPriority(BaseModel):
    provider: Literal["xai"] = "xai"
    model: xAIChatModel


class XAIPriorityDict(TypedDict):
    provider: Literal["xai"]
    model: xAIChatModel
