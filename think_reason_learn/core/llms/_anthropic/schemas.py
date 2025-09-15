from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel
from anthropic.types import Model
from anthropic._types import NOT_GIVEN as NOT_GIVEN, NotGiven as NotGiven


AnthropicChatModel: TypeAlias = Model


class AnthropicChoice(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    model: AnthropicChatModel


class AnthropicChoiceDict(TypedDict):
    provider: Literal["anthropic"]
    model: AnthropicChatModel
