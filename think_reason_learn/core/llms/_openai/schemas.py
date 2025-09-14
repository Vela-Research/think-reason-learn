from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel
from openai.types import ChatModel
from openai._types import NOT_GIVEN as NOT_GIVEN, NotGiven as NotGiven

OpenAIChatModel: TypeAlias = ChatModel


class OpenAIChoice(BaseModel):
    provider: Literal["openai"] = "openai"
    model: OpenAIChatModel


class OpenAIChoiceDict(TypedDict):
    provider: Literal["openai"]
    model: OpenAIChatModel
