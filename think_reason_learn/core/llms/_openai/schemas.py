from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel
from openai.types import ChatModel
from openai._types import NOT_GIVEN as NOT_GIVEN, NotGiven as NotGiven

OpenAIChatModel: TypeAlias = str | ChatModel


class OpenAIChoice(BaseModel):
    """An LLM from OpenAI."""

    provider: Literal["openai"] = "openai"
    model: OpenAIChatModel


class OpenAIChoiceDict(TypedDict):
    """An LLM from OpenAI."""

    provider: Literal["openai"]
    model: OpenAIChatModel
