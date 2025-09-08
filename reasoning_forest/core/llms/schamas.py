from __future__ import annotations

from typing import TypeAlias, Literal, Dict, Tuple

from pydantic import BaseModel, model_validator

from ._anthropic.schamas import AnthropicChatModel
from ._google.schamas import GoogleChatModel
from ._openai.schamas import OpenAIChatModel
from ._xai.schamas import xAIChatModel
from ._response_schema import LLMResponse as LLMResponseSchema, T

LLMChatModel: TypeAlias = Literal[
    AnthropicChatModel,
    GoogleChatModel,
    OpenAIChatModel,
    xAIChatModel,
]

# To avoid getting this from _response_schema
LLMResponse: TypeAlias = LLMResponseSchema[T]

LLMProvider: TypeAlias = Literal["anthropic", "google", "openai", "xai"]


class LLMPriority(BaseModel):
    provider: LLMProvider
    model: LLMChatModel

    @model_validator(mode="after")
    def validate_model_provider(self) -> LLMPriority:
        provider_model: Dict[LLMProvider, Tuple[LLMChatModel, ...]] = {
            "anthropic": AnthropicChatModel.__args__,
            "google": GoogleChatModel.__args__,
            "openai": OpenAIChatModel.__args__,
            "xai": xAIChatModel.__args__,
        }
        if self.provider not in provider_model:
            raise ValueError(
                f"Invalid model provider: {self.provider}. Valid providers are: {list(provider_model.keys())}"
            )
        if self.model not in provider_model[self.provider]:
            raise ValueError(
                f"Invalid model: {self.model} for {self.provider}. Valid models are: {list(provider_model[self.provider])}"
            )
        return self
