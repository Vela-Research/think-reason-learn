from __future__ import annotations

from typing import Type

from .._response_schema import LLMResponse, T
from .schamas import OpenAIChatModel
from reasoning_forest.core._singleton import SingletonMeta


class OpenAILLM(metaclass=SingletonMeta):
    def __init__(self, api_key: str) -> None: ...

    def respond(
        self,
        query: str,
        model: OpenAIChatModel,
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        web_search: bool = False,
    ) -> LLMResponse[T]:
        raise NotImplementedError("Sync response not supported for OpenAI")

    async def arespond(
        self,
        query: str,
        model: OpenAIChatModel,
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        web_search: bool = False,
    ) -> LLMResponse[T]:
        raise NotImplementedError("Async response not supported for OpenAI")
