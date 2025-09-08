from __future__ import annotations

from datetime import datetime
from typing import Type, List

from reasoning_forest.core._config import settings
from .schamas import LLMResponse, T, LLMPriority
from reasoning_forest.core._singleton import SingletonMeta
from ._anthropic import get_anthropic_llm
from ._google import get_google_llm
from ._openai import get_openai_llm
from ._xai import get_xai_llm


class LLM(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.anthropic_llm = get_anthropic_llm(settings.ANTHROPIC_API_KEY)
        self.google_llm = get_google_llm(settings.GOOGLE_AI_API_KEY)
        self.openai_llm = get_openai_llm(settings.OPENAI_API_KEY)
        self.xai_llm = get_xai_llm(settings.XAI_API_KEY)

    def _check_api_key_missing(self, priority: LLMPriority) -> None:
        if getattr(self, f"{priority.provider}_llm") is None:
            raise ValueError(
                f"Cannot use {priority.model}. {priority.provider.upper()}_API_KEY is not set! Please set it in the environment variables."
            )

    def respond(
        self,
        query: str,
        llm_priority: List[LLMPriority],
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        web_search: bool = False,
        google_url_context: bool = False,
        google_web_search_sd: datetime | None = None,
        google_web_search_ed: datetime | None = None,
    ) -> LLMResponse[T]:
        for llmp in llm_priority:
            self._check_api_key_missing(llmp)
            if llmp.provider == "google":
                return self.google_llm.respond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                    url_context=google_url_context,
                    web_search_sd=google_web_search_sd,
                    web_search_ed=google_web_search_ed,
                )
            if llmp.provider == "openai":
                return self.openai_llm.respond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                )
            if llmp.provider == "anthropic":
                return self.anthropic_llm.respond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                )
            if llmp.provider == "xai":
                return self.xai_llm.respond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                )

        raise RuntimeError("Unreachable code path")

    async def arespond(
        self,
        query: str,
        llm_priority: List[LLMPriority],
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        web_search: bool = False,
        google_url_context: bool = False,
        google_web_search_sd: datetime | None = None,
        google_web_search_ed: datetime | None = None,
    ) -> LLMResponse[T]:
        for llmp in llm_priority:
            self._check_api_key_missing(llmp)
            if llmp.provider == "google":
                return await self.google_llm.arespond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                    url_context=google_url_context,
                    web_search_sd=google_web_search_sd,
                    web_search_ed=google_web_search_ed,
                )
            if llmp.provider == "openai":
                return await self.openai_llm.arespond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                )
            if llmp.provider == "anthropic":
                return await self.anthropic_llm.arespond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                )
            if llmp.provider == "xai":
                return await self.xai_llm.arespond(  # type: ignore
                    query=query,
                    model=llmp.model,  # type: ignore
                    response_format=response_format,
                    instructions=instructions,
                    temperature=temperature,
                    web_search=web_search,
                )

        raise RuntimeError("Unreachable code path")
