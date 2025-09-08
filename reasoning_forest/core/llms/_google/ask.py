from __future__ import annotations

from datetime import datetime, timezone
from typing import Type
from typing import cast

from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types as gtypes

from .._response_schema import LLMResponse, T
from .schamas import GoogleChatModel
from reasoning_forest.core._singleton import SingletonMeta


class GeminiLLM(metaclass=SingletonMeta):
    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)

    @classmethod
    def get_ggrounding_tool(
        cls,
        time_range_start: datetime | None = None,
        time_range_end: datetime | None = None,
    ) -> gtypes.GoogleSearch:
        time_range_start = time_range_start or datetime.min
        time_range_end = time_range_end or datetime.max

        time_range_start = time_range_start.replace(
            microsecond=0, tzinfo=time_range_start.tzinfo or timezone.utc
        )

        time_range_end = time_range_end.replace(
            microsecond=0, tzinfo=time_range_end.tzinfo or timezone.utc
        )

        return gtypes.GoogleSearch(
            time_range_filter=gtypes.Interval(
                start_time=time_range_start, end_time=time_range_end
            )
        )

    def _build_config(
        self,
        response_format: Type[T],
        web_search: bool,
        url_context: bool,
        web_search_sd: datetime | None,
        web_search_ed: datetime | None,
        instructions: str | None,
        temperature: float | None,
    ) -> gtypes.GenerateContentConfig:
        if (web_search or url_context) and response_format is not str:
            raise ValueError(
                "Response format must be a string if web search or url context is enabled."
            )

        grounding = (
            self.get_ggrounding_tool(web_search_sd, web_search_ed)
            if web_search
            else None
        )
        url_ctx = gtypes.UrlContext() if url_context else None

        tools = (
            [gtypes.Tool(google_search=grounding, url_context=url_ctx)]
            if grounding and url_ctx
            else (
                [gtypes.Tool(google_search=grounding)]
                if grounding
                else [gtypes.Tool(url_context=url_ctx)] if url_ctx else []
            )
        )

        if response_format is str:
            response_schema, response_mime_type = None, None
        else:
            response_schema, response_mime_type = response_format, "application/json"

        return gtypes.GenerateContentConfig(
            tools=tools,
            temperature=temperature,
            system_instruction=instructions,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
            response_logprobs=True,
        )

    def _parse_response(
        self,
        response: gtypes.GenerateContentResponse,
        response_format: Type[T],
    ) -> LLMResponse[T]:
        total_tokens = (
            response.usage_metadata.total_token_count
            if response.usage_metadata
            else None
        )
        parsed, text, log_probs = None, "", []

        if candidates := response.candidates:
            if log_prob_res := candidates[0].logprobs_result:
                log_probs = [
                    (lp.token, lp.log_probability)
                    for lp in log_prob_res.chosen_candidates or []
                    if lp.token is not None
                ]

            if issubclass(response_format, BaseModel):
                if parsed := response.parsed:
                    return LLMResponse(
                        response=parsed,  # type: ignore
                        logprobs=log_probs,
                        total_tokens=total_tokens,
                    )

            if content := candidates[0].content:
                if parts := content.parts:
                    if part := parts[0]:
                        text = part.text or ""

            if text and issubclass(response_format, BaseModel):
                try:
                    return LLMResponse(
                        response=response_format.model_validate_json(text),
                        logprobs=log_probs,
                        total_tokens=total_tokens,
                    )
                except ValidationError:
                    pass

        return cast(
            LLMResponse[T],
            LLMResponse(response=text, logprobs=log_probs, total_tokens=total_tokens),
        )

    def respond(
        self,
        query: str,
        model: GoogleChatModel,
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        web_search: bool = False,
        url_context: bool = False,
        web_search_sd: datetime | None = None,
        web_search_ed: datetime | None = None,
    ) -> LLMResponse[T]:
        """
        Respond to a query using Gemini.

        Args:
            query (str): The query to respond to.
            model (GeminiChatModel): The model to use.
            response_format (Type[T]): The response format to use.
            instructions (str | None): The instructions to use.
            temperature (float | None): The temperature to use.
            web_search (bool): Whether to use web search.
            url_context (bool): Whether to use url context.
            web_search_sd (datetime | None): The start time of the web search.
            web_search_ed (datetime | None): The end time of the web search.

        Returns:
            LLMResponse[T]: The response from the Gemini model.
        """
        config = self._build_config(
            response_format=response_format,
            web_search=web_search,
            url_context=url_context,
            web_search_sd=web_search_sd,
            web_search_ed=web_search_ed,
            instructions=instructions,
            temperature=temperature,
        )

        response = self.client.models.generate_content(  # type: ignore
            model=model, contents=query, config=config
        )

        return self._parse_response(response, response_format)

    async def arespond(
        self,
        query: str,
        model: GoogleChatModel,
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        web_search: bool = False,
        url_context: bool = False,
        web_search_sd: datetime | None = None,
        web_search_ed: datetime | None = None,
    ) -> LLMResponse[T]:
        """
        Async respond to a query using Gemini.

        Args:
            query (str): The query to respond to.
            model (GeminiChatModel): The model to use.
            response_format (Type[T]): The response format to use.
            instructions (str | None): The instructions to use.
            temperature (float | None): The temperature to use.
            web_search (bool): Whether to use web search.
            url_context (bool): Whether to use url context.
            web_search_sd (datetime | None): The start time of the web search.
            web_search_ed (datetime | None): The end time of the web search.

        Returns:
            LLMResponse[T]: The response from the Gemini model.
        """
        config = self._build_config(
            response_format=response_format,
            web_search=web_search,
            url_context=url_context,
            web_search_sd=web_search_sd,
            web_search_ed=web_search_ed,
            instructions=instructions,
            temperature=temperature,
        )

        response = await self.client.aio.models.generate_content(  # type: ignore
            model=model, contents=query, config=config
        )

        return self._parse_response(response, response_format)
