from typing import Type, Any, Dict
import logging
from typing import cast


from pydantic import BaseModel, ValidationError
from google import genai
from google.genai import types as gtypes

from ..schemas import LLMResponse, T, GoogleChoice
from .schemas import GoogleChatModel
from think_reason_learn.core._singleton import SingletonMeta

logger = logging.getLogger(__name__)


class GeminiLLM(metaclass=SingletonMeta):
    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)

    def _process_kwargs(
        self,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process Google kwargs and query."""
        return {
            k: v or None
            for k, v in kwargs.items()
            if k in gtypes.GenerateContentConfig.model_fields
        }

    def _build_config(
        self,
        response_format: Type[T],
        instructions: str | None,
        temperature: float | None,
        kwargs: Dict[str, Any],
    ) -> gtypes.GenerateContentConfig:
        if response_format is str:
            response_schema, response_mime_type = None, None
        else:
            response_schema, response_mime_type = response_format, "application/json"

        return gtypes.GenerateContentConfig(
            temperature=temperature,
            system_instruction=instructions,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
            response_logprobs=True,
            **kwargs,
        )

    def _parse_response(
        self,
        response: gtypes.GenerateContentResponse,
        model: GoogleChatModel,
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
                        provider_model=GoogleChoice(model=model),
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
                        provider_model=GoogleChoice(model=model),
                    )
                except ValidationError:
                    pass

        return LLMResponse(
            response=cast(T, text),
            logprobs=log_probs,
            total_tokens=total_tokens,
            provider_model=GoogleChoice(model=model),
        )

    def respond_sync(
        self,
        query: str,
        model: GoogleChatModel,
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)
        config = self._build_config(
            response_format=response_format,
            instructions=instructions,
            temperature=temperature,
            kwargs=kwargs,
        )

        try:
            response = self.client.models.generate_content(  # type: ignore
                model=model,
                contents=kwargs.get("contents", query),
                config=config,
            )
            return self._parse_response(response, model, response_format)
        except Exception as e:
            logger.warning(f"Error responding with Google: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    async def respond(
        self,
        query: str,
        model: GoogleChatModel,
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)
        config = self._build_config(
            response_format=response_format,
            instructions=instructions,
            temperature=temperature,
            kwargs=kwargs,
        )

        try:
            response = await self.client.aio.models.generate_content(  # type: ignore
                model=model,
                contents=kwargs.get("contents", query),
                config=config,
            )
            return self._parse_response(response, model, response_format)
        except Exception as e:
            logger.warning(f"Error responding with Google: {e}", exc_info=True)
            if raise_:
                raise e
            return None


def get_google_llm(api_key: str) -> GeminiLLM | None:
    return GeminiLLM(api_key) if api_key else None
