from typing import Type, Any, cast, Dict
import logging

from openai import AsyncOpenAI, OpenAI
from openai.types.responses import Response
from pydantic import BaseModel

from .._schemas import LLMResponse, T, OpenAIChoice
from think_reason_learn.core._singleton import SingletonMeta
from .schemas import OpenAIChatModel, NOT_GIVEN, NotGiven


logger = logging.getLogger(__name__)


class OpenAILLM(metaclass=SingletonMeta):
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.aclient = AsyncOpenAI(api_key=api_key)

    def _process_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v if v else (None if v is None else NOT_GIVEN)
            for k, v in kwargs.items()
            if k in self.client.responses.parse.__annotations__
        }

    def respond_sync(
        self,
        model: OpenAIChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)

        try:
            if issubclass(response_format, BaseModel):
                response = self.client.responses.parse(
                    model=model,
                    input=kwargs.get("input", query),
                    instructions=instructions or None,
                    temperature=temperature or None,
                    text_format=response_format,
                    **kwargs,
                )
                return LLMResponse(
                    response=response.output_parsed,
                    logprobs=[],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=OpenAIChoice(model=model),
                )

            response = self.client.responses.create(  # type: ignore
                model=model,
                input=kwargs.get("input", query),
                instructions=instructions or None,
                temperature=temperature or None,
                **kwargs,
            )
            response = cast(Response, response)

            return LLMResponse(
                response=cast(T, response.output_text),
                logprobs=[],
                total_tokens=response.usage.total_tokens if response.usage else None,
                provider_model=OpenAIChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with OpenAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    async def respond(
        self,
        model: OpenAIChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)

        try:
            if issubclass(response_format, BaseModel):
                response = await self.aclient.responses.parse(
                    model=model,
                    input=kwargs.get("input", query),
                    instructions=instructions or None,
                    temperature=temperature or None,
                    text_format=response_format,
                    **kwargs,
                )
                return LLMResponse(
                    response=response.output_parsed,
                    logprobs=[],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=OpenAIChoice(model=model),
                )

            response = await self.aclient.responses.create(  # type: ignore
                model=model,
                input=kwargs.get("input", query),
                instructions=instructions or None,
                temperature=temperature or None,
                **kwargs,
            )
            response = cast(Response, response)
            return LLMResponse(
                response=cast(T, response.output_text),
                logprobs=[],
                total_tokens=response.usage.total_tokens if response.usage else None,
                provider_model=OpenAIChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with OpenAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None


def get_openai_llm(api_key: str) -> OpenAILLM | None:
    return OpenAILLM(api_key) if api_key else None
