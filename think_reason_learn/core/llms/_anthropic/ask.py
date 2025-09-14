from typing import Type, Any, cast, List, Dict, Iterable, TypedDict
import logging

from pydantic import BaseModel
import anthropic
from anthropic.types import (
    ToolChoiceParam,
    ToolUnionParam,
    MessageParam,
    TextBlockParam,
    Message,
)

from think_reason_learn.core._singleton import SingletonMeta
from .schemas import AnthropicChatModel, NOT_GIVEN, NotGiven
from ..schemas import LLMResponse, T, AnthropicChoice

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096


class AnthropicInput(TypedDict):
    messages: List[MessageParam]
    system: str | Iterable[TextBlockParam] | NotGiven
    tools: List[ToolUnionParam] | NotGiven
    tool_choice: ToolChoiceParam | NotGiven
    temperature: float | NotGiven
    kwargs: Dict[str, Any]


class AnthropicLLM(metaclass=SingletonMeta):
    def __init__(self, api_key: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.aclient = anthropic.AsyncAnthropic(api_key=api_key)

    def _process_input(
        self,
        query: str,
        response_format: Type[T],
        instructions: str | NotGiven,
        temperature: float | NotGiven,
        kwargs: Dict[str, Any],
    ) -> AnthropicInput:
        kwargs = {
            k: v or NOT_GIVEN
            for k, v in kwargs.items()
            if k in self.client.messages.create.__annotations__
        }

        tools: List[ToolUnionParam] = kwargs.get("tools", [])
        tool_choice: ToolChoiceParam | NotGiven = kwargs.get("tool_choice", NOT_GIVEN)

        if issubclass(response_format, BaseModel):
            if tool_choice:
                raise ValueError(
                    "tool_choice is not supported when response_format is a Pydantic model in Anthropic"
                )

            tools.append(
                {
                    "name": response_format.__name__.lower(),
                    "description": response_format.__doc__ or "",
                    "input_schema": response_format.model_json_schema(),
                }
            )
            tool_choice = {"name": response_format.__name__.lower(), "type": "tool"}

        messages: List[MessageParam] = kwargs.get("messages", [])
        if query:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": query}]}
            )

        system: str | Iterable[TextBlockParam] | NotGiven = kwargs.get(
            "system", NOT_GIVEN
        )
        if instructions:
            if not system:
                system = instructions
            elif isinstance(system, str):
                system = [
                    {"type": "text", "text": system},
                    {"type": "text", "text": instructions},
                ]
            else:
                system = list(system)
                system.append({"type": "text", "text": instructions})

        return AnthropicInput(
            messages=messages,
            system=system,
            tools=tools or NOT_GIVEN,
            tool_choice=tool_choice,
            temperature=temperature,
            kwargs=kwargs,
        )

    def respond_sync(
        self,
        model: AnthropicChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        try:
            input_ = self._process_input(
                query,
                response_format,
                instructions,
                temperature,
                kwargs,
            )

            response = self.client.messages.create(  # type: ignore
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                messages=input_["messages"],
                system=input_["system"],
                tools=input_["tools"],
                tool_choice=input_["tool_choice"],
                temperature=input_["temperature"],
                **input_["kwargs"],
            )
            response = cast(Message, response)

            if issubclass(response_format, BaseModel):
                output = response_format(**response.content[0].input)  # type: ignore
            else:
                output = cast(T, response.content[0].text)  # type: ignore

            return LLMResponse(
                response=output,
                logprobs=[],
                total_tokens=response.usage.output_tokens + response.usage.input_tokens,
                provider_model=AnthropicChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with Anthropic: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    async def respond(
        self,
        model: AnthropicChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        try:
            input_ = self._process_input(
                query,
                response_format,
                instructions,
                temperature,
                kwargs,
            )
            response = await self.aclient.messages.create(  # type: ignore
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                messages=input_["messages"],
                system=input_["system"],
                tools=input_["tools"],
                tool_choice=input_["tool_choice"],
                temperature=input_["temperature"],
                **input_["kwargs"],
            )
            response = cast(Message, response)

            if issubclass(response_format, BaseModel):
                output = response_format(**response.content[0].input)  # type: ignore
            else:
                output = cast(T, response.content[0].text)  # type: ignore

            return LLMResponse(
                response=output,
                logprobs=[],
                total_tokens=response.usage.output_tokens + response.usage.input_tokens,
                provider_model=AnthropicChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with Anthropic: {e}", exc_info=True)
            if raise_:
                raise e
            return None


def get_anthropic_llm(api_key: str) -> AnthropicLLM | None:
    return AnthropicLLM(api_key) if api_key else None
