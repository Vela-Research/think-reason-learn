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
from think_reason_learn.core.exceptions import LLMError
from .schemas import AnthropicChatModel, NOT_GIVEN, NotGiven
from .._temperature import drop_rejected_temperature
from .._schemas import LLMResponse, T, AnthropicChoice

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096


def _output(message: Message, response_format: Type[T]) -> T:
    """The tool call's arguments or the text of a message.

    The forced tool call need not be the first content block (a thinking or text
    block may come first), and a message cut short may carry neither.
    """
    if issubclass(response_format, BaseModel):
        for block in message.content:
            if block.type == "tool_use":
                return cast(T, response_format(**cast(Dict[str, Any], block.input)))
        missing = "tool call"
    else:
        text = "".join(b.text for b in message.content if b.type == "text")
        if text:
            return cast(T, text)
        missing = "text"
    raise LLMError(
        f"Anthropic returned no {missing} (stop_reason: {message.stop_reason})."
    )


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
                    "tool_choice is not supported when response_format is a "
                    "Pydantic model in Anthropic"
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
        system = (
            [{"type": "text", "text": instructions}]
            if (not system and instructions)
            else system
        )

        return AnthropicInput(
            messages=messages,
            system=system,
            tools=tools or NOT_GIVEN,
            tool_choice=tool_choice,
            temperature=temperature,
            kwargs=kwargs,
        )

    @drop_rejected_temperature(NOT_GIVEN)
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
                extra_body=_temperature_body(input_["temperature"]),
                **input_["kwargs"],
            )
            response = cast(Message, response)

            return LLMResponse(
                response=_output(response, response_format),
                logprobs=[],
                total_tokens=response.usage.output_tokens + response.usage.input_tokens,
                provider_model=AnthropicChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with Anthropic: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    @drop_rejected_temperature(NOT_GIVEN)
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
                extra_body=_temperature_body(input_["temperature"]),
                **input_["kwargs"],
            )
            response = cast(Message, response)

            return LLMResponse(
                response=_output(response, response_format),
                logprobs=[],
                total_tokens=response.usage.output_tokens + response.usage.input_tokens,
                provider_model=AnthropicChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with Anthropic: {e}", exc_info=True)
            if raise_:
                raise e
            return None


def _temperature_body(temperature: float | NotGiven) -> dict[str, float] | None:
    """The temperature as an extra body field: the SDK no longer takes one."""
    return None if isinstance(temperature, NotGiven) else {"temperature": temperature}


def get_anthropic_llm(api_key: str) -> AnthropicLLM | None:
    return AnthropicLLM(api_key) if api_key else None
