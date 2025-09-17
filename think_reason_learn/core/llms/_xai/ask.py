from typing import Type, List, Any, cast, Dict
import logging
import contextlib

from xai_sdk import Client, AsyncClient
from xai_sdk.chat import system, user
from xai_sdk.proto.v6.chat_pb2 import Message, MessageRole
from pydantic import BaseModel

from .._schemas import LLMResponse, T, XAIChoice
from .schemas import xAIChatModel
from think_reason_learn.core._singleton import SingletonMeta


logger = logging.getLogger(__name__)


class xAILLM(metaclass=SingletonMeta):
    def __init__(self, api_key: str) -> None:
        self.client = Client(api_key=api_key)
        self.aclient = AsyncClient(api_key=api_key)

    def _process_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v or None
            for k, v in kwargs.items()
            if k in self.client.chat.create.__annotations__
        }

    def respond_sync(
        self,
        model: xAIChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | None = None,
        temperature: float | None = None,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)

        try:
            messages: List[Message] = kwargs.get("messages", [])
            with contextlib.suppress(IndexError):
                if query and messages[-1].role != MessageRole.ROLE_USER:
                    messages.append(user(query))
            if not (m for m in messages if m.role == MessageRole.ROLE_SYSTEM):
                if instructions:
                    messages.insert(0, system(instructions))

            chat = self.client.chat.create(
                model=model,
                temperature=temperature,
                messages=messages,
                logprobs=True,
                **kwargs,
            )

            if issubclass(response_format, BaseModel):
                response, parsed = chat.parse(response_format)
                return LLMResponse(
                    response=parsed,
                    logprobs=[
                        (lp.token, lp.logprob) for lp in response.logprobs.content
                    ],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=XAIChoice(model=model),
                )
            else:
                response = chat.sample()
                return LLMResponse(
                    response=cast(T, response.content),
                    logprobs=[
                        (lp.token, lp.logprob) for lp in response.logprobs.content
                    ],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=XAIChoice(model=model),
                )
        except Exception as e:
            logger.warning(f"Error responding with XAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    async def respond(
        self,
        model: xAIChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | None = None,
        temperature: float | None = None,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)

        try:
            messages: List[Message] = kwargs.get("messages", [])
            with contextlib.suppress(IndexError):
                if query and messages[-1].role != MessageRole.ROLE_USER:
                    messages.append(user(query))
            if not (m for m in messages if m.role == MessageRole.ROLE_SYSTEM):
                if instructions:
                    messages.insert(0, system(instructions))

            chat = self.aclient.chat.create(
                model=model,
                temperature=temperature,
                messages=messages,
                logprobs=True,
                **kwargs,
            )

            if issubclass(response_format, BaseModel):
                response, parsed = await chat.parse(response_format)
                return LLMResponse(
                    response=parsed,
                    logprobs=[
                        (lp.token, lp.logprob) for lp in response.logprobs.content
                    ],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=XAIChoice(model=model),
                )
            else:
                response = await chat.sample()
                return LLMResponse(
                    response=cast(T, response.content),
                    logprobs=[
                        (lp.token, lp.logprob) for lp in response.logprobs.content
                    ],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=XAIChoice(model=model),
                )
        except Exception as e:
            logger.warning(f"Error responding with XAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None


def get_xai_llm(api_key: str) -> xAILLM | None:
    return xAILLM(api_key) if api_key else None
