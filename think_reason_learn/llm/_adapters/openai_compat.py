"""OpenAI's Chat Completions protocol: OpenAI and every compatible provider.

Google, xAI, OpenRouter, Groq, Together, DeepSeek, Mistral, Azure, Ollama, vLLM,
LM Studio and llama.cpp all speak it. Settings a model does not accept are sent
only when set, and dropped for good when the model rejects them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Final, TypedDict

import openai
from openai import Omit, omit
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, ValidationError

from .._config import get_config
from .._questions import Question
from .._spec import Endpoint
from .._usage import TokenCounts
from ._base import Answers, GenerateRequest, Reply, classify
from ._http import http_client
from ._prompted import prompted_answer


class OpenAICompatAdapter:
    """Chat Completions over the OpenAI SDK, for any ``base_url``."""

    def max_questions(self, endpoint: Endpoint) -> int:
        """Questions per call, as the model's settings ask."""
        return endpoint.settings.questions_per_call

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        """Text, or an instance of ``request.schema``."""
        client, params = openai_client(endpoint), _params(endpoint, request, dropped)
        try:
            if request.schema is not None and not endpoint.json_object_only:
                raw = await client.chat.completions.with_raw_response.parse(
                    response_format=request.schema, **params
                )
                parsed = raw.parse()
                output: str | BaseModel | None = _parsed(parsed)
                completion: ChatCompletion = parsed
            else:
                raw = await client.chat.completions.with_raw_response.create(
                    response_format={"type": "json_object"} if request.schema else omit,
                    **params,
                )
                completion = raw.parse()
                output = _content(completion, request.schema)
        except openai.LengthFinishReasonError as exc:
            return Reply(None, _tokens(exc.completion.usage), exc.completion.model)
        except openai.ContentFilterFinishReasonError:
            return Reply(None, TokenCounts())
        except openai.APIStatusError as exc:
            raise classify(exc.status_code, exc.message, exc.response.headers) from exc
        except openai.APIConnectionError as exc:
            raise classify(None, str(exc), {}) from exc
        return Reply(
            output,
            _tokens(completion.usage),
            completion.model,
            _provider_cost(completion.usage),
            dict(raw.headers),
        )

    async def answer(
        self,
        endpoint: Endpoint,
        sample: str,
        questions: Mapping[str, Question],
        dropped: frozenset[str],
    ) -> Reply[Answers]:
        """Answers by prompting for a label."""
        return await prompted_answer(
            self.generate, endpoint, sample, questions, dropped
        )


def openai_client(endpoint: Endpoint) -> openai.AsyncOpenAI:
    """The OpenAI SDK client for ``endpoint``'s base URL and account."""
    key = endpoint.client_key
    if key not in _clients:
        timeout = get_config().timeout
        header_key = endpoint.api_key if endpoint.api_key_header else None
        _clients[key] = openai.AsyncOpenAI(
            api_key=endpoint.api_key or "not-needed",
            base_url=endpoint.base_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client(key, timeout),
            default_headers={"api-key": header_key} if header_key else None,
        )
    return _clients[key]


_clients: Final[dict[str, openai.AsyncOpenAI]] = {}


class _Params(TypedDict):
    model: str
    messages: list[ChatCompletionMessageParam]
    temperature: float | Omit
    reasoning_effort: ReasoningEffort | Omit
    max_completion_tokens: int | Omit
    prompt_cache_key: str | Omit
    extra_body: dict[str, str] | None


def _params(
    endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
) -> _Params:
    settings = endpoint.settings

    def kept[V](name: str, value: V | None) -> V | Omit:
        return omit if value is None or name in dropped else value

    messages: list[ChatCompletionMessageParam] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    user = f"{request.prefix}\n\n{request.prompt}" if request.prefix else request.prompt
    messages.append({"role": "user", "content": user})
    cache_key = (
        hashlib.sha256(request.prefix.encode()).hexdigest()[:32]
        if request.prefix and endpoint.provider == "openai"
        else None
    )
    tier = kept("service_tier", settings.service_tier)
    return _Params(
        model=endpoint.model_id,
        messages=messages,
        temperature=kept("temperature", settings.temperature),
        reasoning_effort=kept("reasoning_effort", settings.reasoning_effort),
        max_completion_tokens=kept("max_completion_tokens", settings.max_output_tokens),
        prompt_cache_key=kept("prompt_cache_key", cache_key),
        extra_body=None if isinstance(tier, Omit) else {"service_tier": tier},
    )


def _parsed(completion: ParsedChatCompletion[BaseModel]) -> BaseModel | None:
    return completion.choices[0].message.parsed if completion.choices else None


def _content(
    completion: ChatCompletion, schema: type[BaseModel] | None
) -> str | BaseModel | None:
    content = completion.choices[0].message.content if completion.choices else None
    if schema is None or content is None:
        return content
    try:
        return schema.model_validate_json(content)
    except ValidationError:
        return None


def _tokens(usage: CompletionUsage | None) -> TokenCounts:
    if usage is None:
        return TokenCounts()
    prompt, completion = usage.prompt_tokens_details, usage.completion_tokens_details
    return TokenCounts(
        input=usage.prompt_tokens,
        output=usage.completion_tokens,
        cache_read=(prompt.cached_tokens or 0) if prompt else 0,
        cache_write=(prompt.cache_write_tokens or 0) if prompt else 0,
        reasoning=(completion.reasoning_tokens or 0) if completion else 0,
    )


def _provider_cost(usage: CompletionUsage | None) -> float | None:
    """The cost some providers (OpenRouter) report with the usage."""
    cost = (usage.model_extra or {}).get("cost") if usage is not None else None
    return float(cost) if isinstance(cost, (int, float)) else None
