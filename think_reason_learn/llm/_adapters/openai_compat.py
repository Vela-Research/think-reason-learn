"""OpenAI's Chat Completions protocol: OpenAI and every compatible provider.

Google, xAI, OpenRouter, Groq, Together, DeepSeek, Mistral, Azure, Ollama, vLLM,
LM Studio and llama.cpp all speak it. Settings a model does not accept are sent
only when set, and dropped for good when the model rejects them. Replies are
validated here, so a malformed one is "no usable output", never a crash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final, TypedDict

import openai
from openai import Omit, omit
from openai.lib import type_to_response_format_param
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, ValidationError

from .._config import get_config
from .._spec import Endpoint
from .._usage import TokenCounts
from ._base import GenerateRequest, Reply, classify, read_reply
from ._http import cached, http_client
from ._prompted import ReasoningAdapter

_FLEX_TIMEOUT: Final = 900.0
_JSON_ONLY: Final = (
    "Reply with only a JSON object (no other text) matching this JSON schema:\n"
)


class OpenAICompatAdapter(ReasoningAdapter):
    """Chat Completions over the OpenAI SDK, for any ``base_url``."""

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        """Text, or an instance of ``request.schema``; ``None`` if unusable."""
        client = openai_client(endpoint)
        try:
            raw = await client.chat.completions.with_raw_response.create(
                response_format=_response_format(endpoint, request),
                timeout=_timeout(endpoint),
                **_params(endpoint, request, dropped),
            )
        except openai.APIStatusError as exc:
            raise classify(exc.status_code, exc.message, exc.response.headers) from exc
        except openai.APIConnectionError as exc:
            raise classify(None, str(exc), {}) from exc
        completion = await read_reply(raw.parse, ChatCompletion)
        return Reply(
            _output(completion, request.schema),
            _tokens(completion.usage),
            completion.model,
            _provider_cost(completion.usage),
            dict(raw.headers),
        )


def openai_client(endpoint: Endpoint) -> openai.AsyncOpenAI:
    """The OpenAI SDK client for ``endpoint``'s base URL and account."""

    def build() -> openai.AsyncOpenAI:
        header_key = endpoint.api_key if endpoint.api_key_header else None
        return openai.AsyncOpenAI(
            api_key=endpoint.api_key or "not-needed",
            base_url=endpoint.base_url,
            max_retries=0,
            http_client=http_client(endpoint.client_key),
            default_headers={"api-key": header_key} if header_key else None,
        )

    return cached(f"openai|{endpoint.client_key}", build)


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

    instructions = request.instructions or ""
    if request.schema is not None and endpoint.json_object_only:
        schema = json.dumps(request.schema.model_json_schema())
        instructions = f"{instructions}\n\n{_JSON_ONLY}{schema}".strip()
    messages: list[ChatCompletionMessageParam] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
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


def _response_format(
    endpoint: Endpoint, request: GenerateRequest
) -> ResponseFormat | Omit:
    if request.schema is None:
        return omit
    if endpoint.json_object_only:
        return {"type": "json_object"}
    return type_to_response_format_param(request.schema)


def _output(
    completion: ChatCompletion, schema: type[BaseModel] | None
) -> str | BaseModel | None:
    """The reply as text or ``schema``; ``None`` if truncated, refused or invalid."""
    if not completion.choices:
        return None
    choice = completion.choices[0]
    content = choice.message.content
    if choice.finish_reason == "length" or content is None:
        return None
    if schema is None:
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
        # xAI leaves reasoning out of completion_tokens; the total has it.
        output=max(usage.completion_tokens, usage.total_tokens - usage.prompt_tokens),
        cache_read=(prompt.cached_tokens or 0) if prompt else 0,
        cache_write=(prompt.cache_write_tokens or 0) if prompt else 0,
        reasoning=(completion.reasoning_tokens or 0) if completion else 0,
    )


def _timeout(endpoint: Endpoint) -> float:
    """The call timeout; the flex tier queues calls, so it gets 15 minutes."""
    timeout = get_config().timeout
    return (
        max(timeout, _FLEX_TIMEOUT)
        if endpoint.settings.service_tier == "flex"
        else timeout
    )


def _provider_cost(usage: CompletionUsage | None) -> float | None:
    """The cost some providers (OpenRouter) report with the usage."""
    cost = (usage.model_extra or {}).get("cost") if usage is not None else None
    return float(cost) if isinstance(cost, (int, float)) else None
