"""Anthropic's Messages protocol, with native structured outputs and prompt caching."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from pydantic import BaseModel

from .._config import get_config
from .._questions import Question
from .._spec import Endpoint
from .._usage import TokenCounts
from ._base import Answers, GenerateRequest, Reply, classify
from ._http import http_client
from ._prompted import prompted_answer

if TYPE_CHECKING:
    import anthropic
    from anthropic import Omit
    from anthropic.types import (
        Message,
        MessageParam,
        OutputConfigParam,
        TextBlockParam,
        Usage,
    )

_MAX_TOKENS: Final = 4096
_EFFORTS: Final[dict[str, Literal["low", "medium", "high", "xhigh"]]] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


class AnthropicAdapter:
    """Claude through the Anthropic SDK."""

    def max_questions(self, endpoint: Endpoint) -> int:
        """Questions per call, as the model's settings ask."""
        return endpoint.settings.questions_per_call

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        """Text, or an instance of ``request.schema``."""
        import anthropic

        client, params = _client(endpoint), _params(endpoint, request, dropped)
        headers: Mapping[str, str] = {}
        try:
            if request.schema is not None:
                parsed = await client.messages.parse(
                    output_format=request.schema, **params
                )
                message: Message = parsed
                output: str | BaseModel | None = parsed.parsed_output
            else:
                raw = await client.messages.with_raw_response.create(**params)
                message, headers = await raw.parse(), dict(raw.headers)
                output = (
                    "".join(b.text for b in message.content if b.type == "text") or None
                )
        except anthropic.APIStatusError as exc:
            raise classify(exc.status_code, exc.message, exc.response.headers) from exc
        except anthropic.APIConnectionError as exc:
            raise classify(None, str(exc), {}) from exc
        if message.stop_reason == "refusal":
            output = None
        return Reply(output, _tokens(message.usage), message.model, None, headers)

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


class _Params(TypedDict):
    model: str
    max_tokens: int
    messages: list[MessageParam]
    system: str | Omit
    output_config: OutputConfigParam | Omit
    extra_body: dict[str, float] | None


def _client(endpoint: Endpoint) -> anthropic.AsyncAnthropic:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            f"{endpoint.name} needs the Anthropic SDK: pip install "
            "'think-reason-learn[anthropic]'."
        ) from exc
    key = endpoint.client_key
    if key not in _clients:
        timeout = get_config().timeout
        _clients[key] = anthropic.AsyncAnthropic(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client(key, timeout),
        )
    return _clients[key]


_clients: Final[dict[str, anthropic.AsyncAnthropic]] = {}


def _params(
    endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
) -> _Params:
    from anthropic import omit

    settings = endpoint.settings
    blocks: list[TextBlockParam] = [{"type": "text", "text": request.prompt}]
    if request.prefix:
        # The cache breakpoint sits after the sample, before the question.
        blocks.insert(
            0,
            {
                "type": "text",
                "text": request.prefix,
                "cache_control": {"type": "ephemeral"},
            },
        )
    effort = _EFFORTS.get(settings.reasoning_effort or "")
    if "reasoning_effort" in dropped:
        effort = None
    temperature = None if "temperature" in dropped else settings.temperature
    return _Params(
        model=endpoint.model_id,
        max_tokens=settings.max_output_tokens or _MAX_TOKENS,
        messages=[{"role": "user", "content": blocks}],
        system=request.instructions or omit,
        output_config={"effort": effort} if effort is not None else omit,
        extra_body={"temperature": temperature} if temperature is not None else None,
    )


def _tokens(usage: Usage) -> TokenCounts:
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    return TokenCounts(
        input=usage.input_tokens + cache_read + cache_write,
        output=usage.output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
    )
