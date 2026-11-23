"""Anthropic's Messages protocol, with native structured outputs and prompt caching.

Replies are validated here, so a malformed, refused or truncated one is "no
usable output", never a crash; every call returns the rate-limit headers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypedDict

from pydantic import BaseModel, ValidationError

from .._config import get_config
from .._spec import Endpoint
from .._usage import TokenCounts
from ._base import GenerateRequest, Reply, classify, read_reply
from ._http import cached, http_client
from ._prompted import ReasoningAdapter

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
    "none": "low",  # Claude's lowest; sent so it does not default to high
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}
_UNUSABLE: Final = frozenset({"refusal", "max_tokens", "model_context_window_exceeded"})


class AnthropicAdapter(ReasoningAdapter):
    """Claude through the Anthropic SDK."""

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        """Text, or an instance of ``request.schema``; ``None`` if unusable."""
        client = _client(endpoint)  # first: it explains a missing SDK
        import anthropic

        try:
            raw = await client.messages.with_raw_response.create(
                timeout=get_config().timeout, **_params(endpoint, request, dropped)
            )
        except anthropic.APIStatusError as exc:
            raise classify(exc.status_code, exc.message, exc.response.headers) from exc
        except anthropic.APIConnectionError as exc:
            raise classify(None, str(exc), {}) from exc
        return await read_reply(
            raw.parse,
            anthropic.types.Message,
            lambda message: Reply(
                _output(message, request.schema),
                _tokens(message.usage),
                message.model,
                None,
                dict(raw.headers),
            ),
        )


class _Params(TypedDict):
    model: str
    max_tokens: int
    messages: list[MessageParam]
    system: str | Omit
    output_config: OutputConfigParam | Omit
    extra_body: dict[str, float | str] | None


def _client(endpoint: Endpoint) -> anthropic.AsyncAnthropic:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            f"{endpoint.name} needs the Anthropic SDK: pip install "
            "'think-reason-learn[anthropic]'."
        ) from exc

    def build() -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key=endpoint.api_key or "not-needed",  # never the SDK's own lookup
            base_url=endpoint.base_url,
            max_retries=0,
            http_client=http_client(endpoint.client_key),
        )

    return cached(f"anthropic|{endpoint.client_key}", build)


def _params(
    endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
) -> _Params:
    from anthropic import omit

    settings = endpoint.settings
    extra = {
        name: value
        for name, value in (
            ("temperature", settings.temperature),
            ("service_tier", settings.service_tier),
        )
        if value is not None and name not in dropped
    }
    return _Params(
        model=endpoint.model_id,
        max_tokens=settings.max_output_tokens or _MAX_TOKENS,
        messages=[{"role": "user", "content": _blocks(request)}],
        system=request.instructions or omit,
        output_config=_output_config(endpoint, request, dropped) or omit,
        extra_body=extra or None,
    )


def _blocks(request: GenerateRequest) -> list[TextBlockParam]:
    """The prompt, after the prefix; the cache breakpoint sits after the prefix."""
    blocks: list[TextBlockParam] = [{"type": "text", "text": request.prompt}]
    if request.prefix:
        prefix: TextBlockParam = {
            "type": "text",
            "text": request.prefix,
            "cache_control": {"type": "ephemeral"},
        }
        blocks.insert(0, prefix)
    return blocks


def _output_config(
    endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
) -> OutputConfigParam:
    import anthropic

    config: OutputConfigParam = {}
    effort = _EFFORTS.get(endpoint.settings.reasoning_effort or "")
    if effort and "reasoning_effort" not in dropped:
        config["effort"] = effort
    if request.schema is not None:
        config["format"] = {
            "type": "json_schema",
            "schema": anthropic.transform_schema(request.schema),
        }
    return config


def _output(message: Message, schema: type[BaseModel] | None) -> str | BaseModel | None:
    """The reply as text or ``schema``; ``None`` if refused, truncated or invalid."""
    if message.stop_reason in _UNUSABLE:
        return None
    text = "".join(block.text for block in message.content if block.type == "text")
    if schema is None or not text:
        return text or None
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        return None


def _tokens(usage: Usage) -> TokenCounts:
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    return TokenCounts(
        input=usage.input_tokens + cache_read + cache_write,
        output=usage.output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
    )
