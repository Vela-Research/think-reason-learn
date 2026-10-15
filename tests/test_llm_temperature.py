"""Temperature 0 reaches every provider; models that reject one get their default."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import omit

from think_reason_learn.core.llms import (
    LLM,
    AnthropicChoice,
    GoogleChoice,
    OpenAIChoice,
    XAIChoice,
)
from think_reason_learn.core.llms import _temperature
from think_reason_learn.core.llms._openai.ask import OpenAILLM
from think_reason_learn.core.llms._schemas import LLMResponse

CHOICES: dict[str, OpenAIChoice | AnthropicChoice | GoogleChoice | XAIChoice] = {
    "openai": OpenAIChoice(model="gpt-4.1-nano"),
    "anthropic": AnthropicChoice(model="claude-haiku-4-5"),
    "google": GoogleChoice(model="gemini-2.5-flash"),
    "xai": XAIChoice(model="grok-4"),
}


@pytest.fixture(autouse=True)
def _forget_rejections() -> Iterator[None]:
    _temperature._REJECTING_MODELS.clear()
    yield
    _temperature._REJECTING_MODELS.clear()


@pytest.fixture
def llm() -> Iterator[LLM]:
    instance = LLM()
    saved = {p: getattr(instance, f"{p}_llm") for p in CHOICES}
    yield instance
    for provider, adapter in saved.items():
        setattr(instance, f"{provider}_llm", adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(CHOICES))
@pytest.mark.parametrize("sync", [True, False])
async def test_temperature_zero_reaches_the_adapter(
    llm: LLM, provider: str, sync: bool
) -> None:
    choice = CHOICES[provider]
    reply = LLMResponse(response="ok", logprobs=[], provider_model=choice)
    adapter = MagicMock(
        respond_sync=MagicMock(return_value=reply),
        respond=AsyncMock(return_value=reply),
    )
    setattr(llm, f"{provider}_llm", adapter)

    if sync:
        llm.respond_sync(llm_priority=[choice], query="q", temperature=0.0)
        call = adapter.respond_sync.call_args
    else:
        await llm.respond(
            query="q", llm_priority=[choice], response_format=str, temperature=0.0
        )
        call = adapter.respond.call_args

    assert call.kwargs["temperature"] == 0.0


def _openai(create: MagicMock) -> OpenAILLM:
    adapter = OpenAILLM.__new__(OpenAILLM)
    adapter.client = SimpleNamespace(responses=SimpleNamespace(create=create))  # type: ignore[assignment]
    return adapter


def _text(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(output_text=text, usage=SimpleNamespace(total_tokens=3))


def test_openai_sends_temperature_zero() -> None:
    create = MagicMock(return_value=_text())
    _openai(create).respond_sync(model="gpt-4.1-nano", query="q", temperature=0.0)
    assert create.call_args.kwargs["temperature"] == 0.0


def test_rejected_temperature_is_retried_without_it_and_remembered() -> None:
    rejection = Exception("Unsupported parameter: 'temperature' is not supported")
    create = MagicMock(side_effect=[rejection, _text("first"), _text("second")])
    adapter = _openai(create)

    first = adapter.respond_sync(model="gpt-5-mini", query="q", temperature=0.0)
    second = adapter.respond_sync(model="gpt-5-mini", query="q", temperature=0.0)

    assert first is not None and first.response == "first"
    assert second is not None and second.response == "second"
    sent = [c.kwargs["temperature"] for c in create.call_args_list]
    assert sent == [0.0, omit, omit]


@pytest.mark.asyncio
async def test_async_rejection_is_retried() -> None:
    adapter = OpenAILLM.__new__(OpenAILLM)
    create = AsyncMock(
        side_effect=[Exception("temperature: unsupported value"), _text()]
    )
    adapter.aclient = SimpleNamespace(responses=SimpleNamespace(create=create))  # type: ignore[assignment]

    reply = await adapter.respond(model="o4-mini", query="q", temperature=0.2)

    assert reply is not None and reply.response == "ok"
    assert [c.kwargs["temperature"] for c in create.call_args_list] == [0.2, omit]


def test_other_errors_are_not_retried() -> None:
    create = MagicMock(side_effect=Exception("rate limited"))
    adapter = _openai(create)

    assert (
        adapter.respond_sync(model="gpt-4.1-nano", query="q", temperature=0.0) is None
    )
    with pytest.raises(Exception, match="rate limited"):
        adapter.respond_sync(
            model="gpt-4.1-nano", query="q", temperature=0.0, raise_=True
        )
    assert create.call_count == 2
    assert "gpt-4.1-nano" not in _temperature._REJECTING_MODELS


def test_anthropic_adapter_sends_temperature_in_the_body() -> None:
    """The Anthropic SDK no longer takes ``temperature``; it goes in the body."""
    import json

    import anthropic
    import httpx2

    from think_reason_learn.core.llms._anthropic.ask import AnthropicLLM

    bodies: list[dict[str, object]] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    adapter = AnthropicLLM.__new__(AnthropicLLM)
    adapter.client = anthropic.Anthropic(
        api_key="k", http_client=httpx2.Client(transport=httpx2.MockTransport(reply))
    )
    adapter.respond_sync(model="claude-haiku-4-5", query="q", temperature=0.0)
    adapter.respond_sync(model="claude-haiku-4-5", query="q")

    assert [b.get("temperature") for b in bodies] == [0.0, None]
