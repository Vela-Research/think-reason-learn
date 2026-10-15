"""Empty, refused or oddly shaped provider responses (issue #40), offline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from think_reason_learn.core.exceptions import LLMError
from think_reason_learn.core.llms import LLM, GoogleChoice, OpenAIChoice
from think_reason_learn.core.llms._anthropic.ask import AnthropicInput, AnthropicLLM
from think_reason_learn.core.llms._anthropic.schemas import NOT_GIVEN
from think_reason_learn.core.llms._google.ask import GeminiLLM
from think_reason_learn.core.llms._schemas import LLMResponse

PRIORITY: list[Any] = [
    OpenAIChoice(model="gpt-4.1-nano"),
    GoogleChoice(model="gemini-2.0-flash-lite"),
]


class Out(BaseModel):
    answer: str


def _resp(value: Any, choice: Any, tokens: int | None = 1) -> LLMResponse[Any]:
    return LLMResponse(
        response=value, logprobs=[], total_tokens=tokens, provider_model=choice
    )


@pytest.fixture
def llm() -> Iterator[LLM]:
    """The shared LLM with both providers replaced by mocks, restored after."""
    instance = LLM()
    saved = (instance.openai_llm, instance.google_llm)
    yield instance
    instance.openai_llm, instance.google_llm = saved


def _providers(
    llm: LLM,
    openai_value: Any,
    google_value: Any,
    tokens: tuple[int | None, int | None] = (1, 1),
) -> None:
    first = _resp(openai_value, PRIORITY[0], tokens[0])
    second = _resp(google_value, PRIORITY[1], tokens[1])
    llm.openai_llm = MagicMock(
        respond_sync=MagicMock(return_value=first),
        respond=AsyncMock(return_value=first),
    )
    llm.google_llm = MagicMock(
        respond_sync=MagicMock(return_value=second),
        respond=AsyncMock(return_value=second),
    )


async def _respond(llm: LLM, sync: bool, response_format: Any) -> LLMResponse[Any]:
    if sync:
        return llm.respond_sync(
            llm_priority=PRIORITY, query="q", response_format=response_format
        )
    return await llm.respond(
        query="q", llm_priority=PRIORITY, response_format=response_format
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize(
    ("empty", "response_format", "good"),
    [
        (None, Out, Out(answer="ok")),  # e.g. an OpenAI refusal: no parsed output
        ("", Out, Out(answer="ok")),  # text where a model was asked for
        (None, str, "ok"),
        ("  ", str, "ok"),
    ],
)
async def test_empty_output_falls_back_to_the_next_model(
    llm: LLM, sync: bool, empty: Any, response_format: Any, good: Any
) -> None:
    _providers(llm, empty, good)
    response = await _respond(llm, sync, response_format)
    assert response.response == good
    assert response.provider_model == PRIORITY[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False])
async def test_all_empty_returns_no_output_instead_of_raising(
    llm: LLM, sync: bool
) -> None:
    """Callers already handle response=None; it must not be text posing as a model."""
    _providers(llm, None, "")
    response = await _respond(llm, sync, Out)
    assert response.response is None


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize(
    ("tokens", "total"), [((900, 5), 905), ((None, 5), 5), ((900, None), 900)]
)
async def test_tokens_of_a_discarded_reply_are_counted(
    llm: LLM, sync: bool, tokens: tuple[int | None, int | None], total: int
) -> None:
    """A refusal is billed; the reply that is returned carries its tokens too."""
    _providers(llm, None, Out(answer="ok"), tokens)
    response = await _respond(llm, sync, Out)
    assert response.response == Out(answer="ok")
    assert response.total_tokens == total


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False])
async def test_tokens_add_up_when_every_reply_is_empty(llm: LLM, sync: bool) -> None:
    _providers(llm, None, "", (900, 7))
    response = await _respond(llm, sync, Out)
    assert response.response is None
    assert response.total_tokens == 907


# --- Anthropic: the tool call need not be the first content block ---------------


def _anthropic(content: list[Any], stop_reason: str = "end_turn") -> AnthropicLLM:
    llm = object.__new__(AnthropicLLM)  # not the shared singleton
    message = SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=2, output_tokens=3),
    )
    llm.aclient = MagicMock()
    llm.aclient.messages.create = AsyncMock(return_value=message)
    llm._process_input = lambda *args, **kwargs: AnthropicInput(  # type: ignore[method-assign]
        messages=[],
        system=NOT_GIVEN,
        tools=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        temperature=NOT_GIVEN,
        kwargs={},
    )
    return llm


async def _anthropic_respond(
    llm: AnthropicLLM, response_format: Any, raise_: bool
) -> Any:
    return await llm.respond(
        model="claude-3-5-haiku-latest",
        query="q",
        response_format=response_format,
        raise_=raise_,
    )


@pytest.mark.asyncio
async def test_anthropic_finds_the_tool_call_after_other_blocks() -> None:
    text = SimpleNamespace(type="text", text="Let me answer.")
    tool = SimpleNamespace(type="tool_use", input={"answer": "ok"})
    response = await _anthropic_respond(_anthropic([text, tool]), Out, raise_=True)
    assert response.response == Out(answer="ok")
    assert response.total_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_joins_text_blocks() -> None:
    blocks = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="text", text="world"),
    ]
    response = await _anthropic_respond(_anthropic(blocks), str, raise_=True)
    assert response.response == "Hello world"


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", [Out, str])
async def test_anthropic_empty_content_is_a_clear_error(response_format: Any) -> None:
    llm = _anthropic([], stop_reason="max_tokens")
    with pytest.raises(LLMError, match="max_tokens"):
        await _anthropic_respond(llm, response_format, raise_=True)
    assert await _anthropic_respond(llm, response_format, raise_=False) is None


# --- Google: no candidates for a structured request ------------------------------


def test_google_without_candidates_returns_no_output_for_a_model() -> None:
    google = object.__new__(GeminiLLM)  # not the shared singleton
    response = SimpleNamespace(candidates=None, parsed=None, usage_metadata=None)
    parsed = google._parse_response(response, "gemini-2.0-flash-lite", Out)  # type: ignore[arg-type]
    assert parsed.response is None
