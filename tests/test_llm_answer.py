"""Offline tests for LLM.answer / LLM.answer_many (Jev first, chat fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest
from pydantic import BaseModel

from think_reason_learn.core.exceptions import JevCostCapError, LLMError
from think_reason_learn.core.llms import (
    LLM,
    ChoiceQuestion,
    JevChoice,
    NoulQuestion,
    OpenAIChoice,
    TokenCounter,
)
from think_reason_learn.core.llms._jev.client import JevBudget, JevClient, JevRequest
from think_reason_learn.core.llms._schemas import LLMResponse
from tests.fake_jev import FakeJevServer


class FakeChatProvider:
    """Stands in for OpenAILLM: answers YES or the last choice label."""

    def __init__(self) -> None:
        self.queries: List[str] = []

    async def respond(
        self, query: str, model: str, response_format: Any, **kwargs: Any
    ) -> LLMResponse[Any]:
        self.queries.append(query)
        field = response_format.model_fields["answer"]
        labels = list(field.annotation.__args__)
        answer = "YES" if "YES" in labels else labels[-1]
        return LLMResponse(
            response=response_format(answer=answer),
            logprobs=[],
            total_tokens=11,
            provider_model=OpenAIChoice(model=model),
        )


@pytest.fixture
def llm_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[LLM, FakeChatProvider, FakeJevServer]:
    llm = LLM()
    server = FakeJevServer(
        input_tokens=100, prob=0.3, status_for_state={"too long": 413}
    )
    chat = FakeChatProvider()
    monkeypatch.setattr(
        llm, "jev_llm", JevClient("k", transport=server.transport(), cache_dir=tmp_path)
    )
    monkeypatch.setattr(llm, "openai_llm", chat)
    return llm, chat, server


QUESTIONS = {
    "a": NoulQuestion(instructions="Has fintech experience?"),
    "b": ChoiceQuestion(instructions="Seniority?", labels=["junior", "senior"]),
}


@pytest.mark.asyncio
async def test_answer_with_jev(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, chat, server = llm_with_fakes

    answers = await llm.answer([JevChoice()], "Founder: fintech.", QUESTIONS)

    assert answers == {"a": 0.3, "b": "junior"}
    assert len(server.bodies) == 1
    assert chat.queries == []


@pytest.mark.asyncio
async def test_answer_many_counts_jev_tokens(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, _, _ = llm_with_fakes
    counter = TokenCounter()

    await llm.answer_many(
        [JevChoice()],
        [JevRequest("s1", QUESTIONS), JevRequest("s2", QUESTIONS)],
        token_counter=counter,
        caller="test",
    )

    count = counter.token_counts["jev/jev-latest"]
    assert (count.provider, count.number_of_calls, count.value) == ("jev", 2, 214)
    assert count.callers == {"test": 2}


@pytest.mark.asyncio
async def test_answer_with_chat_model_maps_yes_to_one(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, chat, server = llm_with_fakes

    answers = await llm.answer(
        [OpenAIChoice(model="gpt-4.1-nano")], "Founder: fintech.", QUESTIONS
    )

    assert answers == {"a": 1.0, "b": "senior"}
    assert server.bodies == []
    assert len(chat.queries) == 2
    assert all("Founder: fintech." in q for q in chat.queries)


@pytest.mark.asyncio
async def test_failed_jev_requests_fall_back_to_the_next_choice(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, chat, _ = llm_with_fakes

    results = await llm.answer_many(
        [JevChoice(), OpenAIChoice(model="gpt-4.1-nano")],
        [JevRequest("fine", QUESTIONS), JevRequest("too long", QUESTIONS)],
    )

    assert results[0].answers == {"a": 0.3, "b": "junior"}
    assert results[1].answers == {"a": 1.0, "b": "senior"}
    assert len(chat.queries) == 2 and all("too long" in q for q in chat.queries)


@pytest.mark.asyncio
async def test_cost_cap_is_not_a_reason_to_fall_back(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, chat, _ = llm_with_fakes

    with pytest.raises(JevCostCapError):
        await llm.answer_many(
            [JevChoice(), OpenAIChoice(model="gpt-4.1-nano")],
            [JevRequest("x" * 10_000, QUESTIONS)],
            budget=JevBudget(1e-9),
        )

    assert chat.queries == []


@pytest.mark.asyncio
async def test_jev_choice_cap_applies_when_no_budget_is_given(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, _, server = llm_with_fakes

    with pytest.raises(JevCostCapError):
        await llm.answer_many(
            [JevChoice(max_cost_usd=1e-9)], [JevRequest("x" * 10_000, QUESTIONS)]
        )

    assert server.bodies == []


@pytest.mark.asyncio
async def test_jev_choice_cache_flag_is_honoured(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, _, server = llm_with_fakes

    for _ in range(2):
        await llm.answer([JevChoice(cache=False)], "same sample", QUESTIONS)

    assert len(server.bodies) == 2


@pytest.mark.asyncio
async def test_answer_accepts_dict_choices(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, _, server = llm_with_fakes

    answers = await llm.answer(
        [{"provider": "jev", "model": "jev-latest"}], "dict sample", QUESTIONS
    )

    assert answers == {"a": 0.3, "b": "junior"}


@pytest.mark.asyncio
async def test_answer_without_typesafe_key_names_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = LLM()
    monkeypatch.setattr(llm, "jev_llm", None)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        await llm.answer([JevChoice()], "s", QUESTIONS)


@pytest.mark.asyncio
async def test_respond_refuses_jev_for_generation() -> None:
    class Out(BaseModel):
        text: str

    with pytest.raises(ValueError, match="cannot generate text"):
        await LLM().respond(
            query="Write questions", llm_priority=[JevChoice()], response_format=Out
        )


@pytest.mark.asyncio
async def test_every_request_failing_raises(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, _, _ = llm_with_fakes

    with pytest.raises(LLMError, match="HTTP 413"):
        await llm.answer_many(
            [JevChoice()],
            [JevRequest("too long", QUESTIONS), JevRequest("too long", QUESTIONS)],
        )


@pytest.mark.asyncio
async def test_some_requests_failing_does_not_raise(
    llm_with_fakes: tuple[LLM, FakeChatProvider, FakeJevServer],
) -> None:
    llm, _, _ = llm_with_fakes

    results = await llm.answer_many(
        [JevChoice()],
        [JevRequest("fine", QUESTIONS), JevRequest("too long", QUESTIONS)],
    )

    assert results[0].answers == {"a": 0.3, "b": "junior"}
    assert results[1].answers == {"a": None, "b": None}


@pytest.mark.asyncio
async def test_suite_cannot_reach_typesafe() -> None:
    with pytest.raises(LLMError, match="HTTP 599"):
        await LLM().answer([JevChoice(cache=False)], "offline check", QUESTIONS)
