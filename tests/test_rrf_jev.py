"""Offline tests for RRF answering with Jev (FakeJevLLM, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from think_reason_learn.core.llms import (
    JevBudget,
    JevChoice,
    LLMChoice,
    NoulQuestion,
    OpenAIChoice,
)
from think_reason_learn.rrf import RRF
from think_reason_learn.rrf._rrf import Answer
from tests.fake_jev import FakeJevLLM

LLM_CHOICE: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]
PEOPLE = [
    "A: 30yo woman, CS Stanford, 6yr Google, AI healthcare startup, $2M seed.",
    "B: 25yo man, marketing NYU, 3yr Apple marketing mgr, social media app.",
    "C: LA doctor, UCLA, remote monitoring platform, no tech/startup exp.",
    "D: 40yo man, law UChicago, corporate lawyer, legal-tech idea, no tech bg.",
    "E: 28yo woman, CE Berkeley, fintech YC startup, own fintech product.",
    "F: 32yo man, MBA Columbia, marketing Apple/Spotify, subscription box.",
]
LABELS = ["YES", "NO", "NO", "YES", "NO", "YES"]
# A sits exactly on the threshold, B just below it.
PROBS = {"A": 0.5, "B": 0.49, "C": 0.1, "D": 0.9, "E": 0.51, "F": 0.2}


def _prob(state: str, instructions: str) -> float:
    return PROBS[state.split(": ", 1)[1][0]]


def _rrf(tmp_path: Path, fake: FakeJevLLM, **kwargs: Any) -> RRF:
    return RRF(
        qgen_llmc=LLM_CHOICE,
        qanswer_llmc=kwargs.pop("qanswer_llmc", [JevChoice(max_cost_usd=3.0)]),
        name="jev_rrf",
        save_path=tmp_path,
        max_samples_as_context=5,
        max_generated_questions=6,
        _llm=fake,
        **kwargs,
    )


async def _fit(rrf: RRF) -> RRF:
    await rrf.set_tasks(task_description="Classify founders")
    return await rrf.fit(pd.DataFrame({"data": PEOPLE}), LABELS)


def _active_questions(rrf: RRF) -> dict[str, str]:
    qdf = rrf.get_questions()
    return {str(q): str(qdf.at[q, "question"]) for q in qdf.index}


@pytest.mark.asyncio
async def test_fit_asks_jev_once_per_sample_with_every_question(
    tmp_path: Path,
) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, fake))

    questions = _active_questions(rrf)
    assert len(fake.answer_requests) == len(PEOPLE)
    for request in fake.answer_requests:
        assert {
            qid: q.instructions for qid, q in request.questions.items()
        } == questions
        assert all(isinstance(q, NoulQuestion) for q in request.questions.values())
    assert not any(c["response_format"] is Answer for c in fake.calls)


@pytest.mark.asyncio
async def test_jev_is_sent_sample_text_only(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    await _fit(_rrf(tmp_path, fake))

    assert sorted(r.state for r in fake.answer_requests) == sorted(
        f"data: {p}" for p in PEOPLE
    )


@pytest.mark.asyncio
async def test_probability_maps_to_yes_at_one_half(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, fake))

    answers = rrf.get_answers()
    probs = rrf.get_answer_probabilities()
    expected = ["YES", "NO", "NO", "YES", "YES", "NO"]
    for qid in _active_questions(rrf):
        assert list(answers[qid].astype(str)) == expected
        assert list(probs[qid]) == [PROBS[p[0]] for p in PEOPLE]


@pytest.mark.asyncio
async def test_fit_shares_one_budget_with_the_choice_cap(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = _rrf(tmp_path, fake, cost_sensitive=True)
    await _fit(rrf)

    budgets = [c["budget"] for c in fake.answer_calls]
    assert len(budgets) == 2
    assert isinstance(budgets[0], JevBudget) and budgets[0] is budgets[1]
    assert budgets[0].max_cost_usd == 3.0


@pytest.mark.asyncio
async def test_jev_answers_count_in_token_usage(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, fake))

    count = rrf.token_usage.token_counts["jev/jev-latest"]
    assert count.number_of_calls == len(PEOPLE)


@pytest.mark.asyncio
async def test_predict_with_jev_one_request_per_sample(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, fake))
    fake.answer_requests.clear()
    fake.answer_calls.clear()

    X_new = pd.DataFrame({"data": PEOPLE[:3]})
    rows = [(s, q, a) async for s, q, a, _ in rrf.predict(X_new)]

    questions = _active_questions(rrf)
    assert len(fake.answer_requests) == 3
    assert len(fake.answer_calls) == 1
    assert isinstance(fake.answer_calls[0]["budget"], JevBudget)
    assert sorted(rows) == sorted(
        (i, qid, ["YES", "NO", "NO"][i]) for i in range(3) for qid in questions
    )


@pytest.mark.asyncio
async def test_predict_with_jev_resumes_from_checkpoint(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, fake))
    X_new = pd.DataFrame({"data": PEOPLE[:2]})
    ckpt = tmp_path / "ckpt"

    fake.answer_requests.clear()
    first = [r[:3] async for r in rrf.predict(X_new, checkpoint_path=ckpt)]
    assert len(fake.answer_requests) == 2
    fake.answer_requests.clear()
    again = [r[:3] async for r in rrf.predict(X_new, checkpoint_path=ckpt, resume=True)]

    assert sorted(again) == sorted(first)
    assert fake.answer_requests == []


@pytest.mark.asyncio
async def test_unanswered_questions_stay_missing(tmp_path: Path) -> None:
    def flaky(state: str, instructions: str) -> float | None:
        return None if state.startswith("data: C") else 0.8

    fake = FakeJevLLM(prob=flaky)
    rrf = await _fit(_rrf(tmp_path, fake))

    answers = rrf.get_answers()
    assert answers.iloc[2].isna().all()
    assert bool(answers.drop(index=2).notna().to_numpy().all())


@pytest.mark.asyncio
async def test_probabilities_survive_save_and_load(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, fake))
    rrf.save(tmp_path / "saved")

    loaded = RRF.load(tmp_path / "saved")

    pd.testing.assert_frame_equal(
        loaded.get_answer_probabilities(), rrf.get_answer_probabilities()
    )
    assert loaded.qanswer_llmc[0] in (
        JevChoice(max_cost_usd=3.0),
        JevChoice(max_cost_usd=3.0).model_dump(),
    )


@pytest.mark.asyncio
async def test_chat_answerer_keeps_the_chat_path(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    await _fit(_rrf(tmp_path, fake, qanswer_llmc=LLM_CHOICE))

    assert fake.answer_requests == []
    assert any(c["response_format"] is Answer for c in fake.calls)
