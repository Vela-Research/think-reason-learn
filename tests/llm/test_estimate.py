"""Estimates count calls and tokens the way the session sends them."""

from __future__ import annotations

import pytest

from think_reason_learn import LLM
from think_reason_learn.llm import (
    Price,
    estimate_answers,
    estimate_generations,
    resolve_models,
)
from think_reason_learn.llm._estimate import call_tokens


def test_answers_are_grouped_per_call_like_the_session() -> None:
    reasoning = resolve_models(
        LLM("openai:gpt-5-mini", api_key="k", questions_per_call=3, price=Price(1)),
        param="answerer",
        task="answer",
    )
    samples, questions = ["one sample", "another sample"], [f"Q{i}?" for i in range(7)]
    planned = estimate_answers(reasoning, samples, questions, times=2)
    chunks = [questions[i : i + 3] for i in range(0, 7, 3)]
    sent = [call_tokens(s, *chunk) for s in samples for chunk in chunks]
    assert planned.calls == 2 * len(sent)
    assert planned.input_tokens == 2 * sum(sent)
    assert planned.cost_usd == pytest.approx(planned.input_tokens / 1e6, rel=0.01)


def test_decision_models_take_every_question_in_one_call() -> None:
    jev = resolve_models(LLM("jev", api_key="k"), param="answerer", task="answer")
    assert estimate_answers(jev, ["a", "b"], [20] * 30).calls == 2


def test_generations() -> None:
    writer = resolve_models(
        LLM("openai:gpt-5-mini", api_key="k"), param="g", task="generate"
    )
    planned = estimate_generations(writer, 500, calls=3)
    assert (planned.calls, planned.input_tokens, planned.role) == (3, 1800, "generator")
