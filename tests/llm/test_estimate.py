"""Estimates count calls and tokens the way the session sends them."""

from __future__ import annotations

import math

from think_reason_learn import LLM
from think_reason_learn.llm import (
    GENERATE_OUTPUT_TOKENS,
    Price,
    ReasoningEffort,
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
    assert planned.cost_usd is not None
    assert math.isclose(planned.cost_usd, planned.input_tokens / 1e6, rel_tol=0.01)


def test_decision_models_take_every_question_in_one_call() -> None:
    jev = resolve_models(LLM("jev", api_key="k"), param="answerer", task="answer")
    assert estimate_answers(jev, ["a", "b"], [20] * 30).calls == 2


def test_generations() -> None:
    writer = resolve_models(
        LLM("openai:gpt-5-mini", api_key="k"), param="g", task="generate"
    )
    planned = estimate_generations(writer, 500, calls=3)
    assert (planned.calls, planned.input_tokens, planned.role) == (3, 1800, "generator")


def test_a_writer_thinking_longer_is_estimated_to_write_more() -> None:
    def output(effort: ReasoningEffort | None, cap: int | None = None) -> int:
        writer = resolve_models(
            LLM(
                "openai:gpt-5-mini",
                api_key="k",
                reasoning_effort=effort,
                max_output_tokens=cap,
            ),
            param="g",
            task="generate",
        )
        return estimate_generations(writer, 500).output_tokens

    outputs = [output(e) for e in ("none", "low", "medium", "high", "xhigh")]
    assert outputs == sorted(set(outputs))
    assert output("none") == output("minimal") == GENERATE_OUTPUT_TOKENS
    assert output(None) == output("medium")  # most models' default
    assert output("high", cap=4_000) == 4_000
