"""Prompt helpers every model shares: writing calls run together."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import pytest

from think_reason_learn.core._prompts import most_written
from think_reason_learn.exceptions import ProviderAuthError
from think_reason_learn.llm import LLM, LLMSession, Price, resolve_models, run_sync
from think_reason_learn.llm._adapters import ProviderError
from think_reason_learn.llm._estimate import generate_output_tokens
from think_reason_learn.llm._usage import TokenCounts
from think_reason_learn.testing import FakeLLM, GenerateRequest, fake_llms

type SlowWrites = Callable[[Callable[[str], float]], list[str]]


def test_a_fatal_error_cancels_the_other_writing_calls(
    slow_writes: SlowWrites,
) -> None:
    def write(request: GenerateRequest) -> str:
        if request.prompt == "bad":
            raise ProviderError("auth", "fake key rejected")
        return "questions"

    sent = slow_writes(lambda prompt: 0.0 if prompt == "bad" else 30.0)
    with fake_llms(writer=FakeLLM(generate=write)):
        models = resolve_models(
            LLM("fake:writer", price=Price(1, 4)), param="g", task="generate"
        )
        session = LLMSession()
        calls = [session.generate(models, p, role="g") for p in ("a", "b", "bad")]
        started = time.monotonic()
        with pytest.raises(ProviderAuthError):
            run_sync(most_written(calls, "prompts"))
    assert time.monotonic() - started < 10  # the others did not run their 30 s
    assert sent == ["a", "b", "bad"]
    # Sent and cancelled: the provider may bill them, so they are counted at
    # their estimate, as when the caller cancels a run, and so is the budget.
    price, output = Price(1, 4), generate_output_tokens(models[0])
    records = sorted(session.usage, key=lambda r: r.cost_source)
    assert [(r.cost_source, r.output_tokens) for r in records] == [
        ("estimate", output),
        ("estimate", output),
        ("none", 0),
    ]
    for record in records[:2]:
        tokens = TokenCounts(input=record.input_tokens, output=output)
        assert record.error is not None and record.cost_usd == price.cost(tokens) > 0
    assert records[2].error == "fake key rejected"
    assert math.isclose(session._engine.budget.spent_usd, session.usage.cost_usd)
