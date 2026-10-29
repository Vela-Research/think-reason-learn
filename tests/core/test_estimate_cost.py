"""estimate_cost: the calls a fit or predict would make, before any is sent."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import pytest
from sklearn.exceptions import NotFittedError

from think_reason_learn import LLM, config_context
from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import Price
from think_reason_learn.rrf import RRF
from think_reason_learn.testing import FakeLLM

NEW = [f"Another founder {i} is {'good' if i % 3 else 'bad'}." for i in range(30)]


def test_estimates_match_the_calls_made(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
) -> None:
    (name, make), (writer, judge), (X, y) = any_model, generic_fakes, founders
    with config_context(cache=False):
        model = make()
        planned = model.estimate_cost(X, y)
        assert writer.n_calls == judge.n_calls == 0  # nothing is sent
        assert planned.cost_usd > 0 and not planned.unpriced
        model.fit(X, y)
        made = model.usage_.calls
        predicted = model.estimate_cost(NEW).calls
        model.predict(NEW)
    if name in ("gptree", "policy", "policy-rules", "rule-mining"):
        # Counted at their most: every level, every refinement, every second
        # look. This toy tree stops at one level, a perfectly precise policy
        # stops refining, and only the good founders get a second look.
        assert made <= planned.calls
    else:
        assert planned.calls == made
    assert math.isclose(predicted, model.usage_.calls - made, rel_tol=0.2)


def test_later_prompts_carry_the_instructions(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
) -> None:
    (_, make), (X, y) = any_model, founders
    if make()._instructions_purpose is None:
        pytest.skip("takes no instructions")

    def writer_input(instructions: str) -> int:
        model = make()
        model.instructions = instructions
        planned = model.estimate_cost(X, y)
        return sum(r.input_tokens for r in planned if r.role == "generator")

    assert writer_input("Be thorough. " * 500) > writer_input("Be brief.")


def test_predict_needs_a_fitted_model_and_fit_matching_labels(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
) -> None:
    (_, make), (X, y) = any_model, founders
    with pytest.raises(NotFittedError):
        make().estimate_cost(X)
    with pytest.raises(DataError, match="labels"):
        make().estimate_cost(X, y[:-1])


def test_estimating_needs_no_api_key(
    monkeypatch: pytest.MonkeyPatch, founders: tuple[list[str], list[str]]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = RRF(
        task="Will it succeed?",
        generator="openai:gpt-5-mini",
        answerer="openai:gpt-5-mini",
    )
    assert model.estimate_cost(*founders).calls > 0


def test_each_run_logs_its_estimate_first(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    (_, make), (X, y) = any_model, founders
    with caplog.at_level(logging.INFO, logger="think_reason_learn"):
        make().fit(X, y).predict(X)
    logged = [r.message for r in caplog.records if "before cached replies" in r.message]
    assert len(logged) == 2 and "fitting" in logged[0] and "predicting" in logged[1]


def test_a_costly_run_warns_and_a_cap_refuses_unknown_prices(
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer, judge = generic_fakes
    costly = LLM("fake:judge", price=Price(input=1e3))
    with config_context(cache=False):
        RRF(task="t", generator="fake:writer", answerer=costly, cv=2).fit(*founders)
    assert any(
        r.levelno == logging.WARNING and "estimated $" in r.message
        for r in caplog.records
    )
    calls = writer.n_calls + judge.n_calls
    with (
        config_context(max_cost_usd=100),
        pytest.raises(ValueError, match="no known price"),
    ):
        RRF(task="t", generator="fake:writer", answerer="fake:judge").fit(*founders)
    assert writer.n_calls + judge.n_calls == calls  # refused before any call
