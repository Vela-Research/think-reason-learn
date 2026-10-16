"""estimate_cost: the calls a fit or predict would make, before any is sent."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from sklearn.exceptions import NotFittedError

from think_reason_learn import config_context
from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.exceptions import DataError
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
    if name in ("gptree", "policy", "policy-rules"):
        # Counted in full: every level, every refinement. This toy tree stops at
        # one level, and a perfectly precise policy stops refining.
        assert made <= planned.calls
    else:
        assert planned.calls == made
    assert math.isclose(predicted, model.usage_.calls - made, rel_tol=0.2)


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
