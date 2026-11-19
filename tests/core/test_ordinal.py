"""The ordinal combiner: its loss, what it recovers, and its cut-offs."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.linear_model import LogisticRegression

from think_reason_learn.core import OrdinalScorer, fit_ordinal
from think_reason_learn.core._ordinal import (
    _fit_once,
    balanced_cuts,
    boundary_aucs,
    ordinal_loss,
    quadratic_kappa,
)
from think_reason_learn.core._sklearn import fit, weights

type Floats = npt.NDArray[np.float64]
type Codes = npt.NDArray[np.intp]


def synthetic(
    n: int, coef: list[float], thresholds: list[float], seed: int = 0
) -> tuple[Floats, Codes]:
    """Rows drawn from a known cumulative-logit model on standard normal columns."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, len(coef)))
    eta = x @ np.array(coef)
    reached = rng.logistic(size=n)[:, None] < eta[:, None] - np.array(thresholds)
    return x, reached.sum(axis=1).astype(np.intp)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_the_gradient_matches_finite_differences(k: int) -> None:
    rng = np.random.default_rng(k)
    z = rng.standard_normal((40, 5))
    y = rng.integers(0, k, 40).astype(np.intp)
    weights = rng.uniform(0.5, 1.5, 40)
    params = rng.standard_normal(5 + k - 1) * 0.5
    _, grad = ordinal_loss(params, z, y, k, weights, 0.3)
    step = 1e-6
    numeric = np.array(
        [
            (
                ordinal_loss(params + step * e, z, y, k, weights, 0.3)[0]
                - ordinal_loss(params - step * e, z, y, k, weights, 0.3)[0]
            )
            / (2 * step)
            for e in np.eye(len(params))
        ]
    )
    assert np.abs(grad - numeric).max() < 1e-7


def test_it_recovers_a_known_model() -> None:
    x, y = synthetic(4000, [1.5, -1.0, 0.0], [-1.0, 1.0])
    scorer = _fit_once(x, y, 3, [(-np.inf, np.inf)] * 3, 1e4, np.ones(len(y)))
    coef = scorer.coef / scorer.scale  # back on the scale of x
    assert np.allclose(coef, [1.5, -1.0, 0.0], atol=0.12)
    # eta is centred on the training mean, so the cut-points shift by it.
    shift = float(scorer.mean @ coef)
    assert np.allclose(scorer.thresholds + shift, [-1.0, 1.0], atol=0.12)


def test_probabilities_are_ordered_and_sum_to_one() -> None:
    x, y = synthetic(300, [1.0, 0.5, -0.5], [-1.5, 0.0, 1.5], seed=1)
    scorer, out_of_fold = fit_ordinal(x, y, cv=3, random_state=0)
    proba = scorer.proba(x)
    cumulative = scorer.cumulative(x)
    assert proba.shape == out_of_fold.shape == (300, 4)
    assert np.allclose(proba.sum(axis=1), 1) and (proba >= 0).all()
    assert (np.diff(cumulative, axis=1) <= 0).all()  # P(>=1) >= P(>=2) >= P(>=3)
    assert np.all(np.diff(scorer.thresholds) > 0)
    assert scorer.cv_auc is not None and scorer.cv_auc > 0.7


def test_two_outcomes_is_logistic_regression() -> None:
    x, y = synthetic(500, [1.0, -2.0, 0.5], [0.3], seed=2)
    scorer = _fit_once(x, y, 2, [(-np.inf, np.inf)] * 3, 0.1, np.ones(len(y)))
    z = (x - scorer.mean) / scorer.scale
    logistic = fit(LogisticRegression(C=0.1, tol=1e-10, max_iter=10_000), z, y == 1)
    coef, intercept = weights(logistic)
    assert np.allclose(scorer.coef, coef, atol=1e-4)
    assert np.isclose(-scorer.thresholds[0], intercept, atol=1e-4)


def test_bounds_hold_weights_to_a_sign_or_a_value() -> None:
    x, y = synthetic(400, [1.0, -1.0, 1.0], [-0.5, 0.5], seed=3)
    bounds = [(0.0, np.inf), (0.0, np.inf), (0.25, 0.25)]
    scorer, _ = fit_ordinal(x, y, bounds=bounds, Cs=(10.0,), cv=3, random_state=0)
    assert scorer.coef[0] > 0.5
    assert scorer.coef[1] == 0.0  # it pulls down, so its best allowed weight is 0
    assert scorer.coef[2] == 0.25


def test_a_missing_answer_adds_nothing() -> None:
    x, y = synthetic(200, [1.0, 1.0], [0.0], seed=4)
    scorer, _ = fit_ordinal(x, y, cv=3, random_state=0)
    row = np.array([[np.nan, 2.0]])
    assert scorer.contributions(row)[0, 0] == 0.0
    assert np.isclose(scorer.eta(row)[0], scorer.contributions(row)[0, 1])
    average = scorer.mean[None, :]
    assert np.allclose(scorer.contributions(average), 0.0)


def test_too_few_rows_for_folds_still_fits() -> None:
    x = np.array([[0.0], [1.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 1, 0, 2], dtype=np.intp)  # one row of the top outcome
    scorer, out_of_fold = fit_ordinal(x, y, cv=5)
    assert scorer.cv_auc is None and out_of_fold.shape == (5, 3)


def test_balanced_cuts_split_the_outcomes() -> None:
    scores = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 2.5, 2.6, 2.7, 2.8, 2.9])
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=np.intp)
    cuts = balanced_cuts(scores, y, 3)
    assert np.allclose(cuts, [0.7, 1.85])  # midway between the outcomes' scores
    predicted = (scores[:, None] >= cuts[None, :]).sum(axis=1)
    assert (predicted == y).all()


def test_agreement_measures() -> None:
    y = np.array([0, 1, 2, 2], dtype=np.intp)
    assert quadratic_kappa(y, y, 3) == 1.0
    assert boundary_aucs(y, np.array([0.0, 1.0, 2.0, 3.0]), 3) == [1.0, 1.0]
    assert boundary_aucs(np.zeros(3, np.intp), np.zeros(3), 2) == [None]


def test_a_scorer_checks_its_input() -> None:
    scorer = OrdinalScorer(np.zeros(2), np.ones(2), np.ones(2), np.array([0.0]), C=1.0)
    with pytest.raises(Exception, match="2 checks"):
        scorer.eta(np.zeros((1, 3)))


def balanced(scores: Floats, y: Codes, cuts: Sequence[float]) -> float:
    predicted = (scores[:, None] >= np.array(cuts)[None, :]).sum(axis=1)
    return float(np.mean([(predicted[y == c] == c).mean() for c in np.unique(y)]))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_balanced_cuts_are_the_best_there_are(k: int) -> None:
    rng = np.random.default_rng(k)
    for _ in range(20):
        y = rng.integers(0, k, 14).astype(np.intp)
        scores = np.round(y + rng.normal(0, 1.2, 14), 1)  # ties included
        cuts = balanced_cuts(scores, y, k)
        candidates = [*np.unique(scores).tolist(), np.inf]
        best = max(
            balanced(scores, y, cut)
            for cut in itertools.combinations_with_replacement(candidates, k - 1)
        )
        assert np.all(np.diff(cuts) >= 0)
        assert np.isclose(balanced(scores, y, cuts.tolist()), best)
