"""Answers to ordered outcomes: a cumulative-logit (proportional-odds) combiner.

For decisions with two to four ordered outcomes, such as NO < MAYBE < YES. Each
answer column gets one weight, and the outcomes share them through ``K - 1``
increasing cut-points ``t``:

    P(outcome >= k) = sigmoid(eta - t_k),   eta = z @ coef,

where ``z`` are the answers standardised on the training rows. A missing answer
takes its column's training mean, so it adds nothing to ``eta``. With two
outcomes this is L2 logistic regression. ``C`` is chosen by stratified folds on
the mean AUC of the boundaries (outcome >= k against below, for each k), and
the out-of-fold probabilities are returned with the fit, for reports and for
cut-offs tuned on rows the model did not learn from.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from think_reason_learn.core._sklearn import (
    Codes,
    Floats,
    minimize_bounded,
    stratified_folds,
)
from think_reason_learn.exceptions import DataError

logger = logging.getLogger(__name__)

type Bounds = Sequence[tuple[float, float]]
"""Per column, the lowest and highest weight allowed; ``-inf``/``inf`` for none."""

CS: Final = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
"""Inverse L2 strengths tried by default."""

_MAX_ITER: Final = 2000
_GAP_BOUNDS: Final = (-30.0, 10.0)
"""Bounds on the log gaps between cut-points: an outcome no training row has
pushes its gap towards 0 or infinity, which these keep finite."""
_FLOOR: Final = 1e-12
"""Least probability a row's outcome gets in the loss, so its log is finite."""


@dataclass(frozen=True, slots=True, eq=False)
class OrdinalScorer:
    """A fitted cumulative-logit model over answer columns.

    Args:
        mean: Each column's training mean. A missing answer takes it, so it
            adds nothing.
        scale: Each column's training standard deviation (1 if constant).
        coef: One weight per column, on the standardised answers.
        thresholds: The ``K - 1`` cut-points on ``eta``, increasing.
        C: The inverse L2 strength it was fitted with.
        cv_auc: The mean boundary AUC, out of fold, that chose ``C``; ``None``
            when there were too few rows of an outcome for folds.
    """

    mean: Floats
    scale: Floats
    coef: Floats
    thresholds: Floats
    C: float
    cv_auc: float | None = None

    @property
    def n_outcomes(self) -> int:
        """How many ordered outcomes it predicts."""
        return len(self.thresholds) + 1

    def contributions(self, answers: Floats) -> Floats:
        """Each column's part of ``eta`` per row; 0 for a missing answer.

        An answer at its column's training mean also adds 0, so the parts are
        centred on the average training row.
        """
        answers = np.asarray(answers, dtype=float)
        if answers.ndim != 2 or answers.shape[1] != len(self.coef):
            raise DataError(
                f"Expected answers to {len(self.coef)} checks per row, got shape "
                f"{answers.shape}."
            )
        z = (answers - self.mean) / self.scale
        return np.where(np.isnan(z), 0.0, z) * self.coef

    def eta(self, answers: Floats) -> Floats:
        """The linear score of each row: higher is a higher outcome."""
        return self.contributions(answers).sum(axis=1)

    def cumulative(self, answers: Floats) -> Floats:
        """``P(outcome >= k)`` per row, for ``k = 1 .. K - 1`` (columns)."""
        return _cumulative(self.eta(answers), self.thresholds)

    def proba(self, answers: Floats) -> Floats:
        """The probability of each outcome per row, lowest first; rows sum to 1."""
        return _outcome_proba(self.cumulative(answers))


def fit_ordinal(
    answers: Floats,
    y: Codes,
    *,
    n_outcomes: int | None = None,
    bounds: Bounds | None = None,
    Cs: Sequence[float] = CS,
    cv: int = 5,
    sample_weight: Floats | None = None,
    random_state: int | None = None,
) -> tuple[OrdinalScorer, Floats]:
    """Fit the cumulative-logit model, choosing ``C`` by stratified folds.

    Each ``C`` in ``Cs`` is scored by the mean AUC of the boundaries on
    out-of-fold scores, ties going to the lower out-of-fold log loss (cases
    that every ``C`` ranks perfectly are told apart by how sure it is, not by
    the strongest penalty), and the best is refitted on
    every row. With fewer than two rows of an outcome for folds, the middle
    ``C`` is fitted once, and the returned "out-of-fold" probabilities are the
    fitted rows' own.

    Args:
        answers: One row per case, one column per check; NaN where unanswered.
        y: Each row's outcome, ``0`` (lowest) to ``n_outcomes - 1``.
        n_outcomes: How many outcomes there are, at least 2; by default one
            more than the highest in ``y``. An outcome no row has gets almost
            no probability.
        bounds: Per column, the lowest and highest weight allowed on the
            standardised answers, e.g. ``(0, inf)`` for a check that may only
            raise the outcome, or ``(w, w)`` to hold a weight fixed.
        Cs: Inverse L2 strengths to try.
        cv: Folds for choosing ``C``.
        sample_weight: A weight per row; 1 each by default.
        random_state: Seed for the folds.

    Returns:
        The fitted scorer, and each row's outcome probabilities from a model
        fitted without that row's fold.

    Raises:
        DataError: The shapes disagree, or fewer than two outcomes have rows.
    """
    answers = np.asarray(answers, dtype=float)
    y = np.asarray(y, dtype=np.intp)
    k = n_outcomes if n_outcomes is not None else int(y.max(initial=0)) + 1
    weights = (
        np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    )
    allowed = _checked_bounds(bounds, answers.shape[1])
    _check(answers, y, k, weights)
    if not Cs:
        raise ValueError("Cs needs at least one value.")
    folds = _n_folds(cv, y, k)
    if folds < 2:
        logger.warning(
            "Too few rows of an outcome for folds; fitting once, so the reported "
            "agreement is on rows the model learned from."
        )
        scorer = _fit_once(answers, y, k, allowed, Cs[len(Cs) // 2], weights)
        return scorer, scorer.proba(answers)
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    splits = stratified_folds(splitter, answers, y)
    best: tuple[tuple[float, float], float, Floats] | None = None
    for c in Cs:
        out_of_fold = np.empty((len(y), k))
        eta = np.empty(len(y))
        for train, held_out in splits:
            fold = _fit_once(answers[train], y[train], k, allowed, c, weights[train])
            eta[held_out] = fold.eta(answers[held_out])
            out_of_fold[held_out] = fold.proba(answers[held_out])
        score = (mean_auc(boundary_aucs(y, eta, k)), -_log_loss(out_of_fold, y))
        if best is None or score > best[0]:
            best = (score, c, out_of_fold)
    assert best is not None  # Cs is not empty
    score, c, out_of_fold = best
    scorer = _fit_once(answers, y, k, allowed, c, weights)
    return replace(scorer, cv_auc=score[0]), out_of_fold


def boundary_aucs(y: Codes, scores: Floats, n_outcomes: int) -> list[float | None]:
    """The AUC of ``scores`` for each boundary: outcome >= k against below.

    ``scores`` is one score per row, or one column per boundary (such as
    ``P(outcome >= k)``). A boundary with rows on one side only has ``None``.
    """
    aucs: list[float | None] = []
    for k in range(1, n_outcomes):
        column = scores if scores.ndim == 1 else scores[:, k - 1]
        aucs.append(_auc(column, y >= k))
    return aucs


def mean_auc(aucs: Sequence[float | None]) -> float:
    """The mean of the AUCs that are defined; 0.5 if none is."""
    defined = [a for a in aucs if a is not None]
    return float(np.mean(defined)) if defined else 0.5


def quadratic_kappa(y: Codes, predicted: Codes, n_outcomes: int) -> float | None:
    """Cohen's kappa with quadratic weights: 1 is perfect agreement, 0 chance.

    ``None`` when chance alone agrees perfectly (one outcome only).
    """
    if not len(y):
        return None
    observed = np.zeros((n_outcomes, n_outcomes))
    np.add.at(observed, (y, predicted), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(y)
    index = np.arange(n_outcomes)
    weight = (index[:, None] - index[None, :]) ** 2 / max(n_outcomes - 1, 1) ** 2
    chance = float((weight * expected).sum())
    if chance == 0:
        return None
    return 1 - float((weight * observed).sum()) / chance


def balanced_accuracy(y: Codes, predicted: Codes) -> float | None:
    """The mean recall over the outcomes that have rows; ``None`` with no rows."""
    recalls = [float((predicted[y == k] == k).mean()) for k in np.unique(y)]
    return float(np.mean(recalls)) if recalls else None


def balanced_cuts(scores: Floats, y: Codes, n_outcomes: int) -> Floats:
    """``K - 1`` increasing cut-offs on ``scores`` with the best balanced accuracy.

    A row's outcome is the number of cut-offs its score reaches. Balanced
    accuracy splits into one term per cut-off, so the best cut-offs are found
    exactly, over every distinct score, by dynamic programming. Of equally good
    cut-offs the lowest win; each sits midway to the next lower score, so new
    rows just under a training score are not flipped by a knife edge.
    """
    candidates = np.append(np.unique(scores), np.inf)
    chosen = _best_increasing(_cut_gains(scores, y, n_outcomes, candidates))
    cuts = [
        candidates[g]
        if g == 0 or np.isinf(candidates[g])
        else (candidates[g] + candidates[g - 1]) / 2
        for g in chosen
    ]
    return np.array(cuts, dtype=float)


def _cut_gains(scores: Floats, y: Codes, n_outcomes: int, candidates: Floats) -> Floats:
    """What each cut-off adds to the summed recall, at each candidate (rows).

    Cut-off ``j`` (1-based) separates outcome ``j - 1`` from ``j``: placed at
    ``c`` it gives outcome ``j - 1`` the rows of it below ``c`` and takes those
    rows from outcome ``j``. Outcomes with no rows count for nothing.
    """
    below = np.zeros((n_outcomes, len(candidates)))
    for k in range(n_outcomes):
        members = np.sort(scores[y == k])
        if len(members):
            below[k] = np.searchsorted(members, candidates, side="left") / len(members)
    return np.stack([below[j - 1] - below[j] for j in range(1, n_outcomes)])


def _best_increasing(gains: Floats) -> list[int]:
    """One candidate index per row of ``gains``, non-decreasing, summing highest."""
    n_cuts, n_candidates = gains.shape
    total = gains[0].copy()
    came_from = np.zeros((n_cuts, n_candidates), dtype=np.intp)
    for j in range(1, n_cuts):
        best_before = np.maximum.accumulate(total)
        # The first index reaching each running maximum: ties go to the lowest.
        where = np.zeros(n_candidates, dtype=np.intp)
        for g in range(1, n_candidates):
            where[g] = g if total[g] > best_before[g - 1] else where[g - 1]
        came_from[j] = where
        total = gains[j] + best_before
    chosen = [int(np.argmax(total))]
    for j in range(n_cuts - 1, 0, -1):
        chosen.append(int(came_from[j][chosen[-1]]))
    return chosen[::-1]


def _fit_once(
    answers: Floats,
    y: Codes,
    k: int,
    bounds: Bounds,
    c: float,
    weights: Floats,
) -> OrdinalScorer:
    """The model fitted on these rows with one ``C``, standardised on them."""
    mean, scale = _standardisation(answers)
    z = np.where(np.isnan(answers), 0.0, (answers - mean) / scale)
    m = z.shape[1]
    start = np.concatenate([np.zeros(m), _start_thresholds(y, k, weights)])
    limits: list[tuple[float | None, float | None]] = [
        (None if np.isinf(lo) else lo, None if np.isinf(hi) else hi)
        for lo, hi in bounds
    ]
    limits += [(None, None)] + [_GAP_BOUNDS] * (k - 2)
    start[:m] = np.clip(start[:m], [lo for lo, _ in bounds], [hi for _, hi in bounds])

    def objective(params: Floats) -> tuple[float, Floats]:
        return ordinal_loss(params, z, y, k, weights, c)

    params = minimize_bounded(objective, start, limits, max_iter=_MAX_ITER)
    return OrdinalScorer(
        mean=mean,
        scale=scale,
        coef=params[:m],
        thresholds=_thresholds(params[m:]),
        C=float(c),
    )


def ordinal_loss(
    params: Floats, z: Floats, y: Codes, k: int, weights: Floats, c: float
) -> tuple[float, Floats]:
    """The penalised negative log-likelihood and its gradient at ``params``.

    ``params`` holds the weights, then the first cut-point, then the logs of
    the gaps between cut-points, so the cut-points always increase. The loss
    is the weighted mean over rows plus ``|w|^2 / (2 C n)``, the objective of
    scikit-learn's L2 logistic regression divided by ``C n``.
    """
    n, m = int(z.shape[0]), int(z.shape[1])
    w = params[:m]
    t = _thresholds(params[m:])
    s = _cumulative(z @ w, t)  # P(outcome >= j), j = 1 .. K-1
    d = s * (1 - s)  # its derivative in eta
    # Padded so that outcome y lies between columns y and y + 1.
    ones, zeros = np.ones((n, 1)), np.zeros((n, 1))
    s_padded, d_padded = np.hstack([ones, s, zeros]), np.hstack([zeros, d, zeros])
    rows = np.arange(n)
    upper_s, lower_s = s_padded[rows, y], s_padded[rows, y + 1]
    upper_d, lower_d = d_padded[rows, y], d_padded[rows, y + 1]
    p = np.maximum(upper_s - lower_s, _FLOOR)
    share = weights / n
    loss = float((share * -np.log(p)).sum() + w @ w / (2 * c * n))
    grad_w = z.T @ (share * -(upper_d - lower_d) / p) + w / (c * n)
    # The outcome's probability rises with the cut-point below it and falls
    # with the one above it.
    grad_padded = np.zeros((n, k + 1))
    grad_padded[rows, y] += upper_d / p
    grad_padded[rows, y + 1] -= lower_d / p
    grad_t = (grad_padded[:, 1:k] * share[:, None]).sum(axis=0)
    # t_j = a + sum of exp(gap_i) for i <= j.
    tail = np.cumsum(grad_t[::-1])[::-1]
    grad_gaps = np.exp(params[m + 1 :]) * tail[1:]
    return loss, np.concatenate([grad_w, tail[:1], grad_gaps])


def _log_loss(proba: Floats, y: Codes) -> float:
    """The mean negative log probability of each row's outcome."""
    return float(-np.log(np.maximum(proba[np.arange(len(y)), y], _FLOOR)).mean())


def _thresholds(params: Floats) -> Floats:
    """Cut-points from the first one and the logs of the gaps after it."""
    return np.cumsum(np.concatenate([params[:1], np.exp(params[1:])]))


def _start_thresholds(y: Codes, k: int, weights: Floats) -> Floats:
    """The first cut-point and log gaps that fit the outcome shares alone."""
    counts = np.bincount(y, weights=weights, minlength=k) + 0.5
    below = np.cumsum(counts)[:-1] / counts.sum()  # P(outcome <= j)
    t = np.log(below / (1 - below))
    return np.concatenate([t[:1], np.log(np.diff(t))])


def _cumulative(eta: Floats, thresholds: Floats) -> Floats:
    cumulative: Floats = expit(eta[:, None] - thresholds[None, :])
    return cumulative


def _outcome_proba(cumulative: Floats) -> Floats:
    n = len(cumulative)
    upper = np.hstack([np.ones((n, 1)), cumulative])
    lower = np.hstack([cumulative, np.zeros((n, 1))])
    return np.clip(upper - lower, 0.0, 1.0)


def _standardisation(answers: Floats) -> tuple[Floats, Floats]:
    """Each column's mean and standard deviation, missing answers left out.

    A missing answer counts as the mean, so it does not widen the deviation; a
    column with no answers has mean 0, and a constant one scale 1.
    """
    answered = ~np.isnan(answers)
    counts = answered.sum(axis=0)
    totals = np.where(answered, answers, 0.0).sum(axis=0)
    mean = np.divide(totals, counts, out=np.zeros(answers.shape[1]), where=counts > 0)
    filled = np.where(answered, answers, mean)
    scale = filled.std(axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.0)


def _auc(scores: Floats, positive: npt.NDArray[np.bool_]) -> float | None:
    """The area under the ROC curve, ties counted half; ``None`` if one-sided."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    _, tied = np.unique(scores, return_inverse=True)  # tied scores share a rank
    ranks = (np.bincount(tied, weights=ranks) / np.bincount(tied))[tied]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _n_folds(cv: int, y: Codes, k: int) -> int:
    """Folds for stratified cross-validation: at most the rarest outcome's rows."""
    counts = np.bincount(y, minlength=k)
    return min(cv, int(counts[counts > 0].min()))


def _check(answers: Floats, y: Codes, k: int, weights: Floats) -> None:
    if answers.ndim != 2 or len(answers) != len(y) or len(weights) != len(y):
        raise DataError(
            f"answers {answers.shape}, y ({len(y)},) and sample_weight "
            f"({len(weights)},) need one row per case."
        )
    if k < 2:
        raise ValueError(f"n_outcomes must be at least 2, got {k}.")
    if len(y) and (y.min() < 0 or y.max() >= k):
        raise DataError(f"y needs outcomes from 0 to {k - 1}.")
    if len(np.unique(y)) < 2:
        raise DataError("The rows need at least two different outcomes to learn from.")
    if (weights < 0).any() or not np.isfinite(weights).all():
        raise DataError("sample_weight needs finite weights of 0 or more.")


def _checked_bounds(bounds: Bounds | None, m: int) -> Bounds:
    if bounds is None:
        return [(-np.inf, np.inf)] * m
    if len(bounds) != m or any(lo > hi for lo, hi in bounds):
        raise ValueError(f"bounds needs one (low, high) pair per column, for {m}.")
    return list(bounds)
