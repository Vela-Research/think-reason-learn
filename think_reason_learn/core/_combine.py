"""Turning answers into a decision: logistic combiners and F-beta thresholds.

Shared by every model that weighs answers. Thresholds are tuned on out-of-fold
scores, so they carry over to new data; with too few samples of a class for
folds they are tuned on the fitted rows, which can be optimistic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Final, TypedDict

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from think_reason_learn.core._sklearn import (
    Bools,
    Floats,
    chosen,
    fit,
    out_of_fold_proba,
    penalty,
    positive_proba,
    weights,
)

logger = logging.getLogger(__name__)


def fbeta(tp: Floats, fp: Floats, fn: Floats, beta: float) -> Floats:
    """F-beta from counts, 0 where undefined."""
    b2 = beta**2
    denominator = (1 + b2) * tp + b2 * fn + fp
    return np.divide(
        (1 + b2) * tp, denominator, out=np.zeros_like(tp), where=denominator > 0
    )


def best_threshold(scores: Floats, y: Bools, beta: float) -> float:
    """The score threshold with the best F-beta for ``y``.

    Each distinct score is tried as a cut (positive at or above it); of equally
    good cuts the lowest wins. The threshold sits midway between the best cut
    and the next lower score seen.
    """
    cuts = np.unique(scores)
    positives, negatives = np.sort(scores[y]), np.sort(scores[~y])
    tp = (len(positives) - np.searchsorted(positives, cuts)).astype(float)
    fp = (len(negatives) - np.searchsorted(negatives, cuts)).astype(float)
    best = int(np.argmax(fbeta(tp, fp, len(positives) - tp, beta)))
    # Midway to the next lower score, so new samples just under a training
    # score are not flipped by a knife-edge threshold.
    below = cuts[best - 1] if best > 0 else cuts[best]
    return float((cuts[best] + below) / 2)


@dataclass(frozen=True, slots=True)
class LinearCombiner:
    """A fitted logistic combiner and its decision threshold."""

    coef: Floats
    intercept: float
    threshold: float

    @classmethod
    def from_model(cls, model: LogisticRegression, threshold: float) -> LinearCombiner:
        """The weights of a fitted logistic regression, with ``threshold``."""
        coef, intercept = weights(model)
        return cls(coef, intercept, threshold)

    def scores(self, features: Floats) -> Floats:
        """Probability of the positive class."""
        return 1 / (1 + np.exp(-(features @ self.coef + self.intercept)))


def fit_logistic(
    features: Floats,
    y: Bools,
    *,
    l1_ratios: tuple[float, ...],
    cv: int,
    beta: float,
    class_weight: str | None,
    random_state: int | None,
    cs: int | tuple[float, ...] = 10,
) -> LinearCombiner:
    """Fit a regularised logistic regression and tune its F-beta threshold.

    ``C`` (one of ``cs``, or of that many on a log grid) and ``l1_ratio`` (1 for
    L1, 0 for L2, between for elastic net) are chosen by inner cross-validation;
    the threshold maximises F-beta on out-of-fold probabilities. With too few
    samples of a class for folds, the model is fitted once and the threshold
    tuned on its own scores.
    """
    folds = n_folds(cv, y)
    if folds < 2:
        model = fit(
            _logistic(1.0, l1_ratios[0], class_weight, random_state), features, y
        )
        scores = positive_proba(model, features)
        return LinearCombiner.from_model(model, best_threshold(scores, y, beta))
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    model = _chosen(features, y, splitter, l1_ratios, cs, class_weight, random_state)
    # The threshold is tuned on out-of-fold scores of the model actually used,
    # with the chosen C and l1_ratio, so the scales match.
    out_of_fold = out_of_fold_proba(model, features, y, splitter)
    threshold = best_threshold(out_of_fold, y, beta)
    return LinearCombiner.from_model(fit(model, features, y), threshold)


async def fit_combiner(
    features: Floats,
    y: Bools,
    *,
    l1_ratios: tuple[float, ...],
    cv: int,
    beta: float,
    class_weight: str | None,
    random_state: int | None,
    cs: int | tuple[float, ...] = 10,
    drop_zero: bool = False,
) -> tuple[LinearCombiner, npt.NDArray[np.intp]]:
    """``fit_logistic`` off the event loop: the combiner, and the columns it weighs.

    With ``drop_zero`` the columns whose weight is zero are dropped. A warning
    is logged when none is left: the model then gives every sample one score.
    """
    fitted = await asyncio.to_thread(
        partial(
            fit_logistic,
            features,
            y,
            l1_ratios=l1_ratios,
            cv=cv,
            beta=beta,
            class_weight=class_weight,
            random_state=random_state,
            cs=cs,
        )
    )
    used = np.flatnonzero(fitted.coef) if drop_zero else np.arange(len(fitted.coef))
    if not len(used):
        logger.warning(
            "Every weight is zero, so the model gives every sample the same score. "
            "Try more samples or other rules."
        )
    return LinearCombiner(fitted.coef[used], fitted.intercept, fitted.threshold), used


def n_folds(cv: int, y: Bools) -> int:
    """Folds for stratified cross-validation: at most the smaller class's size."""
    return min(cv, int(np.bincount(y.astype(int), minlength=2).min()))


def held_out_rows(
    positive: Bools, used: npt.NDArray[np.intp], cv: int
) -> npt.NDArray[np.intp]:
    """The rows not in ``used``, if each class has enough of them for ``cv`` folds.

    Otherwise every row, with a warning: what is fitted on them (weights, a
    threshold) is then tuned on samples the model already learned from.
    """
    rest = np.setdiff1d(np.arange(len(positive)), used).astype(np.intp)
    if n_folds(cv, positive[rest]) >= 2:
        return rest
    logger.warning(
        "Too few unused samples of each class to fit on; fitting on every sample, "
        "so the threshold may be optimistic. Use fewer samples to learn from."
    )
    return np.arange(len(positive), dtype=np.intp)


def _chosen(
    features: Floats,
    y: Bools,
    splitter: StratifiedKFold,
    l1_ratios: tuple[float, ...],
    cs: int | tuple[float, ...],
    class_weight: str | None,
    random_state: int | None,
) -> LogisticRegression:
    """An unfitted model with the C and l1_ratio inner cross-validation chose."""
    search = LogisticRegressionCV(
        Cs=cs if isinstance(cs, int) else list(cs),  # pyright: ignore[reportArgumentType]  # a list of Cs works; pyright reads int from the default
        l1_ratios=list(l1_ratios),
        cv=splitter,
        scoring="neg_log_loss",
        class_weight=class_weight,
        random_state=random_state,
        **_SOLVER,
    )
    c, l1_ratio = chosen(search, features, y)
    return _logistic(c, l1_ratio, class_weight, random_state)


def _logistic(
    c: float, l1_ratio: float, class_weight: str | None, random_state: int | None
) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        random_state=random_state,
        **_SOLVER,
    )


class _Solver(TypedDict):
    solver: str
    max_iter: int
    penalty: str


_SOLVER: Final = _Solver(solver="saga", max_iter=5000, penalty=penalty())
"""Solver settings shared by the cross-validated search and the final model."""
