"""Turning answers into a decision: logistic combiners and F-beta thresholds.

Shared by every model that weighs answers. Thresholds are tuned on out-of-fold
scores, never on the rows a model was fitted on, so they carry over to new data.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

type Floats = npt.NDArray[np.float64]
type Bools = npt.NDArray[np.bool_]


def fbeta(tp: Floats, fp: Floats, fn: Floats, beta: float) -> Floats:
    """F-beta from counts, 0 where undefined."""
    b2 = beta**2
    denominator = (1 + b2) * tp + b2 * fn + fp
    return np.divide(
        (1 + b2) * tp, denominator, out=np.zeros_like(tp), where=denominator > 0
    )


def best_threshold(scores: Floats, y: Bools, beta: float) -> tuple[float, float]:
    """The score threshold with the best F-beta for ``y``, and that F-beta.

    The threshold sits midway between the best cut's score and the next lower
    score seen.
    """
    precision, recall, thresholds = precision_recall_curve(y, scores)
    b2 = beta**2
    denominator = b2 * precision[:-1] + recall[:-1]
    f = np.divide(
        (1 + b2) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(f))
    # Midway to the next lower score, so new samples just under a training
    # score are not flipped by a knife-edge threshold.
    below = thresholds[best - 1] if best > 0 else thresholds[best]
    return float((thresholds[best] + below) / 2), float(f[best])


@dataclass(frozen=True, slots=True)
class LinearCombiner:
    """A fitted logistic combiner and its decision threshold."""

    coef: Floats
    intercept: float
    threshold: float

    def scores(self, features: Floats) -> Floats:
        """Probability of the positive class."""
        return 1 / (1 + np.exp(-(features @ self.coef + self.intercept)))


def fit_logistic(
    features: Floats,
    y: Bools,
    *,
    l1_ratios: tuple[float, ...],
    cs: int,
    cv: int,
    beta: float,
    class_weight: str | None,
    random_state: int | None,
) -> LinearCombiner:
    """Fit a regularised logistic regression and tune its F-beta threshold.

    ``C`` and ``l1_ratio`` (1 for L1, 0 for L2, between for elastic net) are
    chosen by inner cross-validation; the threshold maximises F-beta on
    out-of-fold probabilities. With too few samples of a class for folds, the
    model is fitted once and the threshold tuned on its own scores.
    """
    folds = n_folds(cv, y)
    if folds < 2:
        model = _logistic(1.0, l1_ratios[0], class_weight, random_state).fit(
            features, y
        )
        threshold, _ = best_threshold(model.predict_proba(features)[:, 1], y, beta)
        return LinearCombiner(
            model.coef_[0], float(np.ravel(model.intercept_)[0]), threshold
        )
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    chosen = (
        LogisticRegressionCV(
            Cs=cs,
            l1_ratios=list(l1_ratios),  # pyright: ignore[reportArgumentType]  # stubs type it as a sentinel str
            solver="saga",
            cv=splitter,
            scoring="neg_log_loss",
            class_weight=class_weight,
            max_iter=5000,
            random_state=random_state,
            penalty=_PENALTY,
        )
        .set_params(**_NEW_CV_ATTRIBUTES)
        .fit(features, y)
    )
    # The threshold is tuned on out-of-fold scores of the model actually used,
    # with the chosen C and l1_ratio, so the scales match.
    model = _logistic(
        float(np.ravel(chosen.C_)[0]),
        float(np.ravel(chosen.l1_ratio_)[0]),
        class_weight,
        random_state,
    )
    out_of_fold = np.asarray(
        cross_val_predict(model, features, y, cv=splitter, method="predict_proba")
    )[:, 1]
    threshold, _ = best_threshold(out_of_fold, y, beta)
    model.fit(features, y)
    return LinearCombiner(
        model.coef_[0], float(np.ravel(model.intercept_)[0]), threshold
    )


def _logistic(
    c: float, l1_ratio: float, class_weight: str | None, random_state: int | None
) -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        C=c,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        max_iter=5000,
        random_state=random_state,
        penalty=_PENALTY,
    )


def n_folds(cv: int, y: Bools) -> int:
    """Folds for stratified cross-validation: at most the smaller class's size."""
    return min(cv, int(np.bincount(y.astype(int), minlength=2).min()))


def _penalty(version: str = sklearn.__version__) -> str:
    """The ``penalty`` to pass: ``"elasticnet"`` before scikit-learn 1.8.

    Later versions deprecate ``penalty``; passing its default avoids a warning.
    """
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None or (int(match.group(1)), int(match.group(2))) < (1, 8):
        return "elasticnet"
    default: str = inspect.signature(LogisticRegression).parameters["penalty"].default
    return default


_PENALTY: Final = _penalty()

_NEW_CV_ATTRIBUTES: Final[dict[str, bool]] = (
    {"use_legacy_attributes": False}
    if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters
    else {}
)
"""scikit-learn 1.9 warns that LogisticRegressionCV's attributes change; opt in."""
