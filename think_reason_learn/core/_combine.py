"""Turning answers into a decision: logistic combiners and F-beta thresholds.

Shared by every model that weighs answers. Thresholds are tuned on out-of-fold
scores, so they carry over to new data; with too few samples of a class for
folds they are tuned on the fitted rows, which can be optimistic.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from typing import Final, TypedDict

import numpy as np
import numpy.typing as npt
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)

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

    @classmethod
    def from_model(cls, model: LogisticRegression, threshold: float) -> LinearCombiner:
        """The weights of a fitted logistic regression, with ``threshold``."""
        return cls(model.coef_[0], float(np.ravel(model.intercept_)[0]), threshold)

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
    cs: int = 10,
) -> LinearCombiner:
    """Fit a regularised logistic regression and tune its F-beta threshold.

    ``C`` (one of ``cs`` values) and ``l1_ratio`` (1 for L1, 0 for L2, between
    for elastic net) are chosen by inner cross-validation; the threshold
    maximises F-beta on out-of-fold probabilities. With too few samples of a
    class for folds, the model is fitted once and the threshold tuned on its
    own scores.
    """
    folds = n_folds(cv, y)
    if folds < 2:
        model = _logistic(1.0, l1_ratios[0], class_weight, random_state)
        model.fit(features, y)
        scores = model.predict_proba(features)[:, 1]
        return LinearCombiner.from_model(model, best_threshold(scores, y, beta)[0])
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    model = _chosen(features, y, splitter, l1_ratios, cs, class_weight, random_state)
    # The threshold is tuned on out-of-fold scores of the model actually used,
    # with the chosen C and l1_ratio, so the scales match.
    out_of_fold = np.asarray(
        cross_val_predict(model, features, y, cv=splitter, method="predict_proba")
    )[:, 1]
    threshold, _ = best_threshold(out_of_fold, y, beta)
    return LinearCombiner.from_model(model.fit(features, y), threshold)


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
    cs: int,
    class_weight: str | None,
    random_state: int | None,
) -> LogisticRegression:
    """An unfitted model with the C and l1_ratio inner cross-validation chose."""
    search = LogisticRegressionCV(
        Cs=cs,
        l1_ratios=list(l1_ratios),  # pyright: ignore[reportArgumentType]  # stubs type it as a sentinel str
        cv=splitter,
        scoring="neg_log_loss",
        class_weight=class_weight,
        random_state=random_state,
        **_SOLVER,
    ).set_params(**_NEW_CV_ATTRIBUTES)
    search.fit(features, y)
    return _logistic(
        float(np.ravel(search.C_)[0]),
        float(np.ravel(search.l1_ratio_)[0]),
        class_weight,
        random_state,
    )


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


def _penalty() -> str:
    """The ``penalty`` to pass: ``"elasticnet"`` before scikit-learn 1.8.

    Later versions deprecate ``penalty``; passing its default avoids a warning.
    """
    match = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    if match is None or (int(match.group(1)), int(match.group(2))) < (1, 8):
        return "elasticnet"
    default: str = inspect.signature(LogisticRegression).parameters["penalty"].default
    return default


class _Solver(TypedDict):
    solver: str
    max_iter: int
    penalty: str


_SOLVER: Final = _Solver(solver="saga", max_iter=5000, penalty=_penalty())
"""Solver settings shared by the cross-validated search and the final model."""

_NEW_CV_ATTRIBUTES: Final[dict[str, bool]] = (
    {"use_legacy_attributes": False}
    if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters
    else {}
)
"""scikit-learn 1.9 warns that LogisticRegressionCV's attributes change; opt in."""
