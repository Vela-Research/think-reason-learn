"""Typed doors into scikit-learn and joblib, whose own annotations are partial.

Each function states the types this library relies on. The few ``pyright:
ignore`` comments that scikit-learn's partial annotations need sit here, so the
rest of the code is checked strictly.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Final

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
from sklearn.base import (
    clone,  # pyright: ignore[reportUnknownVariableType]  # partial annotation
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# scikit-learn annotates these three only in part.
from sklearn.metrics import (
    precision_recall_curve,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.utils import Tags
from sklearn.utils.estimator_checks import (  # partial annotations
    check_get_params_invariance,  # pyright: ignore[reportUnknownVariableType]
    check_no_attributes_set_in_init,  # pyright: ignore[reportUnknownVariableType]
    check_set_params,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.utils.validation import (
    check_is_fitted,  # pyright: ignore[reportUnknownVariableType]
)

type Floats = npt.NDArray[np.float64]
type Bools = npt.NDArray[np.bool_]

_NEW_CV_ATTRIBUTES: Final = (
    "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters
)


def clone_estimator[E](estimator: E) -> E:
    """An unfitted copy with the same settings, as ``sklearn.clone``."""
    copied: object = clone(estimator)  # pyright: ignore[reportUnknownVariableType, reportArgumentType]  # partial annotation
    if not isinstance(copied, type(estimator)):
        raise TypeError("sklearn.clone returned an object of another type.")
    return copied


def check_constructor(name: str, make: Callable[[], object]) -> None:
    """scikit-learn's checks that a constructor stores its settings untouched."""
    for check in (  # pyright: ignore[reportUnknownVariableType]  # partial annotations
        check_no_attributes_set_in_init,
        check_get_params_invariance,
        check_set_params,
    ):
        check(name, make())


def text_tags(tags: object) -> Tags:
    """scikit-learn's ``tags``, marked for text input and answers that vary."""
    if not isinstance(tags, Tags):
        raise TypeError(f"Expected scikit-learn Tags, got {type(tags).__name__}.")
    tags.input_tags.string = True
    tags.non_deterministic = True
    return tags


def require_fitted(estimator: object) -> None:
    """Raise ``NotFittedError`` unless ``estimator`` is fitted."""
    check_is_fitted(estimator)  # pyright: ignore[reportUnknownVariableType, reportArgumentType]  # sklearn types the estimator loosely


def pr_curve(y: Bools, scores: Floats) -> tuple[Floats, Floats, Floats]:
    """Precision, recall and thresholds, as ``precision_recall_curve``."""
    precision, recall, thresholds = precision_recall_curve(y, scores)  # pyright: ignore[reportUnknownVariableType]  # partial return annotation
    return (
        np.asarray(precision, dtype=float),
        np.asarray(recall, dtype=float),
        np.asarray(thresholds, dtype=float),
    )


def stratified_folds(
    splitter: StratifiedKFold, features: npt.NDArray[np.generic], y: Bools
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """(train, held-out) row indices of each fold."""
    folds = splitter.split(features, y)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # partial annotation
    return [
        (np.asarray(train, dtype=np.intp), np.asarray(held_out, dtype=np.intp))
        for train, held_out in folds  # pyright: ignore[reportUnknownVariableType]  # as above
    ]


def fit(model: LogisticRegression, features: Floats, y: Bools) -> LogisticRegression:
    """``model`` fitted on ``features`` and ``y``."""
    model.fit(features, y)  # pyright: ignore[reportUnknownMemberType]  # partial annotation
    return model


def positive_proba(model: LogisticRegression, features: Floats) -> Floats:
    """The fitted model's probability of the positive class."""
    proba = model.predict_proba(features)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # partial annotation
    return np.asarray(proba, dtype=float)[:, 1]


def weights(model: LogisticRegression) -> tuple[Floats, float]:
    """The fitted model's coefficients and intercept."""
    coef = np.asarray(model.coef_, dtype=float)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # partial annotation
    intercept = np.asarray(model.intercept_, dtype=float)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # partial annotation
    return coef[0], float(intercept.ravel()[0])


def out_of_fold_proba(
    model: LogisticRegression, features: Floats, y: Bools, cv: StratifiedKFold
) -> Floats:
    """Positive-class probabilities, each from a model not fitted on its row."""
    proba = cross_val_predict(model, features, y, cv=cv, method="predict_proba")  # pyright: ignore[reportUnknownVariableType]  # partial annotation
    return np.asarray(proba, dtype=float)[:, 1]


def chosen(
    search: LogisticRegressionCV, features: Floats, y: Bools
) -> tuple[float, float]:
    """The ``C`` and ``l1_ratio`` a cross-validated search chooses, once fitted."""
    if _NEW_CV_ATTRIBUTES:  # scikit-learn 1.9 warns that its attributes change
        search.set_params(use_legacy_attributes=False)  # pyright: ignore[reportUnknownMemberType]  # partial annotation
    search.fit(features, y)  # pyright: ignore[reportUnknownMemberType]  # partial annotation
    c = np.asarray(search.C_, dtype=float).ravel()[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # partial annotation
    l1_ratio = np.asarray(search.l1_ratio_, dtype=float).ravel()[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # partial annotation
    return float(c), float(l1_ratio)


def dump(value: object, path: Path) -> None:
    """Write ``value`` to ``path`` with joblib."""
    joblib.dump(value, path)  # pyright: ignore[reportUnknownMemberType]  # joblib has no annotations


def load(path: Path) -> object:
    """Read a value written by ``dump``. Only load files you trust."""
    value: object = joblib.load(path)  # pyright: ignore[reportUnknownMemberType]  # joblib has no annotations
    return value
