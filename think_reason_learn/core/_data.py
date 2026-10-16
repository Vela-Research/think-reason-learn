"""Samples as text, labels as codes, and class-balanced examples for prompts.

Every model reads its input through these, so a row is rendered the same way in
every prompt, at fit and at predict.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from think_reason_learn.exceptions import DataError

type TextInput = pd.DataFrame | pd.Series | Sequence[str] | npt.NDArray[np.str_]
"""Samples: a DataFrame (each row rendered as ``column: value`` lines), or one
text per sample."""

type LabelInput = Sequence[object] | pd.Series | npt.NDArray[np.generic]
"""One label per sample."""

type ClassRatio = Literal["balanced"] | Mapping[str, float]
"""How the classes share the examples shown in a prompt."""


def to_texts(X: TextInput) -> tuple[list[str], list[str] | None]:
    """Each sample as text, and the column names when ``X`` is a DataFrame.

    Missing values are left out of a row's text.

    Raises:
        DataError: ``X`` is empty or not one of the accepted types.
    """
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
        texts = [_row_text(names, row) for row in X.itertuples(index=False, name=None)]
    elif isinstance(X, (pd.Series, np.ndarray, list, tuple)):
        names = None
        texts = [str(x) for x in X]
    else:
        raise DataError(
            f"X must be a DataFrame or a sequence of texts, got {type(X).__name__}."
        )
    if not texts:
        raise DataError("X has no samples.")
    return texts, names


def encode_labels(
    y: LabelInput,
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.intp]]:
    """The sorted classes, and each label's index among them.

    Raises:
        DataError: ``y`` has fewer than two classes or missing labels.
    """
    labels = pd.Series(list(y), dtype=object)
    if labels.isna().any():
        raise DataError("y has missing labels.")
    classes, codes = np.unique(labels.to_numpy(), return_inverse=True)
    if len(classes) < 2:
        raise DataError(f"y needs at least two classes, got {list(classes)}.")
    return classes, codes.astype(np.intp)


def example_batches(
    codes: npt.NDArray[np.intp],
    classes: npt.NDArray[np.object_],
    *,
    n_batches: int,
    batch_size: int,
    class_ratio: ClassRatio,
    rng: np.random.Generator,
) -> list[npt.NDArray[np.intp]]:
    """Sample indices for prompts, ``batch_size`` each, classes in ``class_ratio``.

    Samples are drawn without repeats until a class runs out, then that class
    is reshuffled and reused, so a rare class still reaches every batch. A
    class's share is filled from the others when it has no samples at all.
    """
    weights = _weights(classes, class_ratio)
    pools = [np.flatnonzero(codes == c) for c in range(len(classes))]
    queues = [rng.permutation(pool) for pool in pools]
    batches: list[npt.NDArray[np.intp]] = []
    for _ in range(n_batches):
        wanted = _split(batch_size, weights, [len(p) > 0 for p in pools])
        picked = [_take(queues, pools, c, n, rng) for c, n in enumerate(wanted)]
        batch = np.concatenate(picked)
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def balanced_subset(
    codes: npt.NDArray[np.intp],
    n_classes: int,
    size: int,
    rng: np.random.Generator,
    *,
    max_share: float = 1.0,
) -> npt.NDArray[np.intp]:
    """Up to ``size`` distinct samples, split evenly across the classes.

    A class gives at most ``max_share`` of its samples; one with too few gives
    what it has.
    """
    per_class = math.ceil(size / n_classes)
    chosen = [
        rng.permutation(members)[: min(per_class, int(len(members) * max_share))]
        for members in (np.flatnonzero(codes == c) for c in range(n_classes))
    ]
    return np.sort(np.concatenate(chosen)).astype(np.intp)


def count(value: int | float, n: int) -> int:
    """An int as is; a float as that fraction of ``n``, rounded up.

    For settings like ``min_samples_leaf`` that take a count or a share.
    """
    return value if isinstance(value, int) else math.ceil(value * n)


def _row_text(names: Sequence[str], values: tuple[object, ...]) -> str:
    return "\n".join(
        f"{name}: {value}" for name, value in zip(names, values) if not _missing(value)
    )


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _weights(
    classes: npt.NDArray[np.object_], class_ratio: ClassRatio
) -> npt.NDArray[np.float64]:
    if class_ratio == "balanced":
        return np.full(len(classes), 1 / len(classes))
    unknown = set(class_ratio) - {str(c) for c in classes}
    if unknown:
        raise ValueError(f"class_ratio names unknown classes: {sorted(unknown)}.")
    weights = np.array([class_ratio.get(str(c), 0.0) for c in classes], dtype=float)
    if weights.sum() <= 0 or (weights < 0).any():
        raise ValueError("class_ratio needs non-negative shares that sum above 0.")
    return weights / weights.sum()


def _split(
    total: int, weights: npt.NDArray[np.float64], available: list[bool]
) -> list[int]:
    """``total`` split by ``weights`` over the classes that have samples."""
    usable = np.where(available, weights, 0.0)
    if usable.sum() == 0:
        usable = np.array(available, dtype=float)
    shares = usable / usable.sum() * total
    counts = np.floor(shares).astype(int)
    for c in np.argsort(counts - shares)[: total - counts.sum()]:
        counts[c] += 1
    return counts.tolist()


def _take(
    queues: list[npt.NDArray[np.intp]],
    pools: list[npt.NDArray[np.intp]],
    c: int,
    n: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.intp]:
    taken: list[npt.NDArray[np.intp]] = []
    while n > 0 and len(pools[c]):
        if not len(queues[c]):
            queues[c] = rng.permutation(pools[c])
        step = queues[c][:n]
        queues[c] = queues[c][len(step) :]
        taken.append(step)
        n -= len(step)
    return np.concatenate(taken) if taken else np.empty(0, dtype=np.intp)
