"""Choosing a split: class impurity, and the best question at a node."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

type Criterion = Literal["gini", "entropy"]


@dataclass(frozen=True, slots=True)
class Split:
    """A question that splits a node, and its weighted impurity decrease."""

    qid: str
    branches: tuple[str, ...]
    default: str
    decrease: float


def impurity(counts: npt.NDArray[np.float64], criterion: Criterion) -> float:
    """Gini impurity or entropy (bits) of weighted class counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    if criterion == "gini":
        return float(1 - (p**2).sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def best_split(
    answers: Mapping[str, Sequence[str | None]],
    y: npt.NDArray[np.intp],
    weights: npt.NDArray[np.float64],
    *,
    n_classes: int,
    criterion: Criterion,
    min_samples_leaf: int,
    min_impurity_decrease: float,
    total_weight: float,
) -> Split | None:
    """The question whose answers most decrease weighted impurity, if any qualifies.

    A sample without an answer follows the largest branch. A question with
    fewer than two non-empty branches, or a branch under ``min_samples_leaf``,
    does not qualify. The decrease is scikit-learn's: the node's share of the
    total weight times its impurity minus its children's weighted impurity.
    """
    node_weight = weights.sum()
    parent = impurity(_counts(y, weights, n_classes), criterion)
    best: Split | None = None
    for qid, labels in answers.items():
        branches = _branches(labels)
        if branches is None:
            continue
        routed = np.array(
            [branches[0] if a is None else a for a in labels], dtype=object
        )
        masks = [routed == b for b in branches]
        if min(int(m.sum()) for m in masks) < min_samples_leaf:
            continue
        child = _children_impurity(masks, y, weights, n_classes, criterion)
        decrease = node_weight / total_weight * (parent - child)
        if best is None or decrease > best.decrease:
            best = Split(qid, branches, branches[0], decrease)
    return best if best is not None and best.decrease >= min_impurity_decrease else None


def _children_impurity(
    masks: Sequence[npt.NDArray[np.bool_]],
    y: npt.NDArray[np.intp],
    weights: npt.NDArray[np.float64],
    n_classes: int,
    criterion: Criterion,
) -> float:
    """The children's impurity, each weighted by its share of the node's weight."""
    return sum(
        weights[m].sum()
        / weights.sum()
        * impurity(_counts(y[m], weights[m], n_classes), criterion)
        for m in masks
    )


def _counts(
    y: npt.NDArray[np.intp], weights: npt.NDArray[np.float64], n_classes: int
) -> npt.NDArray[np.float64]:
    return np.bincount(y, weights=weights, minlength=n_classes).astype(float)


def _branches(labels: Sequence[str | None]) -> tuple[str, ...] | None:
    """Answer labels present, largest first; ``None`` if fewer than two."""
    present = [label for label in labels if label is not None]
    order = sorted(set(present), key=lambda b: (-present.count(b), b))
    return tuple(order) if len(order) >= 2 else None
