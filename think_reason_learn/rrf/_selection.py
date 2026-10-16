"""Choosing RRF's rules: rank, filter, and tune the vote.

Questions are ranked by F-beta, weak and redundant ones dropped, and the vote
(YES when at least T of the top N rules say YES) tuned.

Pure functions over a YES/NO answer matrix: rows are samples, columns questions.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import numpy.typing as npt

from think_reason_learn.llm import YES_AT
from sklearn.model_selection import StratifiedKFold

from think_reason_learn.core._combine import Bools, Floats, fbeta, n_folds


def question_stats(
    answers: Bools, y: Bools, beta: float
) -> tuple[Floats, Floats, Floats]:
    """Precision, recall and F-beta of each question read as "YES means positive"."""
    tp = (answers & y[:, None]).sum(axis=0).astype(float)
    fp = (answers & ~y[:, None]).sum(axis=0).astype(float)
    fn = float(y.sum()) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
    return precision, recall, fbeta(tp, fp, fn, beta)


def rank(
    answers: Bools,
    y: Bools,
    *,
    beta: float,
    drop_below_base_rate: bool,
    hamming_threshold: float,
) -> list[int]:
    """Question indices by F-beta, without weak or redundant questions.

    A question is weak when its precision is at most the base rate, and
    redundant when its answers differ from a better question's on fewer than
    ``hamming_threshold`` of the samples (the paper's Hamming filter).
    """
    precision, _, scores = question_stats(answers, y, beta)
    order = [int(j) for j in np.argsort(-scores, kind="stable")]
    if drop_below_base_rate:
        order = [j for j in order if precision[j] > y.mean()]
    kept: list[int] = []
    for j in order:
        if (
            kept
            and (answers[:, [j]] != answers[:, kept]).mean(axis=0).min()
            < hamming_threshold
        ):
            continue
        kept.append(j)
    return kept


def vote_grid(ranked: Bools, y: Bools, beta: float) -> Floats:
    """F-beta of the vote per N (rows: the top N rules) and T (columns: T or more)."""
    n_rules = ranked.shape[1]
    grid = np.zeros((n_rules, n_rules))
    votes = np.cumsum(ranked, axis=1, dtype=np.int_)
    positives = float(y.sum())
    for n in range(1, n_rules + 1):
        counted = votes[:, n - 1]
        tp, fp = _at_least(counted[y], n), _at_least(counted[~y], n)
        grid[n - 1, :n] = fbeta(tp, fp, positives - tp, beta)
    return grid


def _at_least(votes: npt.NDArray[np.int_], n: int) -> Floats:
    """For T = 1..n, how many of ``votes`` are at least T."""
    return np.cumsum(np.bincount(votes, minlength=n + 1)[::-1])[::-1][1:].astype(float)


def tune_vote(
    answers: Bools,
    y: Bools,
    *,
    beta: float,
    drop_below_base_rate: bool,
    hamming_threshold: float,
    max_rules: int | None,
    cv: int,
    random_state: int | None,
) -> tuple[list[int], int, int]:
    """The ranked rules, and the (N, T) with the best cross-validated F-beta.

    Each fold ranks the questions on its training part and scores every (N, T)
    on its held-out part, so the vote is tuned on answers it was not ranked on.
    """
    ranker = partial(
        rank,
        beta=beta,
        drop_below_base_rate=drop_below_base_rate,
        hamming_threshold=hamming_threshold,
    )
    ranked = ranker(answers, y)[:max_rules]
    if not ranked:
        return [], 0, 0
    folds = n_folds(cv, y)
    grid = (
        vote_grid(answers[:, ranked], y, beta)
        if folds < 2
        else _cross_validated_grid(
            answers, y, len(ranked), ranker, folds, beta, random_state
        )
    )
    n, t = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return ranked, int(n) + 1, int(t) + 1


def _cross_validated_grid(
    answers: Bools,
    y: Bools,
    size: int,
    ranker: Callable[[Bools, Bools], list[int]],
    folds: int,
    beta: float,
    random_state: int | None,
) -> Floats:
    """Each (N, T)'s F-beta on held-out folds, ranked on the rest.

    Only cells every fold could score compete, so a cell scored by one lucky
    fold cannot beat cells averaged over all of them.
    """
    total, seen = np.zeros((size, size)), np.zeros((size, size))
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    for train, held_out in splitter.split(answers, y):
        fold_ranked = ranker(answers[train], y[train])[:size]
        k = len(fold_ranked)
        total[:k, :k] += vote_grid(answers[held_out][:, fold_ranked], y[held_out], beta)
        seen[:k, :k] += 1
    grid = np.divide(total, seen, out=np.zeros_like(total), where=seen > 0)
    complete = seen == folds
    return np.where(complete, grid, -np.inf) if complete.any() else grid


def as_yes(probabilities: npt.NDArray[np.float64]) -> Bools:
    """Answers as YES (probability of yes at least 0.5); missing answers are NO."""
    return np.nan_to_num(probabilities, nan=0.0) >= YES_AT
