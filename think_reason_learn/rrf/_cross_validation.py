"""Cross-validation for RRF aggregation tuning.

Provides :func:`cross_validate_aggregation`, a standalone function that
evaluates (K, T) aggregation quality using repeated stratified k-fold
cross-validation.  Operates entirely on a pre-computed answer matrix and
labels — no LLM calls are made.

.. warning::
    The *answer_matrix* passed to :func:`cross_validate_aggregation` must
    **exclude** any samples whose labels were exposed during question
    generation.  Including them would leak information and produce
    over-optimistic metrics.

Example::

    from think_reason_learn.rrf import cross_validate_aggregation

    result = cross_validate_aggregation(
        answer_matrix=answers_df,   # n_samples × n_questions, "YES"/"NO"
        y=labels,                   # "YES"/"NO" per sample
        n_splits=10,
        n_repeats=10,
        metric="f_beta",
        beta=0.5,
    )
    print(result.summary)
    print(result.fold_metrics)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ._rrf import RRF


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Results from cross-validated aggregation evaluation.

    Attributes:
        fold_metrics: One row per (repeat, fold).  Columns: ``repeat``,
            ``fold``, ``k``, ``t``, ``precision``, ``recall``, ``f1``,
            ``f_beta``, ``accuracy``, ``n_train``, ``n_test``.
        per_founder: One row per (sample, repeat).  Columns:
            ``sample_idx``, ``repeat``, ``fold``, ``y_true``, ``y_pred``,
            ``yes_count``.
        summary: Mean and standard deviation of each metric across folds.
            Keys follow the pattern ``"<metric>_mean"`` and
            ``"<metric>_std"`` (e.g. ``"f_beta_mean"``).
    """

    fold_metrics: pd.DataFrame
    per_founder: pd.DataFrame
    summary: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score_questions(
    binary: npt.NDArray[np.int_],
    y_binary: npt.NDArray[np.int_],
    beta: float,
) -> npt.NDArray[np.float64]:
    """Compute per-question f-beta scores.

    Args:
        binary: (n_samples, n_questions) array of 0/1 answers.
        y_binary: (n_samples,) array of 0/1 true labels.
        beta: Beta parameter for F-beta.

    Returns:
        (n_questions,) array of f-beta scores.
    """
    n_questions = binary.shape[1]
    scores = np.zeros(n_questions, dtype=np.float64)
    beta_sq = beta * beta

    true_pos_mask = y_binary == 1

    for q in range(n_questions):
        pred_yes = binary[:, q] == 1
        tp = int((pred_yes & true_pos_mask).sum())
        fp = int((pred_yes & ~true_pos_mask).sum())
        fn = int((~pred_yes & true_pos_mask).sum())

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        denom = beta_sq * p + r
        scores[q] = (1 + beta_sq) * p * r / denom if denom else 0.0

    return scores


def _grid_search_kt(
    binary: npt.NDArray[np.int_],
    y_binary: npt.NDArray[np.int_],
    question_order: npt.NDArray[np.intp],
    metric: str,
    beta: float,
    max_k: int | None,
) -> tuple[int, int, float]:
    """Find best (K, T) via grid search on pre-sorted binary matrix.

    Args:
        binary: (n_samples, n_questions) binary answers.
        y_binary: (n_samples,) binary labels.
        question_order: Indices into ``binary`` columns sorted by
            descending f-beta score.
        metric: Optimisation metric (passed to ``RRF._compute_metric``).
        beta: Beta for f-beta metric.
        max_k: Maximum K to consider.  ``None`` = all questions.

    Returns:
        ``(best_k, best_t, best_score)``
    """
    sorted_binary = binary[:, question_order]
    cumsum = np.cumsum(sorted_binary, axis=1)
    q_count = len(question_order)
    cap = min(max_k, q_count) if max_k is not None else q_count

    best_score, best_k, best_t = -1.0, 1, 1
    for k in range(1, cap + 1):
        yes_at_k = cumsum[:, k - 1]
        for t_val in range(1, k + 1):
            preds = (yes_at_k >= t_val).astype(np.int_)
            score = RRF._compute_metric(preds, y_binary, metric, beta=beta)
            if score > best_score:
                best_score, best_k, best_t = score, k, t_val

    return best_k, best_t, best_score


def _evaluate_fold(
    binary: npt.NDArray[np.int_],
    y_binary: npt.NDArray[np.int_],
    question_order: npt.NDArray[np.intp],
    k: int,
    t: int,
    beta: float,
) -> dict[str, float]:
    """Evaluate a single (K, T) on a held-out fold.

    Returns:
        Dict with keys: precision, recall, f1, f_beta, accuracy.
    """
    sorted_binary = binary[:, question_order]
    yes_counts = sorted_binary[:, :k].sum(axis=1)
    preds = (yes_counts >= t).astype(np.int_)

    return {
        "precision": RRF._compute_metric(preds, y_binary, "precision"),
        "recall": RRF._compute_metric(preds, y_binary, "recall"),
        "f1": RRF._compute_metric(preds, y_binary, "f1"),
        "f_beta": RRF._compute_metric(preds, y_binary, "f_beta", beta=beta),
        "accuracy": RRF._compute_metric(preds, y_binary, "accuracy"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cross_validate_aggregation(
    answer_matrix: pd.DataFrame,
    y: Sequence[str],
    *,
    n_splits: int = 10,
    n_repeats: int = 10,
    metric: str = "f_beta",
    beta: float = 0.5,
    max_k: int | None = None,
    random_state: int = 42,
) -> CVResult:
    """Evaluate RRF aggregation via repeated stratified k-fold CV.

    For each fold the function:

    1. Scores every question on the **training** split (f-beta).
    2. Ranks questions by that score (descending).
    3. Grid-searches (K, T) on the training split.
    4. Evaluates with the chosen (K, T) on the **test** split.

    No LLM calls are made — everything is computed from the pre-built
    answer matrix.

    Args:
        answer_matrix: ``(n_samples, n_questions)`` DataFrame with
            ``"YES"``/``"NO"`` values.  **Must not include samples whose
            labels were seen during question generation.**
        y: True labels (``"YES"``/``"NO"``), one per sample.
        n_splits: Number of folds per repeat.
        n_repeats: Number of times to repeat the k-fold split.
        metric: Metric to optimise when tuning (K, T).  Any value
            accepted by :meth:`RRF._compute_metric` (e.g. ``"f_beta"``,
            ``"f1"``, ``"precision"``).
        beta: Beta parameter for F-beta scoring and tuning.
        max_k: Upper bound on K during grid search.  ``None`` = use all
            questions.
        random_state: Base random seed; each repeat uses
            ``random_state + repeat``.

    Returns:
        A :class:`CVResult` with fold-level metrics, per-founder
        predictions, and an aggregated summary.
    """
    y_arr = np.array([1 if yi == "YES" else 0 for yi in y], dtype=np.int_)
    binary_full: npt.NDArray[np.int_] = np.asarray(
        answer_matrix.apply(lambda col: (col == "YES").astype(np.int_)).values
    )
    fold_rows: list[dict[str, object]] = []
    founder_rows: list[dict[str, object]] = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )
        for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(y_arr)), y_arr)
        ):
            train_binary: npt.NDArray[np.int_] = binary_full[train_idx]
            train_y: npt.NDArray[np.int_] = y_arr[train_idx]
            test_binary: npt.NDArray[np.int_] = binary_full[test_idx]
            test_y: npt.NDArray[np.int_] = y_arr[test_idx]

            # 1. Score questions on train fold
            q_scores = _score_questions(train_binary, train_y, beta)

            # 2. Rank questions (descending f-beta)
            q_order: npt.NDArray[np.intp] = np.argsort(-q_scores)

            # 3. Grid search (K, T) on train fold
            best_k, best_t, _ = _grid_search_kt(
                train_binary, train_y, q_order, metric, beta, max_k
            )

            # 4. Evaluate on test fold
            test_metrics = _evaluate_fold(
                test_binary, test_y, q_order, best_k, best_t, beta
            )

            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "k": best_k,
                    "t": best_t,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **test_metrics,
                }
            )

            # 5. Per-founder predictions
            sorted_test: npt.NDArray[np.int_] = test_binary[:, q_order]
            yes_counts = sorted_test[:, :best_k].sum(axis=1)
            preds = (yes_counts >= best_t).astype(np.int_)

            for i, sample_pos in enumerate(test_idx):
                founder_rows.append(
                    {
                        "sample_idx": int(sample_pos),
                        "repeat": repeat,
                        "fold": fold,
                        "y_true": "YES" if test_y[i] else "NO",
                        "y_pred": "YES" if preds[i] else "NO",
                        "yes_count": int(yes_counts[i]),
                    }
                )

    # Build result DataFrames
    fold_metrics = pd.DataFrame(fold_rows)
    per_founder = pd.DataFrame(founder_rows)

    # Summary: mean and std of each metric across folds
    metric_cols = ["precision", "recall", "f1", "f_beta", "accuracy"]
    summary: dict[str, float] = {}
    for col in metric_cols:
        summary[f"{col}_mean"] = float(fold_metrics[col].mean())
        summary[f"{col}_std"] = float(fold_metrics[col].std())

    return CVResult(
        fold_metrics=fold_metrics,
        per_founder=per_founder,
        summary=summary,
    )
