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
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
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


def _elasticnet_fold(
    train_binary: npt.NDArray[np.int_],
    train_y: npt.NDArray[np.int_],
    test_binary: npt.NDArray[np.int_],
    test_y: npt.NDArray[np.int_],
    *,
    metric: str,
    beta: float,
    cs: Sequence[float],
    l1_ratios: Sequence[float],
    cv: int,
    random_state: int,
) -> tuple[dict[str, float], npt.NDArray[np.int_], npt.NDArray[np.float64], float]:
    """Fit elastic-net on a train fold and evaluate on the test fold.

    Mirrors the ``RRF`` learned-weights aggregator: an elastic-net logistic
    regression (``C``/``l1_ratio`` chosen by inner CV) plus a decision threshold
    tuned on the **train** fold only (no test leakage). The inner-CV fold count
    is capped at the smaller class's size so tiny folds don't crash.

    Returns:
        ``(test_metrics, test_preds, test_proba, threshold)``.
    """
    min_class = int(np.bincount(train_y).min()) if len(train_y) else 0
    eff_cv = max(2, min(cv, min_class)) if min_class >= 2 else 2

    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        Cs=list(cs),  # type: ignore[arg-type]
        l1_ratios=list(l1_ratios),
        cv=eff_cv,
        scoring="roc_auc",
        max_iter=5000,
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    model.fit(train_binary.astype(float), train_y)

    train_proba = model.predict_proba(train_binary.astype(float))[:, 1]
    best_score, best_thr = -1.0, 0.5
    for thr in np.arange(0.05, 0.951, 0.01):
        preds = (train_proba >= thr).astype(np.int_)
        score = RRF._compute_metric(preds, train_y, metric, beta=beta)
        if score > best_score:
            best_score, best_thr = score, float(thr)

    test_proba = model.predict_proba(test_binary.astype(float))[:, 1]
    test_preds = (test_proba >= best_thr).astype(np.int_)
    test_metrics = {
        "precision": RRF._compute_metric(test_preds, test_y, "precision"),
        "recall": RRF._compute_metric(test_preds, test_y, "recall"),
        "f1": RRF._compute_metric(test_preds, test_y, "f1"),
        "f_beta": RRF._compute_metric(test_preds, test_y, "f_beta", beta=beta),
        "accuracy": RRF._compute_metric(test_preds, test_y, "accuracy"),
    }
    return test_metrics, test_preds, test_proba, best_thr


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
    method: Literal["vote", "elasticnet"] = "vote",
    elasticnet_cs: Sequence[float] = (0.05, 0.1, 0.5),
    elasticnet_l1_ratios: Sequence[float] = (0.1, 0.5),
    elasticnet_cv: int = 3,
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
        method: Aggregation evaluated. ``"vote"`` (default) tunes the
            unit-weight top-(K, T) scheme per fold; ``"elasticnet"`` instead
            refits an elastic-net logistic regression per fold (inner CV for
            ``C``/``l1_ratio``, threshold tuned on the train fold) — mirroring
            ``RRF(aggregation_method="elasticnet")``. In ``"elasticnet"`` mode
            ``max_k`` is unused, ``fold_metrics`` reports ``threshold`` instead
            of ``k``/``t``, and ``per_founder`` reports ``probability`` instead
            of ``yes_count``.
        elasticnet_cs: Inverse-regularisation grid for the per-fold inner CV
            (``"elasticnet"`` only).
        elasticnet_l1_ratios: Elastic-net mixing grid for the inner CV.
        elasticnet_cv: Inner-CV folds (capped at the smaller class's size).

    Returns:
        A :class:`CVResult` with fold-level metrics, per-founder
        predictions, and an aggregated summary.
    """
    if method not in ("vote", "elasticnet"):
        raise ValueError("method must be 'vote' or 'elasticnet'")

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

            if method == "elasticnet":
                # Refit a learned-weight model per fold (nested CV inside),
                # threshold tuned on the train fold; evaluate on the test fold.
                test_metrics, preds, proba, thr = _elasticnet_fold(
                    train_binary,
                    train_y,
                    test_binary,
                    test_y,
                    metric=metric,
                    beta=beta,
                    cs=elasticnet_cs,
                    l1_ratios=elasticnet_l1_ratios,
                    cv=elasticnet_cv,
                    random_state=random_state + repeat,
                )
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "threshold": thr,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **test_metrics,
                    }
                )
                for i, sample_pos in enumerate(test_idx):
                    founder_rows.append(
                        {
                            "sample_idx": int(sample_pos),
                            "repeat": repeat,
                            "fold": fold,
                            "y_true": "YES" if test_y[i] else "NO",
                            "y_pred": "YES" if preds[i] else "NO",
                            "probability": float(proba[i]),
                        }
                    )
                continue

            # ----- vote (unit-weight top-(K, T)) path -----
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
