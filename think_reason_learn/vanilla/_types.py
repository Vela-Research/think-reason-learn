"""Types for the vanilla (direct-prompting) founder classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, TypeAlias

from pydantic import BaseModel, Field


ScoringFormat: TypeAlias = Literal["binary", "ternary", "5point"]
"""Supported scoring formats.

- ``"binary"``  -- 0/1 yes-no.
- ``"ternary"`` -- 0/1/2 (no / unsure / yes).
- ``"5point"``  -- 1-5 Likert-style rating.
"""


class FounderScore(BaseModel):
    """Structured score returned by the LLM for a single founder profile.

    This mirrors the JSON schema used in the research report (Appendix B of
    *The Configuration Gap*), minus the ``request_id`` echo field, which is
    unnecessary here because structured-output parsing handles response/sample
    alignment internally.
    """

    reasoning: str = Field(
        description="Concise, profile-grounded justification for the score."
    )
    score: int = Field(
        description="The rating on the configured scale (e.g. 1-5 for 5point)."
    )
    confidence: float = Field(
        default=0.0,
        description="The model's self-reported confidence in [0, 1].",
    )
    key_signals: List[str] = Field(
        default_factory=list,
        description="Short list of the profile signals that drove the score.",
    )


@dataclass(frozen=True, slots=True)
class ScaleSpec:
    """Definition of a scoring scale.

    Attributes:
        scoring_format: The scale identifier.
        range_text: Human-readable range, injected into the system prompt
            (e.g. ``"1 to 5"``).
        interpretation: How scale values map to a yes/no judgement, injected
            into the system prompt.
        values: The valid integer scores for this scale.
        default_threshold: The recommended decision threshold; a founder is
            predicted positive when ``score >= default_threshold``.
    """

    scoring_format: ScoringFormat
    range_text: str
    interpretation: str
    values: tuple[int, ...]
    default_threshold: int


@dataclass(slots=True)
class VanillaMetrics:
    """Classification metrics for a vanilla run.

    ``f_beta`` uses ``beta = 0.5`` (precision-weighted), matching VCBench's
    primary metric.
    """

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f_beta: float = 0.0
    threshold: int = 0
    n: int = 0
    n_predicted_positive: int = 0

    def to_dict(self) -> dict:
        """Return the metrics as a plain dictionary."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f_beta": self.f_beta,
            "threshold": self.threshold,
            "n": self.n,
            "n_predicted_positive": self.n_predicted_positive,
        }


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    *,
    threshold: int = 0,
    beta: float = 0.5,
) -> VanillaMetrics:
    """Compute precision / recall / F-beta from 0/1 labels and predictions.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        threshold: The decision threshold used to produce ``y_pred`` (stored
            on the result for reporting; does not affect the computation).
        beta: Beta for the F-beta score. Defaults to ``0.5``.

    Returns:
        The populated :class:`VanillaMetrics`.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    f_beta = (1 + beta_sq) * precision * recall / denom if denom else 0.0

    return VanillaMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f_beta=f_beta,
        threshold=threshold,
        n=len(y_true),
        n_predicted_positive=sum(y_pred),
    )


# Re-export for callers that want to construct an empty list default cleanly.
__all__ = [
    "ScoringFormat",
    "FounderScore",
    "ScaleSpec",
    "VanillaMetrics",
    "compute_metrics",
]
