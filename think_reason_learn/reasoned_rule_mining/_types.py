"""Pydantic models and configuration for Reasoned Rule Mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from pydantic import BaseModel


class Vote(BaseModel):
    """A single YES/NO vote from the prediction LLM."""

    vote: Literal["YES", "NO"]


class ExtractedRule(BaseModel):
    """One extracted IF-THEN rule with its forced outcome and perplexity.

    Attributes:
        rule: The natural-language ``IF ... THEN label = YES/NO`` rule text.
        outcome: The label the rule is aligned to (the sample's true label).
        perplexity: Model perplexity of the rule text (lower = more confident).
    """

    rule: str
    outcome: Literal["YES", "NO"]
    perplexity: float


@dataclass
class RRMConfig:
    """Configuration for Reasoned Rule Mining.

    Defaults mirror the standalone VCBench reference implementation.

    Args:
        perplexity_threshold: Keep only rules with perplexity <= this value when
            compiling the decision policy.
        ensemble_size: Number of votes per sample in the ensemble (and harsh)
            stage.
        precision_voting_threshold: Fraction of votes that must be YES for the
            ensemble to keep a YES prediction.
        ensemble_confidence_threshold: A tentative YES is downgraded to NO if the
            ensemble probability is below this value.
        final_confidence_threshold: Extra confidence floor applied only when
            ``enable_final_confidence_check`` is True.
        enable_final_confidence_check: Whether to apply ``final_confidence_threshold``.
        use_harsh: Whether to run the two-stream harsh re-evaluation stage.
        harsh_level: Severity of harsh re-evaluation ("light"/"moderate"/"strict").
        beta: Beta for the F-beta objective optimised by the combiner (0.5 = F0.5).
        recall_floors: Recall floors swept during weight optimisation; the floor
            with the best F-beta is selected.
        weights_alpha_range: (min, max) grid range for the ensemble-stream weight.
        weights_beta_range: (min, max) grid range for the harsh-stream weight.
        weights_bias_range: (min, max) grid range for the combiner bias.
        weights_n_points: Grid resolution per weight axis.
        weights_thresh_points: Number of decision thresholds swept per weight tuple.
        use_memory: Whether to maintain a rolling reasoning-memory summary.
        memory_update_interval: Number of samples between memory-summary updates.
        random_state: Random seed for Platt-scaling logistic regressions.
    """

    perplexity_threshold: float = 1.6
    ensemble_size: int = 3
    precision_voting_threshold: float = 0.5
    ensemble_confidence_threshold: float = 0.80
    final_confidence_threshold: float = 0.60
    enable_final_confidence_check: bool = False
    use_harsh: bool = True
    harsh_level: Literal["light", "moderate", "strict"] = "strict"
    beta: float = 0.5
    recall_floors: Tuple[float, ...] = (0.05, 0.10, 0.15)
    weights_alpha_range: Tuple[float, float] = (0.0, 5.0)
    weights_beta_range: Tuple[float, float] = (0.0, 5.0)
    weights_bias_range: Tuple[float, float] = (-10.0, 10.0)
    weights_n_points: int = 11
    weights_thresh_points: int = 100
    use_memory: bool = True
    memory_update_interval: int = 50
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any value is out of its allowed range.
        """
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        if not (0.0 < self.precision_voting_threshold <= 1.0):
            raise ValueError("precision_voting_threshold must be in (0, 1]")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.harsh_level not in ("light", "moderate", "strict"):
            raise ValueError("harsh_level must be 'light', 'moderate', or 'strict'")
        if self.weights_n_points < 2:
            raise ValueError("weights_n_points must be >= 2")
        if self.weights_thresh_points < 2:
            raise ValueError("weights_thresh_points must be >= 2")
        if len(self.recall_floors) == 0:
            raise ValueError("recall_floors must be non-empty")
        if any(not (0.0 < f <= 1.0) for f in self.recall_floors):
            raise ValueError("each recall floor must be in (0, 1]")
        if self.memory_update_interval < 1:
            raise ValueError("memory_update_interval must be >= 1")
