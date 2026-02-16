"""Configuration for cost-sensitive RRF training."""

from dataclasses import dataclass


@dataclass
class CostSensitiveConfig:
    """Configuration for cost-sensitive training.

    Cost-sensitive mode reduces LLM API costs through a multi-stage pipeline:
    1. Auto semantic filtering (optional)
    2. Screening evaluation on small subset
    3. Pruning low-performing questions
    4. Top-N selection
    5. Full evaluation on complete training set

    For large datasets (9000+ samples), this can reduce costs by 44-74%.
    """

    screening_fraction: float = 0.05
    """Fraction of training data to use for screening (default: 5%)."""

    max_screening_samples: int | None = 500
    """Maximum number of samples for screening, regardless of fraction."""

    screening_metric: str = "f1"
    """Metric to use for screening evaluation ("f1", "precision", or "recall")."""

    screening_baseline: float | str = "majority"
    """Baseline for pruning questions.

    - "majority": F1 score of majority-class classifier
    - "random": Expected F1 of random classifier
    - float: Explicit threshold (e.g., 0.6)

    Questions scoring at or below baseline are excluded.
    """

    max_questions_full_eval: int = 20
    """Maximum number of top questions to evaluate on full training set.

    After screening, only the top N questions by screening_metric are
    fully evaluated. Lower values = more aggressive cost reduction.
    """

    enable_semantic_filter: bool = True
    """Whether to auto-apply semantic filtering before screening."""

    semantic_threshold: float = 0.85
    """Similarity threshold for semantic filtering (0-1)."""

    semantic_emb_model: str = "hashed_bag_of_words"
    """Embedding model for semantic filtering.

    Default "hashed_bag_of_words" is deterministic and offline.
    """
