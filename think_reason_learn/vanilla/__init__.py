"""Vanilla founder classifier.

A direct-prompting (zero-shot) baseline for founder-success screening: each
profile is sent to an LLM once with a fixed scoring prompt and the returned
score is thresholded. This is the canonical baseline that RRF, GPTree and
policy induction are compared against.

The module ships with the empirically best configurations from the research
report (*The Configuration Gap*) as ready-to-use presets, so users get the best
way to run a vanilla model out of the box::

    from think_reason_learn.vanilla import VanillaClassifier, FLASH_5POINT

    clf = VanillaClassifier.from_preset(FLASH_5POINT)
    metrics = await clf.evaluate(X, y)
"""

from ._vanilla import VanillaClassifier
from ._types import (
    ScoringFormat,
    FounderScore,
    ScaleSpec,
    VanillaMetrics,
    compute_metrics,
)
from ._prompts import (
    SCALES,
    SYSTEM_PROMPT_TEMPLATE,
    USER_TEMPLATE,
    build_system_prompt,
)
from ._presets import (
    VanillaPreset,
    PROMPT_PRESETS,
    DEFAULT_PRESET,
    FLASH_5POINT,
    PRO_PRECISION,
    PRO_COVERAGE,
    FLASH_LITE_DETERMINISTIC,
)

__all__ = [
    "VanillaClassifier",
    "ScoringFormat",
    "FounderScore",
    "ScaleSpec",
    "VanillaMetrics",
    "compute_metrics",
    "SCALES",
    "SYSTEM_PROMPT_TEMPLATE",
    "USER_TEMPLATE",
    "build_system_prompt",
    "VanillaPreset",
    "PROMPT_PRESETS",
    "DEFAULT_PRESET",
    "FLASH_5POINT",
    "PRO_PRECISION",
    "PRO_COVERAGE",
    "FLASH_LITE_DETERMINISTIC",
]
