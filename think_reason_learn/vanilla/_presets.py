"""Recommended vanilla configurations from the research report.

These presets encode the "Protocol Cookbook" (Section 6.3 of *The Configuration
Gap: Scoring Format, Temperature, and Architecture in LLM-Based Founder
Screening*). Each preset is the empirically best way to run a given Gemini tier
as a vanilla (direct-prompting) founder screener.

Cookbook summary the presets implement:

- **Format** -- 5-point for flash models (significant advantage over binary
  after multiplicity correction). For the thinking model, the format is
  operational: ternary@2 for precision, 5-point@4 for coverage.
- **Threshold** -- ``>= 4`` for the 5-point format; never ``>= 5``. ``>= 2``
  for ternary.
- **Decoding** -- ``temperature = 0`` by default (no accuracy difference
  survives correction; reproducible). ``temperature = 1`` is a higher-variance,
  higher-ceiling regime that flags modestly more founders.
- **Model** -- use at least a mid-tier model (flash or above); the budget model
  does not exceed the structured-feature reference baselines.

The model ids default to the Gemini-3 family used in the paper, but any Gemini
model string is accepted (the underlying choice type allows arbitrary strings).
Override per call with ``VanillaClassifier.from_preset(preset, model=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ._types import ScoringFormat


@dataclass(frozen=True, slots=True)
class VanillaPreset:
    """A named, recommended vanilla configuration.

    Attributes:
        name: Short identifier.
        description: Human-readable summary, including the cookbook rationale.
        model: Default Gemini model id for this tier.
        scoring_format: The recommended scoring format.
        threshold: The recommended decision threshold (``score >= threshold``
            predicts success).
        temperature: The recommended decoding temperature.
    """

    name: str
    description: str
    model: str
    scoring_format: ScoringFormat
    threshold: int
    temperature: float

    def with_overrides(self, **changes: object) -> "VanillaPreset":
        """Return a copy of the preset with the given fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]


FLASH_5POINT = VanillaPreset(
    name="flash_5point",
    description=(
        "Mid-tier flash model, 5-point @ >=4, T=0. The headline cookbook "
        "recommendation: 5-point gives flash models room to express graduated "
        "uncertainty and significantly beats binary (up to 3.3x more true "
        "positives) after Holm-Bonferroni correction. Best default for most "
        "users."
    ),
    model="gemini-3-flash-preview",
    scoring_format="5point",
    threshold=4,
    temperature=0.0,
)

PRO_PRECISION = VanillaPreset(
    name="pro_precision",
    description=(
        "Thinking model tuned for precision: ternary @ >=2, T=0. The thinking "
        "model is format-agnostic on F0.5, so pick the format that matches "
        "operational needs; ternary@2 is the precision-leaning operating point."
    ),
    model="gemini-3.1-pro-preview",
    scoring_format="ternary",
    threshold=2,
    temperature=0.0,
)

PRO_COVERAGE = VanillaPreset(
    name="pro_coverage",
    description=(
        "Thinking model tuned for coverage: 5-point @ >=4, T=0. Same F0.5 as "
        "ternary@2 for this model but surfaces more candidate founders."
    ),
    model="gemini-3.1-pro-preview",
    scoring_format="5point",
    threshold=4,
    temperature=0.0,
)

FLASH_LITE_DETERMINISTIC = VanillaPreset(
    name="flash_lite_deterministic",
    description=(
        "Budget model for exact determinism at T=0 (5-point @ >=4). Cheapest "
        "and most reproducible, but note the cookbook caveat: the budget model "
        "does NOT clearly exceed the structured-feature baselines -- prefer a "
        "flash-or-above tier when accuracy matters."
    ),
    model="gemini-3.1-flash-lite-preview",
    scoring_format="5point",
    threshold=4,
    temperature=0.0,
)

PROMPT_PRESETS: dict[str, VanillaPreset] = {
    p.name: p
    for p in (FLASH_5POINT, PRO_PRECISION, PRO_COVERAGE, FLASH_LITE_DETERMINISTIC)
}
"""Registry of all named vanilla presets."""

DEFAULT_PRESET = FLASH_5POINT
"""The recommended default configuration for a vanilla founder screener."""


__all__ = [
    "VanillaPreset",
    "FLASH_5POINT",
    "PRO_PRECISION",
    "PRO_COVERAGE",
    "FLASH_LITE_DETERMINISTIC",
    "PROMPT_PRESETS",
    "DEFAULT_PRESET",
]
