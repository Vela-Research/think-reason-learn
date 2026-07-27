"""Prompts and scale definitions for the vanilla founder classifier.

All scoring formats share the same system prompt; only the scoring-rule lines
(scale range and interpretation) vary by format. This is the exact prompt used
in the research report (*The Configuration Gap*, Appendix B), reproduced so the
library default reproduces the paper's vanilla baseline.
"""

from __future__ import annotations

from ._types import ScaleSpec, ScoringFormat


SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert in venture capital tasked with identifying successful "
    "founders from their unsuccessful counterparts.\n"
    "All founders under consideration are sourced from LinkedIn and Crunchbase "
    "profiles of companies that have raised between $100K and $4M in funding.\n"
    "A successful founder is defined as one whose company has achieved either a "
    "total funding of over $500M or an exit/IPO valued at over $500M.\n\n"
    "Important: Do not cheat, ground your reasoning to the anonymized founder "
    "data only without trying to guess who this person is or using online "
    "lookup.\n"
    "SCORING RULE:\n"
    "You must use a scale of {range_text}.\n"
    "Interpretation: {interpretation}\n\n"
    "Provide concise, profile-grounded reasoning, an integer score on the "
    "scale above, a confidence in [0, 1], and a short list of the key signals "
    "that drove your score."
)
"""System prompt with ``{range_text}`` / ``{interpretation}`` placeholders."""


USER_TEMPLATE = (
    "Analyze this founder profile and provide your rating based on the "
    "specific scale rules:\n\n{profile}"
)
"""User message template with a ``{profile}`` placeholder."""


SCALES: dict[ScoringFormat, ScaleSpec] = {
    "binary": ScaleSpec(
        scoring_format="binary",
        range_text="0 or 1",
        interpretation="0: No, 1: Yes",
        values=(0, 1),
        default_threshold=1,
    ),
    "ternary": ScaleSpec(
        scoring_format="ternary",
        range_text="0, 1, or 2",
        interpretation="0: No, 1: Unsure/Needs Analysis, 2: Yes",
        values=(0, 1, 2),
        default_threshold=2,
    ),
    "5point": ScaleSpec(
        scoring_format="5point",
        range_text="1 to 5",
        interpretation="1-2: No, 3: Neutral/Unsure, 4-5: Yes",
        values=(1, 2, 3, 4, 5),
        default_threshold=4,
    ),
}
"""Registry of supported scoring formats."""


def build_system_prompt(scoring_format: ScoringFormat) -> str:
    """Render the system prompt for a given scoring format.

    Args:
        scoring_format: One of ``"binary"``, ``"ternary"``, ``"5point"``.

    Returns:
        The fully rendered system prompt.

    Raises:
        ValueError: If ``scoring_format`` is not a known scale.
    """
    spec = SCALES.get(scoring_format)
    if spec is None:
        raise ValueError(
            f"Unknown scoring_format {scoring_format!r}. Supported: {list(SCALES)}."
        )
    return SYSTEM_PROMPT_TEMPLATE.format(
        range_text=spec.range_text,
        interpretation=spec.interpretation,
    )


__all__ = [
    "SYSTEM_PROMPT_TEMPLATE",
    "USER_TEMPLATE",
    "SCALES",
    "build_system_prompt",
]
