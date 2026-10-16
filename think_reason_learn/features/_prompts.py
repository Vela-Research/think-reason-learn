"""Prompt construction for lambda-based feature generation."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Final

from ._types import CognitiveMode, DataSchema, HelperFunction, Record, Rule

_COGNITIVE_SECTIONS: Final[dict[CognitiveMode, str]] = {
    CognitiveMode.BACKWARD_CHAINING: (
        "## BACKWARD CHAINING\n\n"
        "Reason backward from the target outcome to observable signals.\n\n"
        "For each rule you propose:\n"
        "- State the causal hypothesis explicitly\n"
        "- Explain why this signal would exist *before* the outcome\n"
        "- Map the hypothesis to a measurable quantity in the data\n\n"
        "The final lambda MUST capture something observable pre-outcome "
        "and expressible as a Python lambda.\n"
        "If a hypothesis cannot be mapped to an observable feature, "
        "abandon it."
    ),
    CognitiveMode.SUBGOAL_DECOMPOSITION: (
        "## SUBGOAL DECOMPOSITION\n\n"
        "Organise your feature exploration into NO MORE THAN 4 high-level "
        "subgoals, such as:\n"
        "- Capability formation\n"
        "- Team coordination and complementarity\n"
        "- Market structure and constraints\n"
        "- Early execution dynamics\n\n"
        "For each subgoal:\n"
        "- List candidate mechanisms\n"
        "- Maintain the hierarchy: system behaviour -> mechanism -> "
        "lambda rule\n\n"
        "If a subgoal collapses into a proxy, fails observability, or has "
        "ambiguous causal direction, explicitly ABANDON or REVISE it and "
        "explain why."
    ),
    CognitiveMode.VERIFICATION: (
        "## VERIFICATION\n\n"
        "For each rule you propose, verify:\n"
        "- It captures a signal observable before the outcome\n"
        "- It encodes a plausible causal mechanism\n"
        "- It is not a prestige-based, descriptive, or post-outcome proxy\n\n"
        "For each rule, list:\n"
        "- Potential bias sources\n"
        "- Uncertainty or ambiguity\n\n"
        "If verification fails, reject the rule and propose a replacement."
    ),
    CognitiveMode.BACKTRACKING: (
        "## BACKTRACKING\n\n"
        "Explicitly record every abandoned reasoning path.\n\n"
        "For each abandoned path, record:\n"
        "- Why it initially seemed promising\n"
        "- Which constraint caused rejection (proxy risk, leakage, "
        "observability failure, causal ambiguity)\n\n"
        "Use these abandoned paths to avoid similar dead ends in "
        "subsequent rules."
    ),
}


def build_cognitive_section(modes: set[CognitiveMode]) -> str:
    """The cognitive reasoning section for ``modes``; empty for none."""
    if not modes:
        return ""

    parts = [
        "## COGNITIVE REASONING\n\n"
        "You are performing Cognitive Feature Reasoning.  You must "
        "explicitly apply the following cognitive behaviours.  You must "
        "produce structured outputs and make explicit decisions.\n"
    ]
    # Maintain stable ordering by iterating over the enum definition order
    for mode in CognitiveMode:
        if mode in modes:
            parts.append(_COGNITIVE_SECTIONS[mode])

    return "\n\n".join(parts)


def build_system_prompt(
    schema: DataSchema,
    helpers: Sequence[HelperFunction],
    n_rules: int,
    cognitive_modes: set[CognitiveMode] | None = None,
) -> str:
    """The system prompt for writing ``n_rules`` rules.

    In order: the task, the data, helpers, guidelines, reasoning constraints
    (if any), example rules and final instructions.
    """
    param = schema.param_name
    sections = [
        f"You are an expert feature engineer.  Generate exactly {n_rules} "
        f"binary classification rules.\n\n"
        f"Each rule MUST be a valid Python lambda expression that takes a "
        f"`{param}` dict and returns True or False.",
        f"## DATA STRUCTURE\n\n{schema.description}\n\n"
        f"```python\n{schema.schema_text}\n```",
        _helpers_section(helpers),
        _GUIDELINES,
        build_cognitive_section(cognitive_modes or set()),
        _examples_section(schema.example_rules),
        "## IMPORTANT\n\n"
        "- Every lambda MUST be syntactically valid Python\n"
        "- Every lambda MUST handle edge cases (empty lists, missing keys)\n"
        "- Use only the helper functions and standard Python\n"
        f"- The lambda parameter name must be `{param}`",
    ]
    return "\n\n".join(section for section in sections if section)


def build_user_prompt(
    samples_text: str,
    n_rules: int,
    prior_rules: Sequence[Rule] | None = None,
) -> str:
    """The user prompt: earlier rules as feedback (if any), then the samples.

    Args:
        samples_text: Formatted records, from ``format_samples``.
        n_rules: Rules to ask for.
        prior_rules: Rules from an earlier round, to improve on or diversify.
    """
    parts: list[str] = []
    if prior_rules:
        parts += [
            "## FEEDBACK FROM PREVIOUS RULES\n",
            "Here are rules from a previous iteration.  Use this information "
            "to generate improved, more expressive, or more diverse new rules "
            "that avoid redundancy.\n",
            "Existing rules:\n",
            *(
                f"- {rule.name}: {rule.description}\n  `{rule.expression}`"
                for rule in prior_rules
            ),
            "\n---\n",
        ]
    parts += [
        f"Data samples to analyse:\n\n{samples_text}",
        f"\nGenerate exactly {n_rules} binary rules as Python lambda expressions.",
    ]
    return "\n".join(parts)


def format_samples(
    records: Sequence[Record],
    labels: Sequence[object],
    n_samples: int = 60,
    *,
    positive: object = 1,
    random_state: int | None = 42,
) -> str:
    """A shuffled, roughly half-positive subset of records, as prompt text.

    Args:
        records: The structured records.
        labels: One label per record.
        n_samples: Records to include.
        positive: The label of the class the rules should find.
        random_state: Seed for the subset; ``None`` for a random one.
    """
    rng = random.Random(random_state)
    pairs = list(zip(records, labels, strict=True))
    positives = [pair for pair in pairs if pair[1] == positive]
    negatives = [pair for pair in pairs if pair[1] != positive]
    n_pos = min(n_samples // 2, len(positives))
    n_neg = min(n_samples - n_pos, len(negatives))
    selected = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
    rng.shuffle(selected)
    return "\n".join(
        f"{'[SUCCESS]' if label == positive else '[FAIL]'}\n"
        f"{json.dumps(record, default=str, ensure_ascii=False)}\n---"
        for record, label in selected
    )


_GUIDELINES: Final = (
    "## RULES FOR WRITING LAMBDA EXPRESSIONS\n\n"
    "1. Always handle empty lists: use `any()`, `len()`, or list "
    "comprehensions safely\n"
    "2. Always use `.get()` for dict access with defaults\n"
    "3. Use `.lower()` for string comparisons\n"
    "4. Return boolean (True/False) — it will be converted to 0/1\n"
    "5. Be diverse: cover different aspects of the data\n"
    "6. Generated rules should not be semantically very similar to "
    "each other"
)


def _helpers_section(helpers: Sequence[HelperFunction]) -> str:
    if not helpers:
        return ""
    return "\n".join(
        [
            "## HELPER FUNCTIONS AVAILABLE\n",
            "You can use these helper functions in your lambda expressions:\n",
            "```python",
            *(f"{h.signature}  # {h.docstring}" for h in helpers),
            "```",
        ]
    )


def _examples_section(rules: Sequence[Rule]) -> str:
    if not rules:
        return ""
    return "\n".join(
        [
            "## EXAMPLE RULES\n",
            *(
                f"{i}. {rule.name}: {rule.description}\n   `{rule.expression}`"
                for i, rule in enumerate(rules, 1)
            ),
        ]
    )
