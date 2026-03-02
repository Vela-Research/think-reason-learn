"""Prompt construction for lambda-based feature generation."""

from __future__ import annotations

import json
import random
from typing import Any, Sequence

from ._types import CognitiveMode, DataSchema, HelperFunction, Rule

# ---------------------------------------------------------------------------
# Cognitive reasoning prompt sections (CoFEE-inspired)
# ---------------------------------------------------------------------------

_COGNITIVE_SECTIONS: dict[CognitiveMode, str] = {
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


def build_cognitive_section(
    modes: set[CognitiveMode],
) -> str:
    """Build the cognitive reasoning section of the system prompt.

    Returns the concatenated prompt text for all enabled cognitive modes,
    or an empty string if *modes* is empty.
    """
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
    """Build the system prompt from a data schema and helper functions.

    The prompt has up to seven sections:

    1. Task description (generate *n_rules* binary rules as lambdas)
    2. Data schema (from ``schema.schema_text``)
    3. Available helper functions (signature + docstring for each)
    4. Lambda writing guidelines
    5. **Cognitive reasoning constraints** (optional, from *cognitive_modes*)
    6. Example rules (from ``schema.example_rules``)
    7. Final instructions
    """
    param = schema.param_name

    # -- Section 1: task ----------------------------------------------------
    task = (
        f"You are an expert feature engineer.  Generate exactly {n_rules} "
        f"binary classification rules.\n\n"
        f"Each rule MUST be a valid Python lambda expression that takes a "
        f"`{param}` dict and returns True or False."
    )

    # -- Section 2: data schema --------------------------------------------
    data_section = (
        f"## DATA STRUCTURE\n\n"
        f"{schema.description}\n\n"
        f"```python\n{schema.schema_text}\n```"
    )

    # -- Section 3: helpers -------------------------------------------------
    if helpers:
        helper_lines = ["## HELPER FUNCTIONS AVAILABLE\n"]
        helper_lines.append(
            "You can use these helper functions in your lambda expressions:\n"
        )
        helper_lines.append("```python")
        for h in helpers:
            helper_lines.append(f"{h.signature}  # {h.docstring}")
        helper_lines.append("```")
        helpers_section = "\n".join(helper_lines)
    else:
        helpers_section = ""

    # -- Section 4: guidelines ----------------------------------------------
    guidelines = (
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

    # -- Section 5: examples ------------------------------------------------
    if schema.example_rules:
        example_lines = ["## EXAMPLE RULES\n"]
        for i, rule in enumerate(schema.example_rules, 1):
            example_lines.append(
                f"{i}. {rule.name}: {rule.description}\n   `{rule.expression}`"
            )
        examples_section = "\n".join(example_lines)
    else:
        examples_section = ""

    # -- Section 6: final instruction ---------------------------------------
    final = (
        "## IMPORTANT\n\n"
        f"- Every lambda MUST be syntactically valid Python\n"
        f"- Every lambda MUST handle edge cases (empty lists, missing keys)\n"
        f"- Use only the helper functions and standard Python\n"
        f"- The lambda parameter name must be `{param}`"
    )

    # -- Section 5: cognitive reasoning (optional) ----------------------------
    cognitive_section = build_cognitive_section(cognitive_modes or set())

    sections = [
        s
        for s in [
            task,
            data_section,
            helpers_section,
            guidelines,
            cognitive_section,
            examples_section,
            final,
        ]
        if s
    ]
    return "\n\n".join(sections)


def build_user_prompt(
    samples_text: str,
    n_rules: int,
    prior_rules: Sequence[Rule] | None = None,
) -> str:
    """Build the user prompt with sample data and optional feedback.

    Parameters
    ----------
    samples_text:
        Formatted sample records (from :func:`format_samples`).
    n_rules:
        Number of rules to request.
    prior_rules:
        Optional list of rules from a previous iteration, injected as
        feedback to guide the LLM toward improved or more diverse rules.
    """
    parts: list[str] = []

    if prior_rules:
        parts.append("## FEEDBACK FROM PREVIOUS RULES\n")
        parts.append(
            "Here are rules from a previous iteration.  Use this information "
            "to generate improved, more expressive, or more diverse new rules "
            "that avoid redundancy.\n"
        )
        parts.append("Existing rules:\n")
        for rule in prior_rules:
            parts.append(f"- {rule.name}: {rule.description}\n  `{rule.expression}`")
        parts.append("\n---\n")

    parts.append(f"Data samples to analyse:\n\n{samples_text}")
    parts.append(
        f"\nGenerate exactly {n_rules} binary rules as Python lambda expressions."
    )

    return "\n".join(parts)


def format_samples(
    records: Sequence[dict[str, Any]],
    labels: Sequence[Any],
    n_samples: int = 60,
    *,
    success_value: Any = 1,
    random_state: int | None = 42,
) -> str:
    """Format a diverse subset of records for the LLM prompt.

    Records are stratified by label (roughly 50/50 success/fail), shuffled,
    and serialised to a compact text representation.

    Parameters
    ----------
    records:
        The full list of structured data dicts.
    labels:
        Parallel label sequence (one per record).
    n_samples:
        How many records to include in the prompt.
    success_value:
        The value in *labels* that indicates a positive/successful record.
    random_state:
        Seed for reproducible sampling.  ``None`` for non-deterministic.
    """
    rng = random.Random(random_state)

    positives = [(r, lbl) for r, lbl in zip(records, labels) if lbl == success_value]
    negatives = [(r, lbl) for r, lbl in zip(records, labels) if lbl != success_value]

    n_pos = min(n_samples // 2, len(positives))
    n_neg = min(n_samples - n_pos, len(negatives))

    selected = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
    rng.shuffle(selected)

    lines: list[str] = []
    for record, label in selected:
        tag = "[SUCCESS]" if label == success_value else "[FAIL]"
        compact = json.dumps(record, default=str, ensure_ascii=False)
        lines.append(f"{tag}\n{compact}\n---")

    return "\n".join(lines)
