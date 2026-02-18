"""Prompt construction for lambda-based feature generation."""

from __future__ import annotations

import json
import random
from typing import Any, Sequence

from ._types import DataSchema, HelperFunction, Rule


def build_system_prompt(
    schema: DataSchema,
    helpers: Sequence[HelperFunction],
    n_rules: int,
) -> str:
    """Build the system prompt from a data schema and helper functions.

    The prompt has five sections:

    1. Task description (generate *n_rules* binary rules as lambdas)
    2. Data schema (from ``schema.schema_text``)
    3. Available helper functions (signature + docstring for each)
    4. Lambda writing guidelines
    5. Example rules (from ``schema.example_rules``)
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

    sections = [
        s
        for s in [
            task,
            data_section,
            helpers_section,
            guidelines,
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
