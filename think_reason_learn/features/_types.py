"""Data models for lambda-based feature generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Cognitive reasoning modes (CoFEE-inspired)
# ---------------------------------------------------------------------------


class CognitiveMode(StrEnum):
    """Cognitive reasoning behaviours that can be toggled in the generation prompt.

    Based on the CoFEE framework (Westermann, 2025), which enforces structured
    cognitive behaviours during LLM-based feature discovery.  Each mode injects
    a corresponding section into the system prompt.
    """

    BACKWARD_CHAINING = "backward_chaining"
    SUBGOAL_DECOMPOSITION = "subgoal_decomposition"
    VERIFICATION = "verification"
    BACKTRACKING = "backtracking"


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class GeneratedRule(BaseModel):
    """A single binary classification rule returned by the LLM."""

    name: str = Field(
        ...,
        description=(
            "Snake_case identifier for the rule, e.g. 'has_phd' or 'prior_exit'."
        ),
    )
    description: str = Field(
        ...,
        description="Short description of what this rule checks.",
    )
    expression: str = Field(
        ...,
        description=(
            "A Python lambda expression that takes one dict argument and "
            "returns True or False."
        ),
    )


class GeneratedRules(BaseModel):
    """Structured output containing all generated rules."""

    rules: List[GeneratedRule] = Field(
        ...,
        description="List of generated binary classification rules.",
    )


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A compiled-ready rule with name, description, and expression string."""

    name: str
    description: str
    expression: str  # e.g. "lambda founder: len(founder.get('ipos', [])) > 0"


@dataclass(frozen=True)
class HelperFunction:
    """A helper function available to lambda expressions.

    The function must exist in *both* worlds:
    - In the LLM prompt (so the model knows it can use it)
    - In the eval context (so the lambda can call it at runtime)
    """

    name: str  # e.g. "parse_qs"
    func: Callable[..., Any]  # the actual Python callable
    signature: str  # e.g. "parse_qs(qs_str: str) -> float"
    docstring: str  # e.g. 'Converts "1" -> 1.0, "200+" -> 250.0, "" -> 999.0'


@dataclass(frozen=True)
class DataSchema:
    """Description of the input data structure for the LLM prompt.

    Attributes:
        description: Plain-English description of the record type.
        schema_text: A Python-like schema definition inserted verbatim into the
            system prompt (e.g. the dict structure with types).
        param_name: The lambda parameter name (e.g. ``"founder"``).
        example_rules: A few hand-written :class:`Rule` instances that
            demonstrate the expected lambda style.
    """

    description: str
    schema_text: str
    param_name: str  # "founder"
    example_rules: list[Rule]
