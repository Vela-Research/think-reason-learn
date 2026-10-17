"""Data models for lambda-based feature generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

type Record = Mapping[str, object]
"""One structured sample as a dict, e.g. a customer's or a patient's record."""


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


class GeneratedRule(BaseModel):
    """A single binary classification rule returned by the LLM."""

    name: str = Field(
        ...,
        description=(
            "Snake_case identifier for the rule, e.g. 'has_email' or 'over_limit'."
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

    rules: list[GeneratedRule] = Field(
        ...,
        description="List of generated binary classification rules.",
    )


@dataclass(frozen=True, slots=True)
class Rule:
    """A compiled-ready rule with name, description, and expression string."""

    name: str
    description: str
    expression: str  # e.g. "lambda record: len(record.get('orders', [])) > 3"


@dataclass(frozen=True, slots=True)
class HelperFunction:
    """A helper function the rules may call.

    It is described in the prompt, so the model knows it, and placed in the
    rules' scope, so they can call it.

    Attributes:
        name: The name rules call it by, e.g. ``"parse_qs"``.
        func: The function.
        signature: Shown in the prompt, e.g. ``"parse_qs(qs_str: str) -> float"``.
        docstring: What it does, shown in the prompt.
    """

    name: str
    func: Callable[..., object]
    signature: str
    docstring: str


@dataclass(frozen=True, slots=True)
class DataSchema:
    """Description of the input data structure for the LLM prompt.

    Attributes:
        description: Plain-English description of the record type.
        schema_text: A Python-like schema definition inserted verbatim into the
            system prompt (e.g. the dict structure with types).
        param_name: The lambda parameter name (e.g. ``"record"``).
        example_rules: A few hand-written :class:`Rule` instances that
            demonstrate the expected lambda style.
    """

    description: str
    schema_text: str
    param_name: str
    example_rules: Sequence[Rule]

    def __post_init__(self) -> None:
        """Keep the example rules as a tuple, so the schema cannot change."""
        object.__setattr__(self, "example_rules", tuple(self.example_rules))
