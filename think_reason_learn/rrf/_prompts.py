"""RRF's prompts, after the paper's question-generation prompt."""

from __future__ import annotations

from typing import Final

INSTRUCTIONS_PURPOSE: Final = (
    "YES/NO questions about a sample whose answers, weighed together, predict "
    "its class: each question objective, specific and answerable from the "
    "sample's text alone"
)

QUESTION_INSTRUCTIONS: Final = """\
You write YES/NO questions for a classification task. Another model will answer
each question from one sample's text alone.

Each question must be objective and specific, about an observable trait in a
sample, and must never restate or ask for the label. Write questions that differ
from each other."""


def question_prompt(task: str, instructions: str, count: int, examples: str) -> str:
    """The prompt for one batch of questions."""
    return (
        f"Task: {task}\n\nInstructions:\n{instructions}\n\n"
        f"Write {count} YES/NO questions that can be answered from the samples "
        "below and that may tell their classes apart.\n\n"
        f"Samples:\n\n{examples}"
    )
