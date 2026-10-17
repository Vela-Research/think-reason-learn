"""Reasoned Rule Mining's prompts: mining, compiling, voting and the harsh review."""

from __future__ import annotations

from typing import Final, Literal

type HarshLevel = Literal["light", "moderate", "strict"]

MINING_INSTRUCTIONS: Final = """\
You are a careful analyst. You are given a classification task, one sample and
its true class. First explain step by step why the sample has that class,
grounded in its concrete attributes and favouring signals that generalise. Then
turn your reasoning into one rule: IF <conditions> THEN class = <the class>."""

COMPILE_INSTRUCTIONS: Final = """\
You compile a decision policy for a classification task from rules mined from
labelled samples. Merge the rules for each class into one clear, readable policy
that a reader could apply to a new sample. Keep the conditions that recur, drop
those tied to single samples, and reply with the policy only."""

HARSH: Final[dict[HarshLevel, str]] = {
    "light": "Re-examine the sample with a balanced view of strengths and weaknesses.",
    "moderate": "Re-examine the sample carefully; when in doubt, lean towards no.",
    "strict": "Re-examine the sample with maximum scrutiny: say yes only if it is "
    "exceptional, with no significant weakness.",
}


def mining_prompt(task: str, sample: str, label: str) -> str:
    """The prompt that reasons about one labelled sample and states a rule."""
    return f"Task: {task}\n\nSample:\n{sample}\n\nTrue class: {label}"


def compile_prompt(task: str, rules_by_class: dict[str, list[str]]) -> str:
    """The prompt that merges the mined rules into one policy."""
    blocks = "\n\n".join(
        f"Rules for class {label}:\n" + "\n".join(f"- {rule}" for rule in rules)
        for label, rules in rules_by_class.items()
    )
    return f"Task: {task}\n\n{blocks}"


def vote_question(task: str, policy: str) -> str:
    """The cheap first-pass question: the task's answer, with the policy as rubric."""
    return (
        f"Task: {task}\n\nDecision policy:\n{policy}\n\n"
        "Using the policy as the rubric, is the answer to the task YES?"
    )


def harsh_question(task: str, policy: str, level: HarshLevel) -> str:
    """The second-pass question for samples the first pass rated highly."""
    return f"{HARSH[level]}\n\n{vote_question(task, policy)}"
