"""Policy Induction's prompts, after the paper's."""

from __future__ import annotations

from typing import Final

INSTRUCTIONS_PURPOSE: Final = (
    "a policy: a short list of general rules, each one sentence, that predict "
    "the class of a sample from its text"
)

POLICY_INSTRUCTIONS: Final = """\
You write a policy for a classification task: a list of at most {max_rules}
rules, each one clear sentence, that together predict whether a sample belongs
to the positive class ({positive}). Rules must generalise: describe patterns,
not individual samples, and never restate the label."""

UPDATE_INSTRUCTIONS: Final = """\
You refine a policy for a classification task. You are given the current policy
and new samples with their true classes. Refine and expand the existing policy
so it would classify these samples correctly: keep the rules that still hold,
sharpen vague ones, and add rules for patterns the policy misses. Keep at most
{max_rules} rules, each one clear sentence, general rather than tied to one
sample, and never restating the label."""


def initial_prompt(task: str, instructions: str, examples: str) -> str:
    """The prompt that writes the first policy from labelled examples."""
    return (
        f"Task: {task}\n\nInstructions:\n{instructions}\n\n"
        f"Labelled samples:\n\n{examples}"
    )


def update_prompt(task: str, instructions: str, policy: str, cases: str) -> str:
    """The prompt that refines ``policy`` with new labelled cases."""
    return (
        f"Task: {task}\n\nInstructions:\n{instructions}\n\n"
        f"Current policy:\n{policy}\n\nNew samples:\n\n{cases}"
    )


def policy_question(policy: str, positive: str) -> str:
    """The question the answerer judges a sample with, under the whole policy."""
    return f"Policy:\n{policy}\n\nFollowing this policy, is this sample {positive}?"


def rule_question(rule: str) -> str:
    """The question for one rule, when rules are weighed separately."""
    return f"Does this rule hold for the sample? Rule: {rule}"
