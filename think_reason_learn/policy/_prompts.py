"""The teacher's and the writer's prompts, and the checks built from a policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from ._types import Criterion, Mistake, Outcome, Policy

TEACHER_INSTRUCTIONS: Final = """\
You apply a written policy to one case and decide its outcome. The outcomes are
ordered from lowest to highest. Judge the case by the policy alone, and from the
case's text alone: do not assume facts the text does not give.

Reply with:
- outcome: the name of one outcome, exactly as written;
- confidence: from 0 to 1, how sure you are;
- criteria: the ids of the criteria that decided the outcome;
- unjudged: the ids of the criteria the case's text gives no way to judge;
- why: one or two short sentences."""

WRITER_INSTRUCTIONS: Final = """\
You write yes/no checks for a decision model. A small, literal model answers
each check from one case's text alone, and the decision model weighs the
answers to put each case in one of the policy's ordered outcomes.

Write checks that would tell apart the cases the current checks get wrong. Each
check is one plain statement about the case, true or false: about something the
case's text states or clearly lacks, specific enough to answer without
judgement, and different from every current check. Never restate an outcome.
Mark a check "positive" if its being true points to a higher outcome, and
"negative" if it points to a lower one."""

_SHOWN_IN_FULL: Final = 25
"""Mistakes shown with their case's text; the rest are listed by their reasons."""
_LISTED: Final = 100
"""Most mistakes listed by their reasons alone."""


def policy_block(policy: Policy) -> str:
    """The policy, its criteria and its outcomes, as the teacher and writer see it."""
    parts: list[str] = []
    if policy.text.strip():
        parts.append(f"Policy:\n{policy.text.strip()}")
    if policy.criteria:
        listed = "\n".join(f"- {c.id} [{c.kind}]: {c.text}" for c in policy.criteria)
        parts.append(f"Criteria:\n{listed}")
    outcomes = "\n".join(
        f"{n}. {o.name}: {o.described}" for n, o in enumerate(policy.outcomes, 1)
    )
    parts.append(f"Outcomes, lowest first:\n{outcomes}")
    return "\n\n".join(parts)


def case_prompt(case: str) -> str:
    """The teacher's prompt for one case; the policy goes before it, cached."""
    return f"Case:\n{case}"


def writer_prompt(
    policy: Policy, current: Sequence[str], mistakes: Sequence[Mistake], count: int
) -> str:
    """The prompt asking for up to ``count`` new checks from ``mistakes``.

    The first mistakes are shown with their case's text, the rest by their
    outcomes and reasons only, so the prompt stays a manageable size.
    """
    checks = "\n".join(f"- {text}" for text in current) or "(none)"
    shown = "\n\n".join(
        f"Case {n}: outcome {m.outcome}, model said {m.predicted}."
        f"{_reason(m)}\n{m.case}"
        for n, m in enumerate(mistakes[:_SHOWN_IN_FULL], 1)
    )
    listed = "\n".join(
        f"- outcome {m.outcome}, model said {m.predicted}.{_reason(m)}"
        for m in mistakes[_SHOWN_IN_FULL : _SHOWN_IN_FULL + _LISTED]
    )
    more = (
        f"\n\nMore cases it gets wrong, without their text:\n{listed}" if listed else ""
    )
    return (
        f"{policy_block(policy)}\n\nCurrent checks:\n{checks}\n\n"
        f"Cases the model gets wrong ({len(mistakes)}):\n\n{shown}{more}\n\n"
        f"Write up to {count} new checks."
    )


def reach_check(text: str, outcome: Outcome) -> str:
    """Whether a case reaches ``outcome`` or a higher one, judged by ``text`` alone.

    Asked as "at least", a yes always points up, so its weight may only raise
    the outcome.
    """
    meaning = f": {_sentence(outcome)}" if outcome.meaning.strip() else ""
    return (
        f'Policy: "{text}"\nJudged by this policy alone, the case reaches at least '
        f"{outcome.name}{meaning}."
    )


def applies_check(criterion: Criterion) -> str:
    """Whether ``criterion`` bears on the case at all."""
    return f'This criterion clearly applies to the case: "{criterion.text}"'


def met_check(criterion: Criterion) -> str:
    """Whether the case meets ``criterion``."""
    return f'Criterion: "{criterion.text}"\nThis criterion is met by the case.'


def whole_policy(policy: Policy) -> str:
    """The policy's text and criteria in one, for checks judged by all of it."""
    parts = [policy.text.strip()] if policy.text.strip() else []
    for kind, heading in (
        ("positive", "In favour"),
        ("negative", "Against"),
        ("neutral", "Also weighed"),
    ):
        if listed := [c.text for c in policy.criteria if c.kind == kind]:
            parts.append(f"{heading}:\n" + "\n".join(f"- {t}" for t in listed))
    return "\n".join(parts)


def _sentence(outcome: Outcome) -> str:
    return outcome.described.rstrip(". ")


def _reason(mistake: Mistake) -> str:
    return f" Reason: {mistake.why.strip()}" if mistake.why.strip() else ""
