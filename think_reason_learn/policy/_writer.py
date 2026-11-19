"""New yes/no checks, written by a model from the cases the current ones get wrong."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel

from think_reason_learn.llm import (
    Endpoint,
    LLMSession,
    Models,
    Usage,
    resolve_models,
    run_async,
    run_sync,
)

from ._prompts import WRITER_INSTRUCTIONS, writer_prompt
from ._types import Criterion, Mistake, Policy

CHECKS_PER_ROUND: Final = 30
"""Checks the writer is asked for at most, per round."""
_TOKENS_PER_CHECK: Final = 60
_REPLY_HEADROOM: Final = 200


def writer_reply_tokens(n_checks: int) -> int:
    """The most a reply of ``n_checks`` written checks writes, thinking aside."""
    return _TOKENS_PER_CHECK * n_checks + _REPLY_HEADROOM


async def written_checks(
    session: LLMSession,
    writer: Sequence[Endpoint],
    policy: Policy,
    mistakes: Sequence[Mistake],
    *,
    current: Sequence[str] = (),
    n_checks: int = CHECKS_PER_ROUND,
    prefix: str = "W",
) -> tuple[Criterion, ...]:
    """New checks from the writer, without blanks or repeats of ``current``.

    Each gets the id ``f"{prefix}{n:02d}"``, numbered from 1.
    """
    if not mistakes or n_checks < 1:
        return ()
    reply = await session.generate(
        writer,
        writer_prompt(policy, current, mistakes, n_checks),
        role="writer",
        schema=_Checks,
        instructions=WRITER_INSTRUCTIONS,
        reply_tokens=writer_reply_tokens(n_checks),
    )
    seen = {text.strip().lower() for text in current}
    checks: list[Criterion] = []
    for check in reply.checks:
        text = check.text.strip()
        if not text or text.lower() in seen or len(checks) == n_checks:
            continue
        seen.add(text.lower())
        checks.append(Criterion(f"{prefix}{len(checks) + 1:02d}", text, check.kind))
    return tuple(checks)


def write_checks(
    policy: Policy,
    mistakes: Sequence[Mistake],
    writer: Models,
    *,
    current: Sequence[str] = (),
    n_checks: int = CHECKS_PER_ROUND,
    prefix: str = "W",
    usage: Usage | None = None,
) -> tuple[Criterion, ...]:
    """Write new yes/no checks that would tell apart the cases a model gets wrong.

    One call to the writer, which reads the policy, the checks already asked
    and the mistakes (the first 25 with their case's text, then up to 100 more
    by their reasons), and writes checks a literal model can answer from a
    case's text. Use mistakes from cases the model did not learn from, such as
    out-of-fold predictions, or the checks fit the model's own errors.

    Args:
        policy: The policy the outcomes follow.
        mistakes: The cases the current model gets wrong.
        writer: Reasoning model(s) that write the checks.
        current: The checks already asked, by text; new ones repeat none.
        n_checks: The most checks to write.
        prefix: Ids are ``f"{prefix}01"``, ``f"{prefix}02"``...
        usage: Where to record the call.

    Returns:
        The checks, as criteria of kind ``"positive"`` or ``"negative"``: pass
        them to ``PolicyClassifier(checks=...)`` to ask them as written.
    """
    return run_sync(
        awrite_checks(
            policy,
            mistakes,
            writer,
            current=current,
            n_checks=n_checks,
            prefix=prefix,
            usage=usage,
        )
    )


async def awrite_checks(
    policy: Policy,
    mistakes: Sequence[Mistake],
    writer: Models,
    *,
    current: Sequence[str] = (),
    n_checks: int = CHECKS_PER_ROUND,
    prefix: str = "W",
    usage: Usage | None = None,
) -> tuple[Criterion, ...]:
    """Write new checks from async code. See ``write_checks``."""
    models = resolve_models(writer, param="writer", task="generate")

    async def work() -> tuple[Criterion, ...]:
        return await written_checks(
            LLMSession(usage=usage),
            models,
            policy,
            mistakes,
            current=current,
            n_checks=n_checks,
            prefix=prefix,
        )

    return await run_async(work())


class _Check(BaseModel):
    text: str
    kind: Literal["positive", "negative"]


class _Checks(BaseModel):
    checks: list[_Check]
