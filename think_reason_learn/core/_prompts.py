"""Prompt pieces every model shares: labelled examples and task instructions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from think_reason_learn.core._data import Classes
from think_reason_learn.exceptions import NoOutputError
from think_reason_learn.llm import Endpoint, LLMSession

logger = logging.getLogger(__name__)

_INSTRUCTIONS_WRITER: Final = """\
You write the instructions another model will follow to write {purpose} for a
classification task. Turn the task description into clear, specific guidance:
what the classes mean, which signals in a sample are likely to matter, and what
to avoid (anything that restates the label, or that cannot be judged from a
sample's text alone). Do not invent facts about the data. Reply with the
instructions only."""


async def most_written[T](calls: Sequence[Awaitable[T]], what: str) -> list[T]:
    """The replies of ``calls`` run at once, skipping those that wrote nothing.

    One refusal or unusable reply among many (prompts, mined samples) is
    logged and skipped; any other error, such as a bad key or the budget,
    stops the run, and so does every call writing nothing.
    """
    replies = await asyncio.gather(*calls, return_exceptions=True)
    failed = [r for r in replies if isinstance(r, BaseException)]
    if fatal := next((e for e in failed if not isinstance(e, NoOutputError)), None):
        raise fatal
    written = [r for r in replies if not isinstance(r, BaseException)]
    if failed and not written:
        raise failed[0]
    if failed:
        logger.warning(
            f"{len(failed)} of {len(replies)} {what} wrote nothing: {failed[0]}"
        )
    return written


def examples_block(
    texts: Sequence[str],
    codes: npt.NDArray[np.intp],
    classes: Classes,
    indices: npt.NDArray[np.intp],
) -> str:
    """The samples at ``indices``, each with its label, ready for a prompt."""
    return "\n\n".join(
        f"Example {n} (label: {classes[codes[i]]}):\n{texts[i]}"
        for n, i in enumerate(indices, 1)
    )


async def write_instructions(
    session: LLMSession, models: Sequence[Endpoint], task: str, purpose: str
) -> str:
    """Instructions refined from ``task`` by the generator.

    Args:
        session: The run's session.
        models: The generator's models.
        task: The user's description of the task.
        purpose: What the instructions are for, e.g. ``"questions that split a
            decision tree"``.
    """
    return await session.generate(
        models,
        f"Task description:\n{task}",
        role="generator",
        instructions=_INSTRUCTIONS_WRITER.format(purpose=purpose),
    )
