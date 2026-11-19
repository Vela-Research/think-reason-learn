"""Yes/no checks answered for a set of cases, through the library's answering path."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._data import TextInput, to_texts
from think_reason_learn.llm import (
    LLMSession,
    Models,
    Usage,
    resolve_models,
    run_async,
    run_sync,
)


def answer_checks(
    X: TextInput,
    checks: Sequence[str],
    answerer: Models = "jev",
    *,
    usage: Usage | None = None,
    progress: Callable[[int], None] | None = None,
) -> npt.NDArray[np.float64]:
    """The probability of yes for each case (rows) and yes/no check (columns).

    Answers go through the same path every model uses: fallbacks, retries,
    speed limits, the reply cache (an answer already paid for is reused), and
    the settings in effect, ``max_cost_usd`` included (``config_context``).

    Args:
        X: The cases: one text each, or a DataFrame (rows as ``column: value``).
        checks: The checks' text, each a yes/no question or statement.
        answerer: The model(s) that answer, in fallback order.
        usage: Where to record the calls; read its ``cost_usd`` afterwards.
        progress: Called with the number of answers each time some arrive.

    Returns:
        An array of shape ``(cases, checks)``; NaN where no model answered.
    """
    return run_sync(aanswer_checks(X, checks, answerer, usage=usage, progress=progress))


async def aanswer_checks(
    X: TextInput,
    checks: Sequence[str],
    answerer: Models = "jev",
    *,
    usage: Usage | None = None,
    progress: Callable[[int], None] | None = None,
) -> npt.NDArray[np.float64]:
    """Answer the checks from async code. See ``answer_checks``."""
    texts, _ = to_texts(X)
    models = resolve_models(answerer, param="answerer", task="answer")

    async def work() -> npt.NDArray[np.float64]:
        session = LLMSession(usage=usage, progress=progress)
        return await yes_probabilities(
            session, models, texts, list(checks), missing=np.nan
        )

    return await run_async(work())
