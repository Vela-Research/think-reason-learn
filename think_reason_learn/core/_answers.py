"""Probabilities of yes from the answerer, as a matrix: what every model reads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from think_reason_learn.llm import Endpoint, LLMSession, YesNoQuestion


async def yes_probabilities(
    session: LLMSession,
    models: Sequence[Endpoint],
    texts: Sequence[str],
    questions: Sequence[str],
    *,
    missing: float,
    draw: int = 0,
    ensemble: bool = False,
) -> npt.NDArray[np.float64]:
    """Probability of yes for each sample (rows) and yes/no question (columns).

    Args:
        session: The run's session.
        models: The answerer's models.
        texts: The samples.
        questions: The questions' text.
        missing: The value of an unanswered question: 0 (NO), or NaN to show
            it, read as NO.
        draw: Ask again instead of reusing answers of an earlier draw.
        ensemble: The draws are averaged; see ``LLMSession.answer``.
    """
    if not questions or not texts:
        return np.full((len(texts), len(questions)), missing)
    asked = {f"q{j}": YesNoQuestion(q) for j, q in enumerate(questions)}
    answers = await session.answer(
        models, texts, asked, role="answerer", draw=draw, ensemble=ensemble
    )
    return np.array(
        [
            [
                a.probability() if (a := row[qid]) is not None else missing
                for qid in asked
            ]
            for row in answers
        ]
    )
