"""What calls would cost before they are made, counted the way the session sends them.

The session sizes its calls with ``call_tokens`` and the output estimates here;
models build ``estimate_cost`` from ``estimate_generations`` and
``estimate_answers``, priced like real calls.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Final

from ._adapters import ADAPTERS
from ._spec import Endpoint
from ._usage import TokenCounts, UsageRecord, call_cost

GENERATE_OUTPUT_TOKENS: Final = 2000
"""Output tokens assumed for a writing call, reasoning included."""
ANSWER_OUTPUT_TOKENS: Final = 8
"""Output tokens assumed per answered question."""
FRAMING_TOKENS: Final = 100
"""Tokens of instructions and formatting around a call's text."""
QUESTION_TOKENS: Final = 20
"""Tokens of a question not written yet."""


def text_tokens(text: str | None) -> int:
    """A rough token count: four ASCII characters a token, any other character one."""
    if not text:
        return 0
    ascii_chars = len(text.encode("ascii", "ignore"))
    return ascii_chars // 4 + len(text) - ascii_chars


def call_tokens(*texts: str | None) -> int:
    """Input tokens of one call carrying ``texts``."""
    return FRAMING_TOKENS + sum(map(text_tokens, texts))


def mean_tokens(texts: Sequence[str]) -> float:
    """Tokens of a typical sample."""
    return sum(map(text_tokens, texts)) / len(texts) if texts else 0.0


def estimate_generations(
    models: Sequence[Endpoint],
    input_tokens: float,
    *,
    calls: int = 1,
    role: str = "generator",
) -> UsageRecord:
    """``calls`` writing calls with about ``input_tokens`` of text each."""
    tokens = TokenCounts(
        input=round(calls * (FRAMING_TOKENS + input_tokens)),
        output=calls * GENERATE_OUTPUT_TOKENS,
    )
    return _planned(models[0], role, calls, tokens)


def estimate_answers(
    models: Sequence[Endpoint],
    samples: Sequence[str],
    questions: Sequence[str | int],
    *,
    times: float = 1.0,
    role: str = "answerer",
) -> UsageRecord:
    """Answering ``questions`` about every sample, grouped per call as the session does.

    Args:
        models: The answerer's models; the first is priced.
        samples: The samples asked about.
        questions: Each question's text, or its token count if not written yet.
        times: How often the whole pass runs (draws, refinements, tree levels).
        role: Recorded with the estimate.
    """
    endpoint = models[0]
    chunks = math.ceil(
        len(questions) / ADAPTERS[endpoint.protocol].max_questions(endpoint)
    )
    asked = sum(q if isinstance(q, int) else text_tokens(q) for q in questions)
    calls = len(samples) * chunks
    one_pass = TokenCounts(
        input=calls * FRAMING_TOKENS
        + chunks * sum(map(text_tokens, samples))
        + len(samples) * asked,
        output=len(samples) * len(questions) * ANSWER_OUTPUT_TOKENS,
    )
    tokens = TokenCounts(
        input=round(times * one_pass.input), output=round(times * one_pass.output)
    )
    return _planned(endpoint, role, round(times * calls), tokens)


def _planned(
    endpoint: Endpoint, role: str, calls: int, tokens: TokenCounts
) -> UsageRecord:
    cost, source = call_cost(endpoint, tokens, None)
    return UsageRecord(
        time=time.time(),
        role=role,
        model=endpoint.name,
        cost_usd=cost,
        cost_source=source,
        input_tokens=tokens.input,
        output_tokens=tokens.output,
        calls=calls,
    )
