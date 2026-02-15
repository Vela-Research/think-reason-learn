"""Deterministic fake LLM for offline RRF testing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Type

from think_reason_learn.core.llms._schemas import (
    LLMChoice,
    LLMResponse,
    NOT_GIVEN,
    NotGiven,
    T,
)
from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.rrf._rrf import Answer, Questions
from think_reason_learn.rrf._prompts import num_questions_tag


_FAKE_PROVIDER = OpenAIChoice(model="gpt-4.1-nano")


class FakeLLM:
    """Drop-in replacement for ``LLM`` that returns canned responses.

    Dispatches based on ``response_format`` to handle the 4 RRF call sites:

    1. ``set_tasks``  (``str``)  -- template with ``<number_of_questions>`` tag
    2. ``_generate_questions``  (``Questions``)  -- pydantic model
    3. ``_answer_single_question``  (``Answer``)  -- pydantic model
    4. ``_answer_questions_batch``  (``str``)  -- JSON list

    The two ``str``-typed sites are distinguished by query content:
    ``set_tasks`` sends ``"Generate YES/NO questions for:..."`` while
    batch sends ``"You are a VC analyst..."``.

    Args:
        default_answer: ``"YES"``, ``"NO"``, or ``"ALTERNATE"``.
        questions_per_call: Number of questions returned per generation call.
    """

    def __init__(
        self,
        default_answer: str = "YES",
        questions_per_call: int = 3,
    ) -> None:
        self.default_answer = default_answer
        self.questions_per_call = questions_per_call
        self._call_count = 0
        self.calls: List[Dict[str, Any]] = []

    def _get_answer(self, index: int = 0) -> Literal["YES", "NO"]:
        if self.default_answer == "ALTERNATE":
            return "YES" if index % 2 == 0 else "NO"
        if self.default_answer == "NO":
            return "NO"
        return "YES"

    async def respond(
        self,
        query: str,
        llm_priority: List[LLMChoice],
        response_format: Type[T],
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        **kwargs: Dict[str, Any],
    ) -> LLMResponse[Any]:
        """Route to the appropriate canned response."""
        self._call_count += 1
        self.calls.append(
            {
                "query": query,
                "response_format": response_format,
                "n": self._call_count,
            }
        )

        if response_format is Questions:
            return self._questions_response()
        if response_format is Answer:
            return self._answer_response()
        if response_format is not str:
            raise TypeError(
                f"FakeLLM: unknown response_format {response_format!r}. "
                "Add a handler for this new call site."
            )
        # response_format is str — distinguish set_tasks vs batch by query
        if "generate yes/no" in query.lower():
            return self._set_tasks_response()
        return self._batch_answer_response(query)

    # ------------------------------------------------------------------
    # Canned responses
    # ------------------------------------------------------------------

    def _set_tasks_response(self) -> LLMResponse[str]:
        return LLMResponse(
            response=(
                f"Generate {num_questions_tag} YES/NO questions to determine "
                "whether a founder is likely to succeed."
            ),
            logprobs=[("t", -0.1)],
            total_tokens=50,
            provider_model=_FAKE_PROVIDER,
        )

    def _questions_response(self) -> LLMResponse[Questions]:
        qs = [
            f"Does the person have relevant technical experience? (variant {i})"
            for i in range(self.questions_per_call)
        ]
        return LLMResponse(
            response=Questions(
                questions=qs,
                cumulative_memory="Fake memory: technical background matters.",
            ),
            logprobs=[("t", -0.05)],
            total_tokens=100,
            provider_model=_FAKE_PROVIDER,
        )

    def _answer_response(self) -> LLMResponse[Answer]:
        return LLMResponse(
            response=Answer(answer=self._get_answer(self._call_count)),
            logprobs=[("YES", -0.01)],
            total_tokens=15,
            provider_model=_FAKE_PROVIDER,
        )

    def _batch_answer_response(self, query: str) -> LLMResponse[str]:
        indices = [int(m) for m in re.findall(r"Sample (\d+):", query)]
        results = [{"sample_index": i, "answer": self._get_answer(i)} for i in indices]
        return LLMResponse(
            response=json.dumps(results),
            logprobs=[],
            total_tokens=20 * max(len(indices), 1),
            provider_model=_FAKE_PROVIDER,
        )
