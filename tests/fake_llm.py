"""Deterministic fake LLM for offline RRF and GPTree testing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Type, get_args, get_origin

from think_reason_learn.core.llms._schemas import (
    LLMChoice,
    LLMResponse,
    NOT_GIVEN,
    NotGiven,
    T,
)
from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.rrf._rrf import Answer as RRFAnswer
from think_reason_learn.rrf._rrf import Questions as RRFQuestions
from think_reason_learn.rrf._prompts import num_questions_tag
from think_reason_learn.gptree._gptree import Question as GPTreeQuestion
from think_reason_learn.gptree._gptree import Questions as GPTreeQuestions


_FAKE_PROVIDER = OpenAIChoice(model="gpt-4.1-nano")


def _is_gptree_questions(response_format: type) -> bool:
    """Check if response_format is GPTree's Questions (List[Question] not List[str])."""
    return response_format is GPTreeQuestions


def _is_rrf_questions(response_format: type) -> bool:
    """Check if response_format is RRF's Questions (List[str])."""
    return response_format is RRFQuestions


def _is_dynamic_answer_model(response_format: type) -> bool:
    """Check for dynamically-created Answer model.

    GPTree's _make_answer_model creates models with name "Answer"
    and a single ``answer`` field whose annotation is a Literal type.
    """
    if getattr(response_format, "__name__", None) != "Answer":
        return False
    # Exclude the static RRF Answer class
    if response_format is RRFAnswer:
        return False
    # Check for a Literal-typed answer field (dynamic model from create_model)
    fields = getattr(response_format, "model_fields", {})
    answer_field = fields.get("answer")
    if answer_field is None:
        return False
    annotation = answer_field.annotation
    return get_origin(annotation) is Literal


class FakeLLM:
    """Drop-in replacement for ``LLM`` that returns canned responses.

    Dispatches based on ``response_format`` to handle call sites from both
    RRF and GPTree:

    **RRF call sites:**
    1. ``set_tasks``  (``str``)  -- template with ``<number_of_questions>`` tag
    2. ``_generate_questions``  (``RRFQuestions``)  -- pydantic model
    3. ``_answer_single_question``  (``RRFAnswer``)  -- pydantic model
    4. ``_answer_questions_batch``  (``str``)  -- JSON list

    **GPTree call sites:**
    1. ``set_tasks``  (``str``)  -- query contains "Build a decision tree"
    2. ``_generate_questions``  (``GPTreeQuestions``)  -- pydantic model
    3. ``_answer_question_for_row``  (dynamic ``Answer``)  -- Literal-typed choices

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

    @property
    def call_count(self) -> int:
        """Alias for ``_call_count`` (used by cost-sensitive tests)."""
        return self._call_count

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
                "instructions": instructions,
                "n": self._call_count,
            }
        )

        # --- Pydantic model dispatch (by identity, then by introspection) ---

        if _is_rrf_questions(response_format):
            return self._rrf_questions_response()

        if _is_gptree_questions(response_format):
            return self._gptree_questions_response()

        if response_format is RRFAnswer:
            return self._rrf_answer_response()

        if _is_dynamic_answer_model(response_format):
            fields = getattr(response_format, "model_fields", {})
            choices = get_args(fields["answer"].annotation)
            return self._gptree_answer_response(response_format, choices)

        # --- str dispatch (set_tasks vs batch, RRF vs GPTree) ---

        if response_format is str:
            if "generate yes/no" in query.lower():
                return self._rrf_set_tasks_response()
            if "build a decision tree" in query.lower():
                return self._gptree_set_tasks_response()
            return self._rrf_batch_answer_response(query)

        raise TypeError(
            f"FakeLLM: unknown response_format {response_format!r}. "
            "Add a handler for this new call site."
        )

    # ------------------------------------------------------------------
    # RRF canned responses
    # ------------------------------------------------------------------

    def _rrf_set_tasks_response(self) -> LLMResponse[str]:
        return LLMResponse(
            response=(
                f"Generate {num_questions_tag} YES/NO questions to determine "
                "whether a founder is likely to succeed."
            ),
            logprobs=[("t", -0.1)],
            total_tokens=50,
            provider_model=_FAKE_PROVIDER,
        )

    def _rrf_questions_response(self) -> LLMResponse[RRFQuestions]:
        qs = [
            f"Does the person have relevant technical experience? (variant {i})"
            for i in range(self.questions_per_call)
        ]
        return LLMResponse(
            response=RRFQuestions(
                questions=qs,
                cumulative_memory="Fake memory: technical background matters.",
            ),
            logprobs=[("t", -0.05)],
            total_tokens=100,
            provider_model=_FAKE_PROVIDER,
        )

    def _rrf_answer_response(self) -> LLMResponse[RRFAnswer]:
        return LLMResponse(
            response=RRFAnswer(answer=self._get_answer(self._call_count)),
            logprobs=[("YES", -0.01)],
            total_tokens=15,
            provider_model=_FAKE_PROVIDER,
        )

    def _rrf_batch_answer_response(self, query: str) -> LLMResponse[str]:
        indices = [int(m) for m in re.findall(r"Sample (\d+):", query)]
        results = [{"sample_index": i, "answer": self._get_answer(i)} for i in indices]
        return LLMResponse(
            response=json.dumps(results),
            logprobs=[],
            total_tokens=20 * max(len(indices), 1),
            provider_model=_FAKE_PROVIDER,
        )

    # ------------------------------------------------------------------
    # GPTree canned responses
    # ------------------------------------------------------------------

    def _gptree_set_tasks_response(self) -> LLMResponse[str]:
        return LLMResponse(
            response=(
                f"Generate {num_questions_tag} discriminative questions for "
                "the classification task. Each question should have clear, "
                "mutually exclusive answer choices."
            ),
            logprobs=[("t", -0.1)],
            total_tokens=50,
            provider_model=_FAKE_PROVIDER,
        )

    def _gptree_questions_response(self) -> LLMResponse[GPTreeQuestions]:
        qs = [
            GPTreeQuestion(
                value=f"Does the founder have trait {i}?",
                choices=["Yes", "No"],
                question_type="INFERENCE",
            )
            for i in range(self.questions_per_call)
        ]
        return LLMResponse(
            response=GPTreeQuestions(
                questions=qs,
                cumulative_memory="Fake memory: observed patterns in backgrounds.",
            ),
            logprobs=[("t", -0.05)],
            total_tokens=100,
            provider_model=_FAKE_PROVIDER,
        )

    def _gptree_answer_response(
        self, response_format: type, choices: tuple[str, ...]
    ) -> LLMResponse[Any]:
        """Return an answer cycling through available choices."""
        chosen = choices[self._call_count % len(choices)]
        return LLMResponse(
            response=response_format(answer=chosen),
            logprobs=[("ans", -0.01)],
            total_tokens=15,
            provider_model=_FAKE_PROVIDER,
        )

    def reset(self) -> None:
        """Reset call tracking."""
        self._call_count = 0
        self.calls = []
