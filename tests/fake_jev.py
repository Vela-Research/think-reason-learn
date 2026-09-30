"""Deterministic stand-in for ``LLM.answer_many`` (Jev answering), offline."""

from __future__ import annotations

from typing import Any, Callable, List, Sequence

from think_reason_learn.core.llms import (
    ChoiceQuestion,
    JevRequest,
    JevResult,
    TokenCounter,
)
from tests.fake_llm import FakeLLM


class FakeJevLLM(FakeLLM):
    """``FakeLLM`` for generation plus a Jev-like ``answer_many``.

    Args:
        prob: ``(state, instructions) -> probability`` for noul questions.
        choose: ``(state, question) -> label`` for choice questions.
        **kwargs: Passed to ``FakeLLM``.
    """

    def __init__(
        self,
        prob: Callable[[str, str], float | None] = lambda state, instr: 0.7,
        choose: Callable[[str, ChoiceQuestion], str | None] = lambda s, q: q.labels[0],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prob = prob
        self.choose = choose
        self.answer_requests: List[JevRequest] = []
        self.answer_calls: List[dict[str, Any]] = []

    async def answer_many(
        self,
        llm_priority: List[Any],
        requests: Sequence[JevRequest],
        *,
        budget: Any = None,
        token_counter: TokenCounter | None = None,
        caller: str = "",
        **kwargs: Any,
    ) -> List[JevResult]:
        self.answer_calls.append(
            {
                "llm_priority": llm_priority,
                "n_requests": len(requests),
                "budget": budget,
                "caller": caller,
            }
        )
        results: List[JevResult] = []
        for request in requests:
            self.answer_requests.append(request)
            answers: dict[str, Any] = {}
            for qid, question in request.questions.items():
                if isinstance(question, ChoiceQuestion):
                    answers[qid] = self.choose(request.state, question)
                else:
                    answers[qid] = self.prob(request.state, question.instructions)
            results.append(
                JevResult(answers=answers, input_tokens=100, output_tokens=1)
            )
            if token_counter is not None:
                await token_counter.append(
                    model="jev-latest", provider="jev", value=101, caller=caller
                )
        return results
