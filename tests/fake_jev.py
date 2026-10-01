"""Deterministic stand-in for ``LLM.answer_many`` (Jev answering), offline."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Sequence

import httpx

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


def _answers_for(questions: Dict[str, Any], prob: float = 0.8) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for qid, q in questions.items():
        if q["type"] == "noul":
            out[qid] = {"noul": prob}
        else:
            out[qid] = {"choice": next(iter(q["criteria"]))}
    return out


class FakeJevServer:
    """Records every request and answers like Typesafe's System One endpoint."""

    def __init__(
        self,
        responses: List[Callable[[Dict[str, Any]], httpx.Response]] | None = None,
        input_tokens: int = 1000,
        prob: float = 0.8,
        status_for_state: Dict[str, int] | None = None,
    ) -> None:
        self.bodies: List[Dict[str, Any]] = []
        self.headers: List[httpx.Headers] = []
        self._responses = list(responses or [])
        self.input_tokens = input_tokens
        self.prob = prob
        self.status_for_state = status_for_state or {}

    def ok(self, body: Dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": _answers_for(body["questions"], self.prob),
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 7},
            },
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        if body["state"] in self.status_for_state:
            return httpx.Response(self.status_for_state[body["state"]], text="rejected")
        if self._responses:
            return self._responses.pop(0)(body)
        return self.ok(body)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
