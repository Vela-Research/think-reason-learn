"""The System One decision protocol: Jev's, and OpenRouter's Decisions API.

Typesafe's Jev, Microsoft-Decision-1 on Azure, and every decision model on
OpenRouter take one sample (the state) and typed questions, and return a
probability for each yes/no question and a label for each choice question.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx2
from pydantic import BaseModel, Field, FiniteFloat, JsonValue, ValidationError

from .._config import get_config
from .._questions import Answer, ChoiceQuestion, Question, YesNoQuestion
from .._spec import Endpoint
from .._usage import TokenCounts
from ._base import Answers, DecisionAdapter, ProviderError, Reply, classify
from ._http import http_client

type _Wire = dict[str, str | dict[str, None]]


class SystemOneAdapter(DecisionAdapter):
    """POSTs System One requests with ``httpx2``."""

    async def answer(
        self,
        endpoint: Endpoint,
        sample: str,
        questions: Mapping[str, Question],
        dropped: frozenset[str],
    ) -> Reply[Answers]:
        """Answers with Jev's probabilities."""
        client = http_client(endpoint.client_key)
        body = {
            "model": endpoint.model_id,
            "state": sample,
            "questions": {qid: _wire(q) for qid, q in questions.items()},
        }
        try:
            response = await client.post(
                endpoint.base_url or "",
                json=body,
                headers=_auth(endpoint),
                timeout=get_config().timeout,
            )
        except httpx2.RequestError as exc:
            raise classify(None, f"{type(exc).__name__}: {exc}", {}) from exc
        if response.status_code != 200:
            raise classify(response.status_code, response.text[:300], response.headers)
        reply = _read(response)
        usage = _usage(reply.usage)
        return Reply(
            {qid: _answer(q, reply.answers.get(qid)) for qid, q in questions.items()},
            TokenCounts(input=usage.input_tokens, output=usage.output_tokens),
            reply.model,
            usage.cost,
            dict(response.headers),
        )


def _auth(endpoint: Endpoint) -> dict[str, str]:
    key = endpoint.api_key or ""
    return (
        {"api-key": key}
        if endpoint.api_key_header
        else {"Authorization": f"Bearer {key}"}
    )


def _wire(question: Question) -> _Wire:
    if isinstance(question, ChoiceQuestion):
        criteria: dict[str, None] = dict.fromkeys(question.labels)
        return {"type": "choice", "instructions": question.text, "criteria": criteria}
    return {"type": "noul", "instructions": question.text}


def _read(response: httpx2.Response) -> _Reply:
    """The reply; an unreadable one is a transient failure."""
    try:
        return _Reply.model_validate_json(response.content)
    except ValidationError as exc:
        raise ProviderError("transient", f"Unreadable response: {exc}") from exc


def _answer(question: Question, raw: JsonValue) -> Answer | None:
    """One answer, if it is readable and of the question's kind; else ``None``."""
    try:
        answer = _RawAnswer.model_validate(raw)
    except ValidationError:
        return None
    if isinstance(question, YesNoQuestion):
        return Answer.yes_probability(answer.noul) if answer.noul is not None else None
    label = answer.choice
    if label is None or label not in question.labels:
        return None
    if set(question.labels) <= set(answer.probabilities):
        return Answer(label, {k: answer.probabilities[k] for k in question.labels})
    return Answer.certain(label, question.labels)


def _usage(raw: JsonValue) -> _Usage:
    try:
        return _Usage.model_validate(raw or {})
    except ValidationError:
        return _Usage()


class _Reply(BaseModel):
    answers: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    model: str | None = None
    usage: JsonValue = None


class _RawAnswer(BaseModel):
    noul: float | None = Field(default=None, ge=0, le=1)
    choice: str | None = None
    probabilities: dict[str, FiniteFloat] = Field(default_factory=dict[str, float])


class _Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    """The cost some providers (OpenRouter) report with the usage."""
