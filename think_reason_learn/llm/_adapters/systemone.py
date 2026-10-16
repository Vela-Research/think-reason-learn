"""The System One decision protocol: Jev's, and OpenRouter's Decisions API.

Typesafe's Jev, Microsoft-Decision-1 on Azure, and every decision model on
OpenRouter take one sample (the state) and typed questions, and return a
probability for each yes/no question and a label for each choice question.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import httpx2
from .._config import get_config
from .._questions import Answer, ChoiceQuestion, Question, YesNoQuestion
from .._spec import Endpoint
from .._usage import TokenCounts
from ._base import Answers, DecisionAdapter, ProviderError, Reply, classify
from ._http import http_client

type _Wire = dict[str, str | dict[str, None]]
type _Raw = dict[str, object]


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
        payload = _json(response)
        raw = payload.get("answers")
        raw_answers = raw if isinstance(raw, dict) else {}
        served = payload.get("model")
        return Reply(
            {qid: _answer(q, raw_answers.get(qid)) for qid, q in questions.items()},
            _tokens(payload.get("usage")),
            served if isinstance(served, str) else None,
            _cost(payload.get("usage")),
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


def _json(response: httpx2.Response) -> _Raw:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("transient", f"Unreadable response: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderError("transient", "The response is not a JSON object.")
    return payload


def _answer(question: Question, raw: object) -> Answer | None:
    if not isinstance(raw, dict):
        return None
    if isinstance(question, YesNoQuestion):
        value = raw.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return Answer.yes_probability(float(value)) if 0 <= value <= 1 else None
    label = raw.get("choice")
    if not isinstance(label, str) or label not in question.labels:
        return None
    probabilities = _probabilities(raw.get("probabilities"), question.labels)
    return (
        Answer(label, probabilities)
        if probabilities
        else Answer.certain(label, question.labels)
    )


def _probabilities(raw: object, labels: tuple[str, ...]) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    values = {label: raw.get(label) for label in labels}
    if not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()
    ):
        return None
    return {
        label: float(v) for label, v in values.items() if isinstance(v, (int, float))
    }


def _cost(usage: object) -> float | None:
    """The cost some providers (OpenRouter) report with the usage."""
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return float(cost) if isinstance(cost, (int, float)) else None


def _tokens(usage: object) -> TokenCounts:
    if not isinstance(usage, dict):
        return TokenCounts()
    sent, received = usage.get("input_tokens"), usage.get("output_tokens")
    return TokenCounts(
        input=sent if isinstance(sent, int) else 0,
        output=received if isinstance(received, int) else 0,
    )
