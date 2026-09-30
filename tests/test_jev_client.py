"""Offline tests for the Jev client against an in-process fake Typesafe server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

from think_reason_learn.core._config import Settings
from think_reason_learn.core.exceptions import JevAuthError, JevCostCapError
from think_reason_learn.core.llms import ChoiceQuestion, JevChoice, NoulQuestion
from think_reason_learn.core.llms._jev.client import (
    JevBudget,
    JevClient,
    JevRequest,
)


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
    ) -> None:
        self.bodies: List[Dict[str, Any]] = []
        self.headers: List[httpx.Headers] = []
        self._responses = list(responses or [])
        self.input_tokens = input_tokens

    def ok(self, body: Dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": _answers_for(body["questions"]),
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 7},
            },
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        if self._responses:
            return self._responses.pop(0)(body)
        return self.ok(body)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: List[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _client(
    server: FakeJevServer, tmp_path: Path, **kwargs: Any
) -> tuple[JevClient, SleepRecorder]:
    sleep = SleepRecorder()
    client = JevClient(
        api_key="test-key",
        transport=server.transport(),
        cache_dir=tmp_path / "cache",
        sleep=sleep,
        **kwargs,
    )
    return client, sleep


def _noul_request(
    n: int = 1, state: str = "Founder: ten years in fintech."
) -> JevRequest:
    return JevRequest(
        state=state,
        questions={
            f"q{i}": NoulQuestion(instructions=f"Question {i}?") for i in range(n)
        },
    )


# ---------------------------------------------------------------------------
# Choice and settings
# ---------------------------------------------------------------------------


def test_jev_choice_defaults() -> None:
    choice = JevChoice()
    assert choice.provider == "jev"
    assert choice.model == "jev-latest"
    assert choice.max_cost_usd == 10.0
    assert choice.cache is True


def test_settings_reads_typesafe_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    assert Settings(_env_file=None).TYPESAFE_API_KEY == "from-env"  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sends_one_request_per_sample_with_typed_questions(
    tmp_path: Path,
) -> None:
    server = FakeJevServer()
    client, _ = _client(server, tmp_path)
    request = JevRequest(
        state="Founder: ten years in fintech.",
        questions={
            "a": NoulQuestion(instructions="Has fintech experience?"),
            "b": ChoiceQuestion(instructions="Seniority?", labels=["junior", "senior"]),
        },
    )

    [result] = await client.answer_many(
        [request], model="jev-latest", budget=JevBudget(10.0)
    )

    assert len(server.bodies) == 1
    assert server.bodies[0] == {
        "model": "jev-latest",
        "state": "Founder: ten years in fintech.",
        "questions": {
            "a": {"type": "noul", "instructions": "Has fintech experience?"},
            "b": {
                "type": "choice",
                "instructions": "Seniority?",
                "criteria": {"junior": None, "senior": None},
            },
        },
    }
    assert server.headers[0]["authorization"] == "Bearer test-key"
    assert result.answers == {"a": 0.8, "b": "junior"}
    assert result.error is None
    assert result.served_model == "jev-1.13.0"


@pytest.mark.asyncio
async def test_returns_token_usage(tmp_path: Path) -> None:
    server = FakeJevServer(input_tokens=1234)
    client, _ = _client(server, tmp_path)

    [result] = await client.answer_many(
        [_noul_request()], model="jev-latest", budget=JevBudget(10.0)
    )

    assert (result.input_tokens, result.output_tokens) == (1234, 7)
    assert result.cached is False


@pytest.mark.asyncio
async def test_chunks_questions_at_fifty_per_request(tmp_path: Path) -> None:
    server = FakeJevServer()
    client, _ = _client(server, tmp_path)

    [result] = await client.answer_many(
        [_noul_request(120)], model="jev-latest", budget=JevBudget(10.0)
    )

    assert sorted(len(b["questions"]) for b in server.bodies) == [20, 50, 50]
    assert set(result.answers) == {f"q{i}" for i in range(120)}
    assert all(v == 0.8 for v in result.answers.values())
    assert result.input_tokens == 3000


@pytest.mark.asyncio
async def test_out_of_range_or_unknown_answers_become_none(tmp_path: Path) -> None:
    def bad(body: Dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"a": {"noul": 1.7}, "b": {"choice": "maybe"}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    server = FakeJevServer([bad])
    client, _ = _client(server, tmp_path)
    request = JevRequest(
        state="s",
        questions={
            "a": NoulQuestion(instructions="?"),
            "b": ChoiceQuestion(instructions="?", labels=["x", "y"]),
            "c": NoulQuestion(instructions="missing from the response"),
        },
    )

    [result] = await client.answer_many(
        [request], model="jev-latest", budget=JevBudget(10.0)
    )

    assert result.answers == {"a": None, "b": None, "c": None}


# ---------------------------------------------------------------------------
# Retries and errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_429_honouring_retry_after(tmp_path: Path) -> None:
    server = FakeJevServer(
        [lambda b: httpx.Response(429, headers={"Retry-After": "7"}, text="slow")]
    )
    client, sleep = _client(server, tmp_path)

    [result] = await client.answer_many(
        [_noul_request()], model="jev-latest", budget=JevBudget(10.0)
    )

    assert len(server.bodies) == 2
    assert sleep.calls and sleep.calls[0] >= 7
    assert result.answers == {"q0": 0.8}


@pytest.mark.asyncio
async def test_retries_timeouts(tmp_path: Path) -> None:
    def timeout(body: Dict[str, Any]) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    server = FakeJevServer([timeout])
    client, sleep = _client(server, tmp_path)

    [result] = await client.answer_many(
        [_noul_request()], model="jev-latest", budget=JevBudget(10.0)
    )

    assert len(server.bodies) == 2
    assert len(sleep.calls) == 1
    assert result.answers == {"q0": 0.8}


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts_on_5xx(tmp_path: Path) -> None:
    server = FakeJevServer([lambda b: httpx.Response(503, text="down")] * 10)
    client, sleep = _client(server, tmp_path, max_attempts=3)

    [result] = await client.answer_many(
        [_noul_request()], model="jev-latest", budget=JevBudget(10.0)
    )

    assert len(server.bodies) == 3
    assert result.answers == {"q0": None}
    assert result.error is not None and "503" in result.error
    assert sleep.calls == sorted(sleep.calls)


@pytest.mark.asyncio
async def test_unauthorised_key_raises(tmp_path: Path) -> None:
    server = FakeJevServer([lambda b: httpx.Response(401, text="bad key")])
    client, _ = _client(server, tmp_path)

    with pytest.raises(JevAuthError, match="TYPESAFE_API_KEY"):
        await client.answer_many(
            [_noul_request()], model="jev-latest", budget=JevBudget(10.0)
        )


@pytest.mark.asyncio
async def test_rejected_sample_fails_without_retry_or_truncation(
    tmp_path: Path,
) -> None:
    server = FakeJevServer([lambda b: httpx.Response(413, text="state too long")])
    client, sleep = _client(server, tmp_path)
    long_state = "x" * 50_000

    [result] = await client.answer_many(
        [_noul_request(state=long_state)], model="jev-latest", budget=JevBudget(10.0)
    )

    assert len(server.bodies) == 1
    assert server.bodies[0]["state"] == long_state
    assert sleep.calls == []
    assert result.answers == {"q0": None}
    assert result.error is not None and "state too long" in result.error


# ---------------------------------------------------------------------------
# Cost: estimate, cap before spending, cap mid-run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prints_estimate_before_spending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeJevServer()
    client, _ = _client(server, tmp_path)

    await client.answer_many(
        [_noul_request(), _noul_request(state="other")],
        model="jev-latest",
        budget=JevBudget(10.0),
    )

    err = capsys.readouterr().err
    assert "Jev" in err and "2 requests" in err and "$" in err and "cap" in err


@pytest.mark.asyncio
async def test_estimate_above_cap_stops_before_any_request(tmp_path: Path) -> None:
    server = FakeJevServer()
    client, _ = _client(server, tmp_path)
    requests = [_noul_request(state="s" * 3000 + str(i)) for i in range(10)]

    with pytest.raises(JevCostCapError, match="max_cost_usd"):
        await client.answer_many(
            requests, model="jev-latest", budget=JevBudget(0.000001)
        )

    assert server.bodies == []


@pytest.mark.asyncio
async def test_actual_spend_above_cap_stops_mid_run(tmp_path: Path) -> None:
    # Each response reports 1M input tokens = $0.042, far above the estimate.
    server = FakeJevServer(input_tokens=1_000_000)
    client, _ = _client(server, tmp_path, concurrency=1)
    budget = JevBudget(0.05)
    requests = [_noul_request(state=f"sample {i}") for i in range(5)]

    with pytest.raises(JevCostCapError):
        await client.answer_many(requests, model="jev-latest", budget=budget)

    assert len(server.bodies) == 2
    assert budget.spent_usd == pytest.approx(2 * 0.042)


@pytest.mark.asyncio
async def test_budget_is_shared_across_calls(tmp_path: Path) -> None:
    server = FakeJevServer(input_tokens=1_000_000)
    client, _ = _client(server, tmp_path)
    budget = JevBudget(1.0)

    await client.answer_many([_noul_request()], model="jev-latest", budget=budget)
    await client.answer_many(
        [_noul_request(state="second")], model="jev-latest", budget=budget
    )

    assert budget.spent_usd == pytest.approx(2 * 0.042)


# ---------------------------------------------------------------------------
# Reuse cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_answers_repeat_requests_without_calling_jev(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeJevServer()
    client, _ = _client(server, tmp_path)

    await client.answer_many(
        [_noul_request(3)], model="jev-latest", budget=JevBudget(10.0)
    )
    capsys.readouterr()
    budget = JevBudget(10.0)
    [again] = await client.answer_many(
        [_noul_request(3)], model="jev-latest", budget=budget
    )

    assert len(server.bodies) == 1
    assert again.cached is True
    assert again.answers == {"q0": 0.8, "q1": 0.8, "q2": 0.8}
    assert budget.spent_usd == 0
    assert "0 requests" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_cache_key_includes_model_and_questions(tmp_path: Path) -> None:
    server = FakeJevServer()
    client, _ = _client(server, tmp_path)
    budget = JevBudget(10.0)

    await client.answer_many([_noul_request(1)], model="jev-latest", budget=budget)
    await client.answer_many([_noul_request(1)], model="jev-1.13.0", budget=budget)
    await client.answer_many([_noul_request(2)], model="jev-latest", budget=budget)

    assert len(server.bodies) == 3


@pytest.mark.asyncio
async def test_cache_off_switch(tmp_path: Path) -> None:
    server = FakeJevServer()
    client, _ = _client(server, tmp_path)
    budget = JevBudget(10.0)

    for _ in range(2):
        await client.answer_many(
            [_noul_request()], model="jev-latest", budget=budget, use_cache=False
        )

    assert len(server.bodies) == 2
    assert not (tmp_path / "cache").exists()


@pytest.mark.asyncio
async def test_failed_requests_are_not_cached(tmp_path: Path) -> None:
    server = FakeJevServer([lambda b: httpx.Response(422, text="bad question")])
    client, _ = _client(server, tmp_path)
    budget = JevBudget(10.0)

    await client.answer_many([_noul_request()], model="jev-latest", budget=budget)
    [second] = await client.answer_many(
        [_noul_request()], model="jev-latest", budget=budget
    )

    assert len(server.bodies) == 2
    assert second.answers == {"q0": 0.8}


# Billed by Typesafe on 30 September 2026 (jev-1.13.0): this 410-byte request
# used 342 input tokens; 2,194 ECHR requests used 188 + bytes / 4.42 on average.
SMOKE = JevRequest(
    state="text: The applicant waited nine years for the domestic courts to rule "
    "on his claim.",
    questions={
        "q0": NoulQuestion(
            instructions="Was there a delay in the judicial proceedings "
            "attributable to the State?"
        ),
        "q1": ChoiceQuestion(
            instructions="How long did the proceedings last?",
            labels=["under 1 year", "1 to 5 years", "over 5 years"],
        ),
    },
)


def test_estimate_covers_a_small_request_as_billed(tmp_path: Path) -> None:
    client, _ = _client(FakeJevServer(), tmp_path)

    estimate = client.estimate([SMOKE], "jev-latest")

    assert estimate.input_tokens >= 342


def test_estimate_for_long_samples_stays_close_to_billing(tmp_path: Path) -> None:
    client, _ = _client(FakeJevServer(), tmp_path)
    request = _noul_request(14, state="x" * 20_000)
    [chunk] = client._chunks([request], "jev-latest")
    billed = 188 + len(chunk.body) / 4.42

    estimate = client.estimate([request], "jev-latest")

    assert billed <= estimate.input_tokens <= 1.15 * billed
