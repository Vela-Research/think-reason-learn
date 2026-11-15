"""Cancelling a run: no call starts after the cancel, cut-off calls are counted."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable, Iterator, Mapping
from concurrent import futures

import numpy as np
import pytest
from pydantic import BaseModel

from think_reason_learn import set_limits
from think_reason_learn.core._classifier import Labels, Proba, TRLClassifier
from think_reason_learn.llm import (
    LLM,
    Answers,
    Endpoint,
    LLMSession,
    Question,
    YesNoQuestion,
    _loop,
    resolve_models,
    run_sync,
)
from think_reason_learn.llm._adapters import ADAPTERS, Reply
from think_reason_learn.llm._loop import io_loop, stop_requested
from think_reason_learn.testing import FakeLLM, GenerateRequest, fake_llms

QUESTION = "Is the founder an engineer?"
X = [
    f"Founder {i:03d} is {'an engineer' if i % 2 else 'a lawyer'}." for i in range(200)
]
Y = ["hit" if i % 2 else "miss" for i in range(200)]
IN_FLIGHT = 4


class Written(BaseModel):
    text: str


class OneQuestion(TRLClassifier):
    """Writes one question, then asks it about every sample (one call each)."""

    def __init__(self, task: str = "t", judge: LLM | None = None) -> None:
        self.task = task
        self.judge = judge
        self.instructions = None

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        writer = resolve_models("fake:writer", param="writer", task="generate")
        self.question_ = (
            await session.generate(writer, "Ask", role="generator", schema=Written)
        ).text
        answers = await session.answer(
            resolve_models(self.judge or "fake:judge", param="judge", task="answer"),
            texts,
            {"q": YesNoQuestion(self.question_)},
            role="answerer",
        )
        self.answered_ = sum(a["q"] is not None for a in answers)

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        return np.full((len(texts), len(self.classes_)), 1 / len(self.classes_))


class Held:
    """The fake adapter, holding answers on the library's loop until let go.

    The first ``free`` answers are not held. ``sent`` counts the answer
    requests started; ``filled`` is set once ``IN_FLIGHT`` are held.
    """

    def __init__(self) -> None:
        self.inner = ADAPTERS["fake"]
        self.free = 0
        self.sent = 0
        self.filled = threading.Event()
        self.gate = asyncio.Event()  # bound to the library's loop on first wait

    def max_questions(self, endpoint: Endpoint) -> int:
        return self.inner.max_questions(endpoint)

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        return await self.inner.generate(endpoint, request, dropped)

    async def answer(
        self,
        endpoint: Endpoint,
        sample: str,
        questions: Mapping[str, Question],
        dropped: frozenset[str],
    ) -> Reply[Answers]:
        self.sent += 1
        if self.sent == self.free + IN_FLIGHT:
            self.filled.set()
        if self.sent > self.free:
            await self.gate.wait()
        return await self.inner.answer(endpoint, sample, questions, dropped)

    def let_go(self) -> None:
        io_loop().call_soon_threadsafe(self.gate.set)


@pytest.fixture
def held() -> Iterator[Held]:
    """Fakes whose answers wait to be let go, at most ``IN_FLIGHT`` at once."""
    writer = FakeLLM(generate=lambda request: Written(text=QUESTION))
    judge = FakeLLM(
        answer=lambda s, q: 0.9 if "engineer" in s else 0.1, kind="decision"
    )
    with fake_llms(writer=writer, judge=judge):
        adapter = Held()
        ADAPTERS["fake"] = adapter
        set_limits("fake:judge", max_concurrency=IN_FLIGHT)
        try:
            yield adapter
        finally:
            adapter.let_go()
            ADAPTERS["fake"] = adapter.inner


async def _started(model: OneQuestion, held: Held) -> asyncio.Task[OneQuestion]:
    """``model.afit`` running, with ``IN_FLIGHT`` answers held and the rest queued."""
    task = asyncio.create_task(model.afit(X, Y))
    assert await asyncio.to_thread(held.filled.wait, 10)
    return task


async def _cancel(task: asyncio.Task[OneQuestion]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _library_tasks() -> list[asyncio.Task[object]]:
    async def others() -> list[asyncio.Task[object]]:
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    return asyncio.run_coroutine_threadsafe(others(), io_loop()).result(10)


@pytest.mark.asyncio
async def test_no_call_starts_after_cancel_returns(held: Held) -> None:
    task = await _started(OneQuestion(), held)
    # Hold the library's loop with the answers out let go: their slots free up
    # with queued calls ready to take them, as the cancel crosses threads.
    holding, go_on = threading.Event(), threading.Event()

    def hold() -> None:
        held.gate.set()
        holding.set()
        go_on.wait(10)

    io_loop().call_soon_threadsafe(hold)
    try:
        assert await asyncio.to_thread(holding.wait, 10)
        task.cancel()
        for _ in range(3):  # the caller's side of the cancel runs
            await asyncio.sleep(0)
    finally:
        go_on.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    alive = _library_tasks()  # after a turn of the library's loop
    assert held.sent == IN_FLIGHT  # no call started after the cancel
    assert alive == []  # and nothing is left to start one later


@pytest.mark.asyncio
async def test_a_refit_after_cancel_works_and_reuses_the_cache(held: Held) -> None:
    held.free = 6  # answered and cached before the cancel
    model = OneQuestion()
    await _cancel(await _started(model, held))
    sent_before = held.sent
    held.let_go()

    model = await model.afit(X, Y)

    assert model.answered_ == len(X)
    from_cache = {r.role: r.calls for r in model.usage_ if r.cached}
    assert from_cache == {"generator": 1, "answerer": held.free}
    assert held.sent - sent_before == len(X) - held.free


def test_ctrl_c_in_sync_code_stops_the_work_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, finished = threading.Event(), threading.Event()
    stopped_inside: list[bool] = []

    async def work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped_inside.append(stop_requested())
            finished.set()

    waits = 0

    def interrupted(
        fs: Iterable[futures.Future[None]], timeout: float | None = None
    ) -> object:
        nonlocal waits
        waits += 1
        if waits == 1:
            started.wait(10)
            raise KeyboardInterrupt
        return futures.wait(fs, timeout=timeout)

    monkeypatch.setattr(_loop, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_sync(work())
    assert finished.is_set() and stopped_inside == [True]
    assert run_sync(asyncio.sleep(0, "still usable")) == "still usable"
