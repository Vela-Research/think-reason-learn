"""Every test runs offline, with a fresh cache and fresh shared state."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx2
import pytest

from think_reason_learn import LLM
from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.gptree import GPTree
from think_reason_learn.llm import (
    Price,
    Question,
    _calls,
    _catalog,
    _config,
    _limiter,
    config_context,
)
from think_reason_learn.llm._adapters import _http
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining
from think_reason_learn.rrf import RRF
from think_reason_learn.testing import FakeLLM, GenerateRequest, fake_llms

type Handler = Callable[[httpx2.Request], httpx2.Response]
type Make = Callable[[], TRLClassifier]

_WRITER, _JUDGE = (
    LLM("fake:writer", price=Price(1, 4)),
    LLM("fake:judge", price=Price(1)),
)
_X = [
    f"Founder {i} is {'good' if i % 2 else 'bad'} at building companies."
    for i in range(80)
]
_Y = ["good" if i % 2 else "bad" for i in range(80)]

_MODELS: dict[str, Make] = {
    "rrf": lambda: RRF(task="t", generator=_WRITER, answerer=_JUDGE, n_prompts=3, cv=2),
    "rrf-screened": lambda: RRF(
        task="t",
        generator=_WRITER,
        answerer=_JUDGE,
        n_prompts=3,
        cv=2,
        screening_size=20,
        screened_questions=5,
    ),
    "policy": lambda: PolicyInduction(
        task="t",
        generator=_WRITER,
        answerer=_JUDGE,
        n_samples=20,
        n_rounds=2,
        batch_size=4,
        pos_label="good",
    ),
    "policy-rules": lambda: PolicyInduction(
        task="t",
        generator=_WRITER,
        answerer=_JUDGE,
        n_samples=20,
        n_rounds=1,
        batch_size=4,
        predictor="rules",
        cv=2,
        pos_label="good",
    ),
    "rule-mining": lambda: ReasonedRuleMining(
        task="t",
        generator=_WRITER,
        answerer=_JUDGE,
        n_samples=20,
        cv=2,
        pos_label="good",
    ),
    "gptree": lambda: GPTree(task="t", generator=_WRITER, answerer=_JUDGE, critic=True),
}

OPENROUTER_DECISIONS = {
    "data": [
        {
            "id": model_id,
            "architecture": {"output_modalities": ["decisions"]},
            "pricing": {"prompt": prompt, "completion": "0"},
        }
        for model_id, prompt in [
            ("openai/gpt-6-luna-decisions", "0.0000001"),
            ("typesafe/jev-1.14", "0.000000042"),
            ("microsoft/microsoft-decision-1", "0.000000042"),
        ]
    ]
}
"""What OpenRouter's catalog of decision models returns, for offline tests."""


def _reset() -> None:
    _limiter._limiters.clear()
    _config._limits.clear()
    _calls._dropped.clear()
    _http.set_test_transport(None)
    _catalog._loaded = None


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A cache in a temporary folder, two retries, no shared state, no network."""
    _reset()
    monkeypatch.setattr(_catalog, "_download", lambda: OPENROUTER_DECISIONS)
    with config_context(cache_dir=tmp_path, max_retries=2):
        yield
    _reset()


@pytest.fixture
def wire() -> Callable[[Handler], list[httpx2.Request]]:
    """Answer HTTP calls with ``handler``; returns the requests sent."""

    def install(handler: Handler) -> list[httpx2.Request]:
        sent: list[httpx2.Request] = []

        def record(request: httpx2.Request) -> httpx2.Response:
            sent.append(request)
            return handler(request)

        _http.set_test_transport(httpx2.MockTransport(record))
        return sent

    return install


@pytest.fixture(params=sorted(_MODELS))
def any_model(request: pytest.FixtureRequest) -> tuple[str, Make]:
    """Every LLM model, in each of its main settings: its name and factory."""
    name: str = request.param
    return name, _MODELS[name]


@pytest.fixture
def generic_fakes() -> Iterator[tuple[FakeLLM, FakeLLM]]:
    """A writer that fills every model's schemas, and a decision judge."""
    writer, judge = _writer(), FakeLLM(answer=_judge, kind="decision")
    with fake_llms(writer=writer, judge=judge):
        yield writer, judge


@pytest.fixture
def founders() -> tuple[list[str], list[str]]:
    """Eighty founders, half of them good."""
    return list(_X), list(_Y)


def _writer() -> FakeLLM:
    ids = itertools.count()

    def write(request: GenerateRequest) -> dict[str, object] | str:
        fields = request.schema.model_fields if request.schema is not None else {}
        if "questions" in fields:
            texts = [f"Is fact {next(ids)} true?" for _ in range(10)]
            plain = fields["questions"].annotation == list[str]
            return {"questions": texts if plain else [{"text": t} for t in texts]}
        if "rules" in fields:
            return {"rules": [f"Rule {next(ids)}." for _ in range(5)]}
        if "rule" in fields:
            return {"reasoning": "r", "rule": f"IF fact {next(ids)} THEN good"}
        return "Instructions."

    return FakeLLM(generate=write)


def _judge(sample: str, question: Question) -> float:
    return 0.97 if "good" in sample else 0.1
