"""Every test runs offline, with a fresh cache and fresh shared state."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx2
import pytest

from think_reason_learn.llm import _calls, _catalog, _config, _limiter, config_context
from think_reason_learn.llm._adapters import _http

type Handler = Callable[[httpx2.Request], httpx2.Response]

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
