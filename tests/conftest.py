"""Configuration for tests."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from collections.abc import Callable, Iterator  # noqa: E402

import httpx2  # noqa: E402
import pytest  # noqa: E402

from think_reason_learn.llm import _calls, _config, _limiter, config_context  # noqa: E402
from think_reason_learn.llm._adapters import _http, anthropic, openai_compat  # noqa: E402

from think_reason_learn.core._config import settings  # noqa: E402

type Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture
def typesafe_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A placeholder TYPESAFE_API_KEY so Jev answerers can be constructed offline."""
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def offline_jev(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests never reach Typesafe, even with a real TYPESAFE_API_KEY in .env."""
    import httpx

    from think_reason_learn.core.llms import llm
    from think_reason_learn.core.llms._jev.client import JevClient

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(599, text="tests must not call Typesafe")

    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setattr(
        llm,
        "jev_llm",
        JevClient(
            api_key="offline",
            transport=httpx.MockTransport(refuse),
            cache_dir=tmp_path / "jev-cache",
            sleep=no_wait,
        ),
    )


def _reset() -> None:
    _limiter._limiters.clear()
    _config._limits.clear()
    _calls._dropped.clear()
    openai_compat._clients.clear()
    anthropic._clients.clear()
    _http.set_test_transport(None)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path) -> Iterator[None]:
    """A cache in a temporary folder, two retries, and no shared state."""
    _reset()
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
