"""Configuration for tests."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pytest  # noqa: E402

from think_reason_learn.core._config import settings  # noqa: E402


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
