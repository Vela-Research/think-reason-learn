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
