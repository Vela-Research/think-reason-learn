"""Clients shared by the adapters, one per endpoint and account.

Clients are bound to the library's event loop, so they are dropped when a
forked child starts its own loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import httpx2

from .._loop import on_fork

_LIMITS: Final = httpx2.Limits(max_connections=256, max_keepalive_connections=64)
"""Keep enough connections open for the default concurrency ceiling."""

_transport: httpx2.AsyncBaseTransport | None = None
"""A transport for tests, used by clients created after it is set."""

_clients: Final[dict[str, object]] = {}


def set_test_transport(transport: httpx2.AsyncBaseTransport | None) -> None:
    """Route new clients through ``transport`` (tests only); ``None`` to stop."""
    global _transport
    _transport = transport
    _clients.clear()


def http_client(key: str) -> httpx2.AsyncClient:
    """The HTTP client for ``key`` (an endpoint's base URL and account)."""
    return cached(
        f"http|{key}", lambda: httpx2.AsyncClient(transport=_transport, limits=_LIMITS)
    )


def cached[C](key: str, build: Callable[[], C]) -> C:
    """The client stored under ``key``, built on first use."""
    if key not in _clients:
        _clients[key] = build()
    client: C = _clients[key]  # type: ignore[assignment]  # stored by this function with the same key
    return client


def forget_clients() -> None:
    """Drop every client (tests, and forked children)."""
    _clients.clear()


on_fork(forget_clients)
