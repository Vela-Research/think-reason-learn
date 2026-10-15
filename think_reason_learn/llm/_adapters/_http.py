"""HTTP clients shared by the adapters, one per endpoint and account."""

from __future__ import annotations

from typing import Final

import httpx2

_transport: httpx2.AsyncBaseTransport | None = None
"""A transport for tests, used by clients created after it is set."""


def set_test_transport(transport: httpx2.AsyncBaseTransport | None) -> None:
    """Route new clients through ``transport`` (tests only); ``None`` to stop."""
    global _transport
    _transport = transport
    _clients.clear()


def http_client(key: str, timeout: float) -> httpx2.AsyncClient:
    """The client for ``key`` (an endpoint's base URL and account)."""
    if key not in _clients:
        _clients[key] = httpx2.AsyncClient(transport=_transport, timeout=timeout)
    return _clients[key]


_clients: Final[dict[str, httpx2.AsyncClient]] = {}
