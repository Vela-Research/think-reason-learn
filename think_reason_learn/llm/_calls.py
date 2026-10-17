"""Send one call under the endpoint's limits, retrying what is worth retrying."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable
from typing import Final

from ._adapters._base import ProviderError, Reply
from ._config import Config
from ._limiter import Limiter, Outcome, limiter_for
from ._loop import on_fork
from ._spec import Endpoint

logger = logging.getLogger(__name__)

_MAX_WAIT: Final = 120.0
_RETRIED: Final = frozenset({"rate_limit", "transient"})

_dropped: Final[dict[str, frozenset[str]]] = {}
"""Settings each endpoint rejected, left out of its later calls."""
on_fork(_dropped.clear)


class Skipped(ProviderError):
    """A call not sent, because its pass stopped while it waited for a slot."""

    def __init__(self) -> None:
        """A skipped call; nothing was sent or spent."""
        super().__init__("unavailable", "Skipped: this model's pass stopped.")


async def send[T](
    endpoint: Endpoint,
    call: Callable[[frozenset[str]], Awaitable[Reply[T]]],
    *,
    estimated_tokens: int,
    config: Config,
    stopped: Callable[[], bool] = lambda: False,
) -> Reply[T]:
    """Run ``call`` under the endpoint's limiter.

    Rate limits, server errors and timeouts are retried with backoff, waiting at
    least what the provider asked: at most ``config.max_retries`` times, and only
    while a retry can finish within ``config.max_retry_time`` of the first
    failure. A setting the model rejects is dropped for this endpoint and the
    call resent at once.

    Raises:
        ProviderError: The call failed and retrying cannot help.
        Skipped: ``stopped()`` turned true before the call was sent.
    """
    limiter = limiter_for(endpoint)
    retries, deadline = 0, math.inf
    while True:
        dropped = _dropped.get(endpoint.key, frozenset())
        try:
            return await _attempt(limiter, call, dropped, estimated_tokens, stopped)
        except ProviderError as error:
            if _drop(endpoint, error, dropped):
                continue
            if deadline == math.inf:
                deadline = time.monotonic() + config.max_retry_time
            wait = _backoff(retries + 1, error.retry_after)
            if (
                error.failure not in _RETRIED
                or retries >= config.max_retries
                or time.monotonic() + wait + config.timeout > deadline
            ):
                raise
            retries += 1
            await asyncio.sleep(wait)


def _drop(endpoint: Endpoint, error: ProviderError, dropped: frozenset[str]) -> bool:
    if (
        error.failure != "rejected_param"
        or error.param is None
        or error.param in dropped
    ):
        return False
    _dropped[endpoint.key] = _dropped.get(endpoint.key, frozenset()) | {error.param}
    logger.warning(
        f"{endpoint.name} rejected {error.param}; calling it without from now on."
    )
    return True


def _backoff(retry: int, retry_after: float | None) -> float:
    """At least what the provider asked, and never no wait at all."""
    backoff = min(0.5 * 2 ** (retry - 1), 30.0) * random.uniform(0.5, 1.0)
    return min(max(retry_after or 0.0, backoff), _MAX_WAIT)


async def _attempt[T](
    limiter: Limiter,
    call: Callable[[frozenset[str]], Awaitable[Reply[T]]],
    dropped: frozenset[str],
    estimated: int,
    stopped: Callable[[], bool],
) -> Reply[T]:
    """One call in one limiter slot; the slot is returned whatever happens."""
    if not await limiter.acquire(estimated, stopped):
        raise Skipped
    outcome: Outcome = "failed"
    used: int | None = None
    headers: dict[str, str] = {}
    retry_after: float | None = None
    try:
        reply = await call(dropped)
        outcome, used = "ok", reply.tokens.input + reply.tokens.output
        headers = dict(reply.headers)
        return reply
    except ProviderError as error:
        outcome = "rate_limited" if error.failure == "rate_limit" else "failed"
        retry_after = error.retry_after
        raise
    finally:
        await limiter.release(
            estimated=estimated,
            used=used,
            headers=headers,
            outcome=outcome,
            retry_after=retry_after,
        )
