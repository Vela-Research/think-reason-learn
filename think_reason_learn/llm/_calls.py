"""Send one call under the endpoint's limits, retrying what is worth retrying."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Final

from ._adapters._base import ProviderError, Reply
from ._limiter import limiter_for
from ._spec import Endpoint

logger = logging.getLogger(__name__)

_MAX_WAIT: Final = 120.0
_RETRIED: Final = frozenset({"rate_limit", "transient"})


async def send[T](
    endpoint: Endpoint,
    call: Callable[[frozenset[str]], Awaitable[Reply[T]]],
    *,
    estimated_tokens: int,
    max_retries: int,
) -> Reply[T]:
    """Run ``call`` under the endpoint's limiter.

    Rate limits, server errors and timeouts are retried with backoff, waiting at
    least what the provider asked. A setting the model rejects is dropped for
    this endpoint and the call resent at once.

    Raises:
        ProviderError: The call failed and retrying cannot help.
    """
    limiter = limiter_for(endpoint)
    retries = 0
    while True:
        dropped = _dropped.get(endpoint.key, frozenset())
        await limiter.acquire(estimated_tokens)
        try:
            reply = await call(dropped)
        except ProviderError as error:
            await limiter.release(
                estimated=estimated_tokens,
                used=None,
                headers={},
                rate_limited=error.failure == "rate_limit",
                retry_after=error.retry_after,
            )
            if _drop(endpoint, error, dropped):
                continue
            if error.failure not in _RETRIED or retries >= max_retries:
                raise
            retries += 1
            await asyncio.sleep(_backoff(retries, error.retry_after))
            continue
        except BaseException:
            await limiter.release(
                estimated=estimated_tokens,
                used=None,
                headers={},
                rate_limited=False,
                retry_after=None,
            )
            raise
        await limiter.release(
            estimated=estimated_tokens,
            used=reply.tokens.input + reply.tokens.output,
            headers=reply.headers,
            rate_limited=False,
            retry_after=None,
        )
        return reply


_dropped: Final[dict[str, frozenset[str]]] = {}
"""Settings each endpoint rejected, left out of its later calls."""


def _drop(endpoint: Endpoint, error: ProviderError, dropped: frozenset[str]) -> bool:
    if (
        error.failure != "rejected_param"
        or error.param is None
        or error.param in dropped
    ):
        return False
    _dropped[endpoint.key] = dropped | {error.param}
    logger.warning(
        f"{endpoint.name} rejected {error.param}; calling it without from now on."
    )
    return True


def _backoff(retry: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, _MAX_WAIT)
    return min(0.5 * 2 ** (retry - 1), 30.0) * random.uniform(0.5, 1.0)
