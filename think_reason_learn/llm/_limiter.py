"""Shared speed limits per model and account: concurrency, requests and tokens.

One ``Limiter`` per endpoint key serves every estimator in the process. It caps
calls in flight, spends request and token buckets when ``rpm``/``tpm`` are set
or learned from the provider's headers, pauses everyone when the provider asks,
and adapts concurrency: halved on a rate limit, raised by one per success.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

from ._config import Limits, limits_for
from ._loop import on_fork
from ._spec import Endpoint

_START: Final = 8
_HOSTED_CEILING: Final = 64
_LOCAL_CEILING: Final = 4
_BURST_SECONDS: Final = 10.0
_MAX_PAUSE: Final = 120.0

logger = logging.getLogger(__name__)

_limiters: Final[dict[str, Limiter]] = {}
on_fork(_limiters.clear)


type Outcome = Literal["ok", "rate_limited", "failed"]
"""How a call ended: answered, rate limited, or failed otherwise."""


class Limiter:
    """Admission control for one endpoint on one account."""

    def __init__(self, limits: Limits, *, local: bool) -> None:
        """Start below the ceiling, with buckets for any limits set."""
        self._cond = asyncio.Condition()
        self._in_flight = 0
        self._paused_until = 0.0
        self._last_cut = -math.inf
        self._requests: _Bucket | None = None
        self._tokens: _Bucket | None = None
        self._set_rpm = self._set_tpm = False
        self._allowed = _START
        self.configure(limits, local=local)

    def configure(self, limits: Limits, *, local: bool) -> None:
        """Apply limits set with ``set_limits``; an unchanged rate keeps its bucket."""
        self._ceiling = limits.max_concurrency or (
            _LOCAL_CEILING if local else _HOSTED_CEILING
        )
        self._allowed = min(self._allowed, self._ceiling)
        if limits.rpm is not None:
            self._learn("requests", limits.rpm)
        elif self._set_rpm:  # lifted
            self._requests = None
        if limits.tpm is not None:
            self._learn("tokens", limits.tpm)
        elif self._set_tpm:
            self._tokens = None
        self._set_rpm, self._set_tpm = limits.rpm is not None, limits.tpm is not None

    async def acquire(
        self, tokens: int, stopped: Callable[[], bool] = lambda: False
    ) -> bool:
        """Wait for a slot, a request and ``tokens`` under the limits.

        Returns ``False``, having taken nothing, if ``stopped()`` turns true first.
        """
        async with self._cond:
            while (delay := self._delay(tokens)) > 0 and not stopped():
                timeout = None if math.isinf(delay) else delay
                with contextlib.suppress(TimeoutError):  # time to check again
                    await asyncio.wait_for(self._cond.wait(), timeout)
            if stopped():
                self._cond.notify()  # pass the wake-up on to the next waiter
                return False
            self._in_flight += 1
            for bucket, amount in ((self._requests, 1), (self._tokens, tokens)):
                if bucket is not None:
                    bucket.take(amount)
            return True

    async def release(
        self,
        *,
        estimated: int,
        used: int | None,
        headers: Mapping[str, str],
        outcome: Outcome,
        retry_after: float | None = None,
    ) -> None:
        """Return the slot, correct token spend and adapt to the outcome.

        A rate limit halves concurrency (at most once a second, so one burst
        of 429s does not collapse it) and pauses everyone; a success raises it
        by one; other failures leave it alone.
        """
        async with self._cond:
            self._in_flight -= 1
            if self._tokens is not None and used is not None:
                self._tokens.take(used - estimated)
            paused = self._observe(headers)
            if outcome == "rate_limited":
                now = time.monotonic()
                if now - self._last_cut >= 1.0:
                    self._allowed, self._last_cut = max(1, self._allowed // 2), now
                paused = self._pause(retry_after if retry_after is not None else 1.0)
            elif outcome == "ok":
                self._allowed = min(self._ceiling, self._allowed + 1)
            if paused:
                self._cond.notify_all()
            else:
                self._cond.notify(2)  # the freed slot, and one more if allowed grew

    def _delay(self, tokens: int) -> float:
        now = time.monotonic()
        if self._paused_until > now:
            return self._paused_until - now
        if self._in_flight >= self._allowed:
            return math.inf
        waits = [
            bucket.wait(amount, now)
            for bucket, amount in ((self._requests, 1), (self._tokens, tokens))
            if bucket is not None
        ]
        return max(waits, default=0.0)

    def _pause(self, seconds: float) -> bool:
        """Hold every call for ``seconds`` (at most ``_MAX_PAUSE``)."""
        seconds = min(seconds, _MAX_PAUSE)
        if seconds > 10:
            logger.warning(
                f"The provider asked to wait; pausing calls for {seconds:.0f}s."
            )
        self._paused_until = max(self._paused_until, time.monotonic() + seconds)
        return True

    def _observe(self, headers: Mapping[str, str]) -> bool:
        """Learn limits the user did not set; pause when a window is spent."""
        lowered = {k.lower(): v for k, v in headers.items()}
        paused = False
        for unit in ("requests", "tokens"):
            limit, remaining, reset = _rate_headers(lowered, unit)
            if (
                limit is not None
                and 1 <= limit < math.inf  # below 1 a minute is no real limit
                and not (self._set_rpm if unit == "requests" else self._set_tpm)
            ):
                self._learn(unit, limit)
            if remaining == 0 and reset is not None:
                paused = self._pause(reset)
        return paused

    def _learn(self, unit: str, per_minute: float) -> None:
        if per_minute <= 0:
            return
        current = self._requests if unit == "requests" else self._tokens
        if current is not None and math.isclose(current.per_second * 60, per_minute):
            return
        bucket = _Bucket.per_minute(per_minute)
        if unit == "requests":
            self._requests = bucket
        else:
            self._tokens = bucket


def limiter_for(endpoint: Endpoint) -> Limiter:
    """The shared limiter of ``endpoint``'s model and account, with current limits."""
    limiter = _limiters.get(endpoint.key)
    if limiter is None:
        limiter = _limiters[endpoint.key] = Limiter(
            limits_for(endpoint), local=endpoint.local
        )
    return limiter


def apply_limits(endpoint: Endpoint) -> None:
    """Re-read ``set_limits`` for ``endpoint`` (at the start of each run)."""
    limiter_for(endpoint).configure(limits_for(endpoint), local=endpoint.local)


_DURATION_RE: Final = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS: Final = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _rate_headers(
    headers: Mapping[str, str], unit: str
) -> tuple[float | None, float | None, float | None]:
    """(limit per minute, remaining, seconds to reset), OpenAI or Anthropic style."""
    openai = (
        f"x-ratelimit-limit-{unit}",
        f"x-ratelimit-remaining-{unit}",
        f"x-ratelimit-reset-{unit}",
    )
    anthropic = tuple(
        f"anthropic-ratelimit-{unit}-{part}" for part in ("limit", "remaining", "reset")
    )
    for limit_key, remaining_key, reset_key in (openai, anthropic):
        if remaining_key in headers:
            return (
                _number(headers.get(limit_key)),
                _number(headers.get(remaining_key)),
                _reset_seconds(headers.get(reset_key)),
            )
    return None, None, None


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _reset_seconds(value: str | None) -> float | None:
    """``"6m0s"``/``"20ms"`` (OpenAI) or an RFC 3339 time (Anthropic) to seconds."""
    if not value:
        return None
    if parts := _DURATION_RE.findall(value):
        return sum(float(amount) * _UNIT_SECONDS[unit] for amount, unit in parts)
    try:
        return max(datetime.fromisoformat(value).timestamp() - time.time(), 0.0)
    except ValueError:
        return None


@dataclass(slots=True)
class _Bucket:
    per_second: float
    capacity: float
    level: float
    stamp: float

    @classmethod
    def per_minute(cls, amount: float) -> _Bucket:
        capacity = max(1.0, amount * _BURST_SECONDS / 60)
        return cls(amount / 60, capacity, capacity, time.monotonic())

    def wait(self, amount: float, now: float) -> float:
        """Seconds until ``amount`` can be taken (more than fits: a full bucket)."""
        self.level = min(
            self.capacity, self.level + (now - self.stamp) * self.per_second
        )
        self.stamp = now
        missing = min(amount, self.capacity) - self.level
        return max(missing, 0.0) / self.per_second

    def take(self, amount: float) -> None:
        self.level -= amount
