"""What every provider adapter implements, and how provider failures are named."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel

from .._usage import TokenCounts

if TYPE_CHECKING:
    from .._questions import Answer, Question
    from .._spec import Endpoint

type Failure = Literal[
    "rate_limit", "transient", "auth", "quota", "bad_request", "rejected_param"
]
"""Why a call failed: retried (rate_limit, transient), fatal to the run (auth,
quota), tried on the next model (bad_request), or resent without a setting
(rejected_param)."""

OPTIONAL_PARAMS: Final = (
    "temperature",
    "reasoning_effort",
    "service_tier",
    "prompt_cache_key",
    "max_completion_tokens",
)
"""Settings a model may reject; the call is then resent without them."""


class ProviderError(Exception):
    """A provider call failed, named by what to do about it."""

    def __init__(
        self,
        failure: Failure,
        message: str,
        *,
        retry_after: float | None = None,
        param: str | None = None,
    ) -> None:
        """Name the failure, with the wait the provider asked for, if any."""
        super().__init__(message)
        self.failure: Failure = failure
        self.retry_after = retry_after
        self.param = param


@dataclass(frozen=True, slots=True)
class Reply[T]:
    """A provider's reply: the output, its tokens and the rate-limit headers."""

    output: T
    tokens: TokenCounts
    served_model: str | None = None
    provider_cost: float | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """A request for text or a structured object.

    ``prefix`` is sent first and marked for the provider's prompt cache: the
    sample, when one sample gets many prompts.
    """

    prompt: str
    instructions: str | None = None
    schema: type[BaseModel] | None = None
    prefix: str | None = None


type Answers = dict[str, Answer | None]


class Adapter(Protocol):
    """Calls one wire protocol. ``dropped`` names settings the model rejected."""

    def max_questions(self, endpoint: Endpoint) -> int:
        """Questions about one sample sent in one call."""
        ...

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        """Text, or an instance of ``request.schema``; ``None`` if unusable."""
        ...

    async def answer(
        self,
        endpoint: Endpoint,
        sample: str,
        questions: Mapping[str, Question],
        dropped: frozenset[str],
    ) -> Reply[Answers]:
        """An answer per question id; ``None`` where a question went unanswered."""
        ...


def classify(
    status: int | None, message: str, headers: Mapping[str, str]
) -> ProviderError:
    """Name a failed call by its HTTP status (``None``: no response)."""
    retry_after = retry_after_seconds(headers)
    lowered = message.lower()
    if status is None or status in _TRANSIENT:
        return ProviderError("transient", message, retry_after=retry_after)
    if status in (401, 403):
        return ProviderError("auth", message)
    if status == 429:
        quota = any(word in lowered for word in _QUOTA_WORDS)
        return ProviderError(
            "quota" if quota else "rate_limit", message, retry_after=retry_after
        )
    if status in (400, 422) and (param := _rejected_param(lowered)) is not None:
        return ProviderError("rejected_param", message, param=param)
    return ProviderError("bad_request", message)


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Seconds the provider asked to wait, from ``retry-after(-ms)``."""
    lowered = {k.lower(): v for k, v in headers.items()}
    if (millis := lowered.get("retry-after-ms")) and _is_number(millis):
        return float(millis) / 1000
    value = lowered.get("retry-after")
    if not value:
        return None
    if _is_number(value):
        return max(float(value), 0.0)
    try:
        return max(parsedate_to_datetime(value).timestamp() - time.time(), 0.0)
    except (TypeError, ValueError):
        return None


_TRANSIENT: Final = frozenset({408, 409, 500, 502, 503, 504, 520, 522, 524, 529})
_QUOTA_WORDS: Final = ("insufficient_quota", "spend limit", "billing", "credit balance")
_PARAM_RE: Final = re.compile("|".join(OPTIONAL_PARAMS))


def _rejected_param(message: str) -> str | None:
    match = _PARAM_RE.search(message)
    return match.group(0) if match else None


def _is_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False
