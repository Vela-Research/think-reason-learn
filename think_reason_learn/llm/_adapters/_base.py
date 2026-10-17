"""What every provider adapter implements, and how provider failures are named."""

from __future__ import annotations

import inspect
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel

from .._usage import TokenCounts

if TYPE_CHECKING:
    from .._questions import Answer, Question
    from .._spec import Endpoint

type Failure = Literal[
    "rate_limit",
    "transient",
    "auth",
    "quota",
    "unavailable",
    "bad_request",
    "rejected_param",
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
    headers: Mapping[str, str] = field(default_factory=dict[str, str])


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


class DecisionAdapter:
    """What every decision-model adapter shares: many questions per call, no writing."""

    max_questions_per_call: int = 50

    def max_questions(self, endpoint: Endpoint) -> int:
        """Questions about one sample sent in one call."""
        return self.max_questions_per_call

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        """Decision models do not write; roles that write reject them earlier."""
        raise ProviderError("bad_request", f"{endpoint.name} cannot write text.")


async def read_reply[T, R](
    parse: Callable[[], object], kind: type[T], read: Callable[[T], R]
) -> R:
    """``read`` applied to a reply's body, parsed as ``kind``.

    An unreadable body (a proxy's HTML page, cut-off JSON, a missing field) is a
    transient failure, never a crash. SDKs build replies without checking them,
    so a missing field only shows when ``read`` touches it.
    """
    try:
        body = parse()
        if inspect.isawaitable(body):
            body = await body
        if not isinstance(body, kind):
            raise TypeError(f"not a {kind.__name__}")
        return read(body)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise ProviderError("transient", f"Unreadable reply: {exc}") from exc


def classify(
    status: int | None, message: str, headers: Mapping[str, str]
) -> ProviderError:
    """Name a failed call by its HTTP status (``None``: no response) and message.

    The message is cut to ``_MESSAGE_CHARS``: a proxy's error page can be long,
    and every failed call keeps it in usage.
    """
    retry_after = retry_after_seconds(headers)
    lowered = message.lower()
    if len(message) > _MESSAGE_CHARS:
        message = f"{message[:_MESSAGE_CHARS]}..."
    if status is None or status in _TRANSIENT:
        return ProviderError("transient", message, retry_after=retry_after)
    if _mentions(lowered, _FLAGGED_WORDS):  # one sample's problem, on a 403
        return ProviderError("bad_request", message)
    if status == 404 or _mentions(lowered, _NO_MODEL_WORDS):
        return ProviderError("unavailable", message)
    if status in (401, 403) or _mentions(lowered, _AUTH_WORDS):
        return ProviderError("auth", message)
    if status == 402 or _mentions(lowered, _QUOTA_WORDS):
        return ProviderError("quota", message)
    if status == 429:
        return ProviderError("rate_limit", message, retry_after=retry_after)
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


_MESSAGE_CHARS: Final = 500
_TRANSIENT: Final = frozenset({408, 409, 500, 502, 503, 504, 520, 522, 524, 529})
_AUTH_WORDS: Final = (
    "valid api key",
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
)
_FLAGGED_WORDS: Final = ("flagged", "moderation")
"""An input a moderation check refused, as OpenRouter's 403 says."""
_NO_MODEL_WORDS: Final = ("model_not_found", "does not have access to model")
"""A model this account cannot use: try the next model, not a fatal key error."""
_QUOTA_WORDS: Final = (
    "insufficient_quota",
    "insufficient credits",
    "credit balance",
    "spend limit",
)
_PARAM_RE: Final = re.compile("|".join([*OPTIONAL_PARAMS, "effort"]))
_REJECTION_WORDS: Final = (
    "unsupported",
    "not supported",
    "does not support",
    "unrecognized",
    "unknown",
    "not allowed",
    "extra inputs",
    "only the default",
)


def _rejected_param(message: str) -> str | None:
    """The setting a 400 says the model does not support, if it says so."""
    if not _mentions(message, _REJECTION_WORDS):
        return None
    match = _PARAM_RE.search(message)
    if match is None:
        return None
    return "reasoning_effort" if match.group(0) == "effort" else match.group(0)


def _mentions(message: str, words: tuple[str, ...]) -> bool:
    return any(word in message for word in words)


def _is_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False
