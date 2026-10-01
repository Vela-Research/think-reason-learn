"""Async client for Jev, Typesafe's System One model.

One request carries one sample (the ``state``) and up to
``MAX_QUESTIONS_PER_REQUEST`` typed questions about it. Labels are never part of
a request: callers send sample text and questions only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Sequence, Tuple

import httpx

from think_reason_learn.core._config import settings
from think_reason_learn.core.exceptions import (
    JevAuthError,
    JevCostCapError,
    MissingAPIKeyError,
)

from .schemas import AnswerValue, ChoiceQuestion, JevQuestion, NoulQuestion
from .schemas import first_jev_choice, is_jev

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
USD_PER_INPUT_TOKEN = 42 / 1e9
"""Typesafe list price for Jev input tokens (checked 22 September 2026).

Output tokens are reported but not priced at this rate.
"""
MAX_QUESTIONS_PER_REQUEST = 50
REQUEST_OVERHEAD_TOKENS = 250
BYTES_PER_TOKEN_ESTIMATE = 4.2
"""Estimate: 250 input tokens per request plus request bytes / 4.2, with each
non-ASCII character counted as one token.

Calibrated on billed usage (jev-1.13.0, 30 September 2026): 2,194 ECHR requests
used 188 + bytes / 4.42 tokens on average, and a 410-byte request used 342. The
estimate runs about 9% above the ECHR total and covers small requests.
"""

_RETRY_STATUSES = {429, 500, 502, 503, 504, 529}
_AUTH_STATUSES = {401, 403}
_MAX_WAIT_SECONDS = 120.0


def default_cache_dir() -> Path:
    """Return the folder that holds reusable Jev answers."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "think_reason_learn" / "jev"


@dataclass(slots=True)
class JevRequest:
    """One sample (the state) and the questions to ask about it."""

    state: str
    questions: Dict[str, JevQuestion]


@dataclass(slots=True)
class JevResult:
    """Answers for one ``JevRequest``; a failed or invalid answer is ``None``."""

    answers: Dict[str, AnswerValue | None]
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    served_model: str | None = None
    error: str | None = None


class JevBudget:
    """Cost cap shared by every Jev request in one run.

    Args:
        max_cost_usd: The most a run may spend, in US dollars.
    """

    def __init__(self, max_cost_usd: float) -> None:
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0
        self._reserved_usd = 0.0

    @property
    def remaining_usd(self) -> float:
        """Dollars left under the cap, net of requests in flight."""
        return self.max_cost_usd - self.spent_usd - self._reserved_usd

    def _reserve(self, usd: float) -> None:
        if usd > self.remaining_usd:
            raise JevCostCapError(
                f"The next Jev request (about ${usd:.4f}) would take spend past "
                f"the cap of ${self.max_cost_usd:.2f} for this run "
                f"(max_cost_usd; ${self.spent_usd:.4f} spent), so the run "
                f"stopped. {self._resume_hint()}"
            )
        self._reserved_usd += usd

    def _settle(self, reserved_usd: float, actual_usd: float) -> None:
        self._reserved_usd -= reserved_usd
        self.spent_usd += actual_usd

    def _check(self) -> None:
        if self.spent_usd > self.max_cost_usd:
            raise JevCostCapError(self._passed_message())

    def _passed_message(self) -> str:
        return (
            f"Jev spend ${self.spent_usd:.4f} passed the cap of "
            f"${self.max_cost_usd:.2f} for this run (max_cost_usd), so the run "
            f"stopped. {self._resume_hint()}"
        )

    @staticmethod
    def _resume_hint() -> str:
        return (
            "Answers received so far are cached: rerun with a higher cap, e.g. "
            "JevChoice(max_cost_usd=...), to continue without paying for them "
            "again."
        )


@dataclass(slots=True)
class _Chunk:
    request_index: int
    questions: Dict[str, JevQuestion]
    body: bytes
    key: str

    @property
    def estimated_tokens(self) -> int:
        # Non-Latin scripts take about a token per character or more, which
        # bytes / 4.2 would under-count; count those characters one each.
        non_ascii = [ch for ch in self.body.decode() if ord(ch) > 127]
        ascii_bytes = len(self.body) - len("".join(non_ascii).encode())
        return math.ceil(
            REQUEST_OVERHEAD_TOKENS
            + len(non_ascii)
            + ascii_bytes / BYTES_PER_TOKEN_ESTIMATE
        )


def _wire_question(question: JevQuestion) -> Dict[str, Any]:
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": {label: None for label in question.labels},
        }
    return {"type": "noul", "instructions": question.instructions}


def _parse_answer(question: JevQuestion, raw: Any) -> AnswerValue | None:
    if not isinstance(raw, dict):
        return None
    if isinstance(question, NoulQuestion):
        value = raw.get("noul")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        ):
            return float(value)
        return None
    value = raw.get("choice")
    return value if isinstance(value, str) and value in question.labels else None


def _parse_answers(
    questions: Dict[str, JevQuestion], raw_answers: Any
) -> Dict[str, AnswerValue | None]:
    raw_answers = raw_answers if isinstance(raw_answers, dict) else {}
    return {
        qid: _parse_answer(question, raw_answers.get(qid))
        for qid, question in questions.items()
    }


def _usage_tokens(payload: Dict[str, Any], key: str) -> int | None:
    usage = payload.get("usage")
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and value >= 0 else None


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class JevClient:
    """Concurrent, retrying, cost-capped client for Jev.

    Args:
        api_key: The ``TYPESAFE_API_KEY``.
        endpoint: The System One endpoint.
        concurrency: Requests in flight at once.
        timeout: Seconds before a request times out (then it is retried).
        max_attempts: Attempts per request for rate limits, server errors and
            timeouts. Other client errors (e.g. a sample Jev rejects as too long)
            fail at once; samples are never truncated.
        cache_dir: Folder of reusable answers. Defaults to ``default_cache_dir()``.
        transport: httpx transport, for tests.
        sleep: Coroutine used to wait between retries, for tests.
    """

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = ENDPOINT,
        concurrency: int = 20,
        timeout: float = 120.0,
        max_attempts: int = 6,
        cache_dir: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.cache_dir = cache_dir or default_cache_dir()
        self._transport = transport
        self._sleep = sleep

    # -- chunks and cache ----------------------------------------------------

    def _chunks(self, requests: Sequence[JevRequest], model: str) -> List[_Chunk]:
        chunks: List[_Chunk] = []
        for index, request in enumerate(requests):
            items = list(request.questions.items())
            for start in range(0, len(items), MAX_QUESTIONS_PER_REQUEST):
                questions = dict(items[start : start + MAX_QUESTIONS_PER_REQUEST])
                payload = {
                    "model": model,
                    "state": request.state,
                    "questions": {
                        qid: _wire_question(q) for qid, q in questions.items()
                    },
                }
                canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
                chunks.append(
                    _Chunk(
                        request_index=index,
                        questions=questions,
                        body=json.dumps(payload, ensure_ascii=False).encode(),
                        key=hashlib.sha256(canonical.encode()).hexdigest(),
                    )
                )
        return chunks

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, chunk: _Chunk) -> Dict[str, Any] | None:
        path = self._cache_path(chunk.key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        answers = _parse_answers(chunk.questions, payload.get("answers"))
        if any(v is None for v in answers.values()):
            return None
        return payload

    def _write_cache(self, chunk: _Chunk, payload: Dict[str, Any]) -> None:
        path = self._cache_path(chunk.key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.warning("Could not write the Jev cache at %s", path, exc_info=True)

    # -- sending ---------------------------------------------------------------

    async def _post(
        self, client: httpx.AsyncClient, chunk: _Chunk
    ) -> Tuple[Dict[str, Any] | None, str | None]:
        """POST one chunk with retries. Returns (payload, error)."""
        delay, error = 1.0, None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, self.max_attempts + 1):
            wait = delay
            try:
                response = await client.post(
                    self.endpoint, content=chunk.body, headers=headers
                )
            except httpx.RequestError as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status == 200:
                    try:
                        payload = response.json()
                        if not isinstance(payload, dict):
                            raise ValueError("response is not a JSON object")
                        return payload, None
                    except ValueError as exc:
                        error = f"Invalid response: {exc}"
                elif status in _AUTH_STATUSES:
                    raise JevAuthError(
                        f"Typesafe rejected TYPESAFE_API_KEY (HTTP {status}: "
                        f"{response.text[:200]}). Check the key in your "
                        "environment or .env file."
                    )
                elif status in _RETRY_STATUSES:
                    error = f"HTTP {status}: {response.text[:300]}"
                    retry_after = _retry_after_seconds(
                        response.headers.get("Retry-After")
                    )
                    if retry_after is not None:
                        wait = max(wait, retry_after)
                else:
                    return None, f"HTTP {status}: {response.text[:300]}"
            if attempt < self.max_attempts:
                logger.info(
                    "Jev request failed (%s); retry %d in %.1fs", error, attempt, wait
                )
                await self._sleep(min(wait, _MAX_WAIT_SECONDS))
                delay = min(delay * 2, 60.0)
        return None, error

    async def answer_many(
        self,
        requests: Sequence[JevRequest],
        *,
        model: str,
        budget: JevBudget,
        use_cache: bool = True,
    ) -> List[JevResult]:
        """Answer every request, printing the estimated cost before spending.

        Args:
            requests: One request per sample.
            model: The Jev model, e.g. ``"jev-latest"``.
            budget: The run's cost cap, shared across calls in the same run.
            use_cache: Reuse and store answers in ``cache_dir``.

        Returns:
            One ``JevResult`` per request, in order.

        Raises:
            JevCostCapError: The estimate exceeds what is left under the cap
                (nothing is sent), or actual spend passes the cap mid-run.
            JevAuthError: Typesafe rejected the API key.
        """
        chunks = self._chunks(requests, model)
        results = [JevResult(answers={}, cached=True) for _ in requests]
        pending: List[_Chunk] = []
        for chunk in chunks:
            result = results[chunk.request_index]
            cached = self._read_cache(chunk) if use_cache else None
            if cached is None:
                pending.append(chunk)
                result.answers.update({qid: None for qid in chunk.questions})
                result.cached = False
                continue
            result.answers.update(_parse_answers(chunk.questions, cached["answers"]))
            result.served_model = result.served_model or cached.get("model")

        est_tokens = sum(c.estimated_tokens for c in pending)
        est_usd = est_tokens * USD_PER_INPUT_TOKEN
        message = (
            f"Jev ({model}): sending {len(pending)} requests "
            f"({len(chunks) - len(pending)} answered from cache), "
            f"estimated cost ${est_usd:.4f}; cap ${budget.max_cost_usd:.2f} per "
            f"run, ${budget.spent_usd:.4f} spent so far."
        )
        print(message, file=sys.stderr, flush=True)
        logger.info(message)
        if est_usd > budget.remaining_usd:
            needed = budget.spent_usd + est_usd
            raise JevCostCapError(
                f"Estimated Jev cost ${est_usd:.4f} for {len(pending)} requests "
                f"exceeds the ${max(budget.remaining_usd, 0):.4f} left under the "
                f"cap of ${budget.max_cost_usd:.2f} for this run. Nothing was "
                "sent. To allow it, raise the cap, e.g. "
                f"JevChoice(max_cost_usd={math.ceil(needed * 1.2 + 0.5)})."
            )
        if not pending:
            return results

        semaphore = asyncio.Semaphore(self.concurrency)

        async def send(client: httpx.AsyncClient, chunk: _Chunk) -> None:
            async with semaphore:
                reserved = chunk.estimated_tokens * USD_PER_INPUT_TOKEN
                budget._reserve(reserved)
                try:
                    payload, error = await self._post(client, chunk)
                except BaseException:
                    budget._settle(reserved, 0.0)
                    raise
                result = results[chunk.request_index]
                if payload is None:
                    budget._settle(reserved, 0.0)
                    result.error = error
                    logger.warning("Jev request failed: %s", error)
                    return
                input_tokens = _usage_tokens(payload, "input_tokens")
                output_tokens = _usage_tokens(payload, "output_tokens") or 0
                if input_tokens is None:
                    input_tokens = chunk.estimated_tokens
                budget._settle(reserved, input_tokens * USD_PER_INPUT_TOKEN)
                answers = _parse_answers(chunk.questions, payload.get("answers"))
                result.answers.update(answers)
                result.input_tokens += input_tokens
                result.output_tokens += output_tokens
                result.served_model = payload.get("model") or result.served_model
                if use_cache and all(v is not None for v in answers.values()):
                    self._write_cache(chunk, payload)
                if any(v is None for v in answers.values()):
                    result.error = "Jev returned missing or invalid answers"
                budget._check()

        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self.timeout
            ) as client:
                async with asyncio.TaskGroup() as tg:
                    for chunk in pending:
                        tg.create_task(send(client, chunk))
        except BaseExceptionGroup as group:
            for exc in group.exceptions:
                if isinstance(exc, (JevCostCapError, JevAuthError)):
                    raise exc from None
            raise
        return results


def get_jev_client(api_key: str) -> JevClient | None:
    """Return a Jev client, or None when no TYPESAFE_API_KEY is set."""
    return JevClient(api_key) if api_key else None


def new_run_budget(llm_priority: Sequence[object]) -> JevBudget | None:
    """A fresh budget with the cap of the first Jev choice, if there is one."""
    choice = first_jev_choice(llm_priority)
    return JevBudget(choice.max_cost_usd) if choice is not None else None


_CHECK_KEYS: ContextVar[bool] = ContextVar("jev_check_keys", default=True)


@contextmanager
def loading_saved_model() -> Iterator[None]:
    """Skip ``require_typesafe_key`` while a saved model is rebuilt.

    A saved model can then be loaded and inspected without the key; answering
    with Jev still needs it.
    """
    token = _CHECK_KEYS.set(False)
    try:
        yield
    finally:
        _CHECK_KEYS.reset(token)


def require_typesafe_key(
    llm_priority: Sequence[object], *, method: str, param: str
) -> None:
    """Raise if Jev is among the answerers and TYPESAFE_API_KEY is not set.

    Args:
        llm_priority: The method's answering models.
        method: The class name shown in the message, e.g. ``"RRF"``.
        param: The argument that picks the answerers, e.g. ``"qanswer_llmc"``.

    Raises:
        ValueError: Jev is chosen but not first. A chat model ahead of Jev would
            answer on the chat path, which cannot hand questions on to Jev.
        MissingAPIKeyError: Jev is chosen and the key is missing.
    """
    if first_jev_choice(llm_priority) is None or not _CHECK_KEYS.get():
        return
    if not is_jev(llm_priority[0]):
        raise ValueError(
            f"{method}: put JevChoice first in {param}, or leave it out. Chat "
            "models can follow Jev as fallbacks, but cannot come before it."
        )
    if settings.TYPESAFE_API_KEY:
        return
    raise MissingAPIKeyError(
        f"{method} answers questions with Jev (Typesafe's System One model), "
        "which needs TYPESAFE_API_KEY, and TYPESAFE_API_KEY is not set. Set it "
        "in your environment or in a .env file, or answer with a chat model "
        f'instead: {method}(..., {param}=[OpenAIChoice(model="gpt-4.1-mini")]).'
    )
