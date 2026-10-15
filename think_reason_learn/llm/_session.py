"""A run's model calls: fallbacks, the reply cache, the budget and usage, written once.

Estimators open one ``LLMSession`` per ``fit`` or ``predict`` and make every
call through it, so every model and provider gets the same retries, limits,
caching, cost accounting and fallbacks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeIs, overload

from pydantic import BaseModel, ValidationError

from think_reason_learn.core.exceptions import (
    BudgetExceededError,
    LLMError,
    ProviderAuthError,
)

from ._adapters import ADAPTERS, Answers, GenerateRequest, ProviderError, Reply
from ._cache import JSON, Cache, cache_key, open_cache
from ._calls import send
from ._config import get_config
from ._limiter import apply_limits
from ._questions import Answer, Question
from ._spec import Endpoint
from ._usage import Budget, TokenCounts, Usage, UsageRecord, call_cost, format_usd

logger = logging.getLogger(__name__)

_WINDOW: Final = 256
"""Answer requests scheduled at once; the limiter decides how many are sent."""
_GENERATE_OUTPUT_ESTIMATE: Final = 2000
_ANSWER_OUTPUT_ESTIMATE: Final = 8


class LLMSession:
    """Every model call of one run.

    The settings in effect when it is created (``get_config()``) apply to all
    its calls, wherever they run.

    Args:
        usage: Where calls are recorded; a new ``Usage`` by default.
    """

    def __init__(self, *, usage: Usage | None = None) -> None:
        """Start a run under the current settings."""
        self.config = get_config()
        self.usage = usage if usage is not None else Usage()
        self.budget = Budget(self.config.max_cost_usd)
        self._cache: Cache | None = (
            open_cache(self.config.cache_path) if self.config.cache else None
        )
        self._configured: set[str] = set()

    @overload
    async def generate[S: BaseModel](
        self,
        models: Sequence[Endpoint],
        prompt: str,
        *,
        role: str,
        schema: type[S],
        instructions: str | None = None,
        prefix: str | None = None,
        draw: int = 0,
    ) -> S: ...

    @overload
    async def generate(
        self,
        models: Sequence[Endpoint],
        prompt: str,
        *,
        role: str,
        schema: None = None,
        instructions: str | None = None,
        prefix: str | None = None,
        draw: int = 0,
    ) -> str: ...

    async def generate(
        self,
        models: Sequence[Endpoint],
        prompt: str,
        *,
        role: str,
        schema: type[BaseModel] | None = None,
        instructions: str | None = None,
        prefix: str | None = None,
        draw: int = 0,
    ) -> BaseModel | str:
        """Text, or an instance of ``schema``, from the first model that gives it.

        Args:
            models: Models in fallback order.
            prompt: The prompt.
            role: Who is asking, recorded with usage, e.g. ``"generator"``.
            schema: A Pydantic model to fill instead of free text.
            instructions: System instructions.
            prefix: Text sent before the prompt and marked for prompt caching.
            draw: Different draws of one request are different calls; the same
                draw is answered from the cache.

        Raises:
            LLMError: No model gave usable output.
            ProviderAuthError: A provider rejected its key or has no quota.
            BudgetExceededError: The run's cost cap would be or was passed.
        """
        request = GenerateRequest(prompt, instructions, schema, prefix)
        errors: list[str] = []
        for endpoint in models:
            key = _generation_key(endpoint, request, draw)
            if (hit := self._cached_generation(key, schema)) is not None:
                self._record_hits(endpoint, role, 1)
                return hit
            adapter = ADAPTERS[endpoint.protocol]
            try:
                reply = await self._call(
                    endpoint,
                    role,
                    lambda dropped: adapter.generate(endpoint, request, dropped),
                    TokenCounts(
                        input=_tokens(prompt, instructions, prefix),
                        output=_GENERATE_OUTPUT_ESTIMATE,
                    ),
                )
            except ProviderError as error:
                errors.append(f"{endpoint.name}: {error}")
                continue
            if _usable(reply.output, schema):
                self._store({key: _dump(reply.output)})
                self.budget.check()
                return reply.output
            errors.append(f"{endpoint.name}: no usable output")
        raise LLMError(f"No model gave usable output for {role}: {'; '.join(errors)}")

    async def answer(
        self,
        models: Sequence[Endpoint],
        samples: Sequence[str],
        questions: Mapping[str, Question],
        *,
        role: str,
        draw: int = 0,
        progress: Callable[[int], None] | None = None,
    ) -> list[Answers]:
        """Answer every question about every sample.

        A question a model fails to answer goes to the next model.

        Args:
            models: Models in fallback order, reasoning or decision.
            samples: Sample texts.
            questions: Questions by id.
            role: Who is asking, recorded with usage, e.g. ``"answerer"``.
            draw: Different draws ask again; the same draw reuses answers.
            progress: Called with the number of answers each time some arrive.

        Returns:
            Per sample, the answer to each question id; ``None`` if unanswered.

        Raises:
            LLMError: Not one question was answered.
            ProviderAuthError: A provider rejected its key or has no quota.
            BudgetExceededError: The run's cost cap would be or was passed.
        """
        results: list[Answers] = [dict.fromkeys(questions) for _ in samples]
        errors: list[str] = []
        digests = [hashlib.sha256(sample.encode()).hexdigest() for sample in samples]
        for endpoint in models:
            pass_ = _AnswerPass(
                self, endpoint, samples, digests, questions, role, draw, progress
            )
            if not await pass_.run(results, errors):
                break
        if (
            samples
            and questions
            and all(a is None for r in results for a in r.values())
        ):
            first = errors[0] if errors else "no usable answers"
            raise LLMError(
                f"No answers from {[m.name for m in models]}. First error: {first}"
            )
        if errors:
            logger.warning(f"{len(errors)} answer calls failed; the first: {errors[0]}")
        return results

    async def _call[T](
        self,
        endpoint: Endpoint,
        role: str,
        call: Callable[[frozenset[str]], Awaitable[Reply[T]]],
        estimate: TokenCounts,
    ) -> Reply[T]:
        """One call under the budget, recorded in usage."""
        if endpoint.key not in self._configured:
            apply_limits(endpoint)
            self._configured.add(endpoint.key)
        held = self.budget.reserve(call_cost(endpoint, estimate, None)[0])
        start = time.perf_counter()
        try:
            reply = await send(
                endpoint,
                call,
                estimated_tokens=estimate.input,
                max_retries=self.config.max_retries,
            )
        except ProviderError as error:
            self.budget.settle(held, None)
            self._record(endpoint, role, TokenCounts(), start, error=str(error))
            if error.failure in ("auth", "quota"):
                raise ProviderAuthError(f"{endpoint.name}: {error}") from error
            raise
        except BaseException:
            self.budget.settle(held, None)
            raise
        cost = self._record(
            endpoint,
            role,
            reply.tokens,
            start,
            served=reply.served_model,
            provider_cost=reply.provider_cost,
        )
        self.budget.settle(held, cost)
        return reply

    def _record(
        self,
        endpoint: Endpoint,
        role: str,
        tokens: TokenCounts,
        start: float,
        *,
        served: str | None = None,
        provider_cost: float | None = None,
        error: str | None = None,
    ) -> float | None:
        cost, source = (
            (0.0, "none")
            if error is not None
            else call_cost(endpoint, tokens, provider_cost)
        )
        self.usage.add(
            UsageRecord(
                time=time.time(),
                role=role,
                model=endpoint.name,
                served_model=served,
                input_tokens=tokens.input,
                output_tokens=tokens.output,
                cache_read_tokens=tokens.cache_read,
                cache_write_tokens=tokens.cache_write,
                reasoning_tokens=tokens.reasoning,
                cost_usd=cost,
                cost_source=source,
                latency_s=time.perf_counter() - start,
                error=error,
            )
        )
        return cost

    def _record_hits(self, endpoint: Endpoint, role: str, count: int) -> None:
        self.usage.add(
            UsageRecord(
                time=time.time(),
                role=role,
                model=endpoint.name,
                served_model=None,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
                cost_usd=0.0,
                cost_source="cache",
                latency_s=0.0,
                calls=count,
                cached=True,
            )
        )

    def _cached_generation(
        self, key: str, schema: type[BaseModel] | None
    ) -> BaseModel | str | None:
        if self._cache is None:
            return None
        stored = self._cache.get_many([key]).get(key)
        if schema is None:
            return stored if isinstance(stored, str) and stored.strip() else None
        try:
            return schema.model_validate(stored) if isinstance(stored, dict) else None
        except ValidationError:
            return None

    def _cached(self, keys: Iterable[str]) -> dict[str, JSON]:
        return self._cache.get_many(keys) if self._cache is not None else {}

    def _store(self, replies: Mapping[str, JSON]) -> None:
        if self._cache is not None and replies:
            self._cache.put_many(replies)


@dataclass(slots=True)
class _AnswerPass:
    """One model's turn at the questions still unanswered."""

    session: LLMSession
    endpoint: Endpoint
    samples: Sequence[str]
    digests: Sequence[str]
    questions: Mapping[str, Question]
    role: str
    draw: int
    progress: Callable[[int], None] | None

    async def run(self, results: list[Answers], errors: list[str]) -> bool:
        """Fill what this model can. Returns whether anything was left to ask."""
        cells = [
            (i, qid) for i, r in enumerate(results) for qid, a in r.items() if a is None
        ]
        if not cells:
            return False
        keys = {cell: self._key(*cell) for cell in cells}
        stored = self.session._cached(keys.values())
        missing: list[tuple[int, str]] = []
        for (i, qid), key in keys.items():
            answer = _load_answer(stored.get(key), self.questions[qid])
            if answer is None:
                missing.append((i, qid))
            else:
                results[i][qid] = answer
        if hits := len(cells) - len(missing):
            self.session._record_hits(self.endpoint, self.role, hits)
            self._advance(hits)
        requests = self._requests(missing, keys)
        self._check_estimate(requests)
        try:
            await _bounded(requests, lambda r: self._send(r, results, errors), _WINDOW)
        except BaseExceptionGroup as group:
            raise _fatal(group) from None
        return True

    def _key(self, i: int, qid: str) -> str:
        question = self.questions[qid]
        return cache_key(
            kind="answer",
            endpoint=_endpoint_identity(self.endpoint),
            sample=self.digests[i],
            question=[question.text, list(question.labels), type(question).__name__],
            draw=self.draw,
        )

    def _requests(
        self, cells: Sequence[tuple[int, str]], keys: Mapping[tuple[int, str], str]
    ) -> list[_Request]:
        """One request per sample and chunk of questions; identical ones merged."""
        per_call = ADAPTERS[self.endpoint.protocol].max_questions(self.endpoint)
        by_sample: dict[int, list[str]] = {}
        for i, qid in cells:
            by_sample.setdefault(i, []).append(qid)
        merged: dict[tuple[str, ...], _Request] = {}
        for i, qids in by_sample.items():
            for start in range(0, len(qids), per_call):
                chunk = qids[start : start + per_call]
                chunk_keys = tuple(keys[(i, qid)] for qid in chunk)
                if chunk_keys in merged:
                    merged[chunk_keys].targets.append(i)
                    continue
                merged[chunk_keys] = _Request(i, chunk, chunk_keys, [i])
        return list(merged.values())

    def _check_estimate(self, requests: Sequence[_Request]) -> None:
        if not requests:
            return
        tokens = TokenCounts(
            input=sum(self._estimate(r) for r in requests),
            output=sum(len(r.qids) for r in requests) * _ANSWER_OUTPUT_ESTIMATE,
        )
        cost = call_cost(self.endpoint, tokens, None)[0]
        what = f"Answering with {self.endpoint.name} ({len(requests)} calls)"
        logger.info(
            f"{what}: estimated "
            f"{format_usd(cost) if cost is not None else 'at an unknown cost'}."
        )
        if cost is not None:
            self.session.budget.check_estimate(cost, what)

    def _estimate(self, request: _Request) -> int:
        texts = [self.questions[qid].text for qid in request.qids]
        return _tokens(self.samples[request.sample], *texts)

    async def _send(
        self, request: _Request, results: list[Answers], errors: list[str]
    ) -> None:
        sample = self.samples[request.sample]
        asked = {qid: self.questions[qid] for qid in request.qids}
        adapter, endpoint = ADAPTERS[self.endpoint.protocol], self.endpoint
        try:
            reply = await self.session._call(
                endpoint,
                self.role,
                lambda dropped: adapter.answer(endpoint, sample, asked, dropped),
                TokenCounts(
                    input=self._estimate(request),
                    output=len(asked) * _ANSWER_OUTPUT_ESTIMATE,
                ),
            )
        except ProviderError as error:
            errors.append(f"{endpoint.name}: {error}")
            return
        stored: dict[str, JSON] = {}
        for qid, key in zip(request.qids, request.keys):
            if (answer := reply.output.get(qid)) is None:
                continue
            probabilities: dict[str, JSON] = {**answer.probabilities}
            stored[key] = {"label": answer.label, "probabilities": probabilities}
            for i in request.targets:
                results[i][qid] = answer
        self.session._store(stored)
        self._advance(len(stored) * len(request.targets))
        self.session.budget.check()

    def _advance(self, count: int) -> None:
        if self.progress is not None and count:
            self.progress(count)


@dataclass(slots=True)
class _Request:
    sample: int
    qids: list[str]
    keys: tuple[str, ...]
    targets: list[int]


async def _bounded[X](
    items: Iterable[X], run: Callable[[X], Awaitable[None]], limit: int
) -> None:
    """Run ``run`` on each item, at most ``limit`` at a time."""
    slots = asyncio.Semaphore(limit)

    async def guarded(item: X) -> None:
        try:
            await run(item)
        finally:
            slots.release()

    async with asyncio.TaskGroup() as group:
        for item in items:
            await slots.acquire()
            group.create_task(guarded(item))


def _fatal(group: BaseExceptionGroup[BaseException]) -> BaseException:
    """The error that stopped the run, out of the task group's errors."""
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            return _fatal(error)
        if isinstance(error, (ProviderAuthError, BudgetExceededError)):
            return error
    return group.exceptions[0]


def _endpoint_identity(endpoint: Endpoint) -> dict[str, JSON]:
    settings = endpoint.settings
    return {
        "protocol": endpoint.protocol,
        "base_url": endpoint.base_url,
        "model": endpoint.model_id,
        "temperature": settings.temperature,
        "reasoning_effort": settings.reasoning_effort,
        "questions_per_call": settings.questions_per_call,
    }


def _generation_key(endpoint: Endpoint, request: GenerateRequest, draw: int) -> str:
    schema = request.schema.model_json_schema() if request.schema is not None else None
    return cache_key(
        kind="generate",
        endpoint=_endpoint_identity(endpoint),
        prompt=request.prompt,
        instructions=request.instructions,
        prefix=request.prefix,
        schema=schema,
        draw=draw,
    )


def _usable(output: object, schema: type[BaseModel] | None) -> TypeIs[BaseModel | str]:
    if schema is None:
        return isinstance(output, str) and bool(output.strip())
    return isinstance(output, schema)


def _dump(output: BaseModel | str | None) -> JSON:
    return output.model_dump(mode="json") if isinstance(output, BaseModel) else output


def _load_answer(stored: JSON, question: Question) -> Answer | None:
    if not isinstance(stored, dict):
        return None
    label, probabilities = stored.get("label"), stored.get("probabilities")
    if (
        not isinstance(label, str)
        or label not in question.labels
        or not isinstance(probabilities, dict)
    ):
        return None
    return Answer(
        label,
        {k: float(v) for k, v in probabilities.items() if isinstance(v, (int, float))},
    )


def _tokens(*texts: str | None) -> int:
    """A rough token count: four characters a token, plus framing."""
    return 100 + sum(len(text) for text in texts if text) // 4
