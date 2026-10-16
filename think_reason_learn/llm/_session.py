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
from dataclasses import astuple, dataclass
from functools import partial
from typing import Final, TypeIs, overload

from pydantic import BaseModel, ValidationError

from think_reason_learn.exceptions import (
    BudgetExceededError,
    NoOutputError,
    ProviderAuthError,
)

from ._adapters import ADAPTERS, Answers, GenerateRequest, ProviderError, Reply
from ._cache import JSON, Cache, cache_key, open_cache
from ._calls import Skipped, send
from ._config import get_config
from ._estimate import (
    GENERATE_OUTPUT_TOKENS,
    answer_output_tokens,
    call_tokens,
    planned_cost,
)
from ._limiter import apply_limits
from ._questions import Answer, Question
from ._spec import Endpoint
from ._usage import Budget, TokenCounts, Usage, UsageRecord, call_cost, format_usd

logger = logging.getLogger(__name__)

_WINDOW: Final = 256
"""Answer requests scheduled at once; the limiter decides how many are sent."""
_DEAD_AFTER: Final = 5
"""Failed requests in a row, retries spent, after which the next model takes over."""


class LLMSession:
    """Every model call of one run.

    Its cost cap, cache and retries are those in effect when it is created
    (``get_config()``).

    Args:
        usage: Where calls are recorded; a new ``Usage`` by default.
        progress: Called with the number of answers each time some arrive.
    """

    def __init__(
        self,
        *,
        usage: Usage | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> None:
        """Start a run under the current settings."""
        self._engine = _Engine(usage, progress)
        self.usage = self._engine.usage

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
            if (hit := self._engine.cached_generation(key, schema)) is not None:
                self._engine.record_hits(endpoint, role, 1)
                return hit
            adapter = ADAPTERS[endpoint.protocol]
            try:
                reply = await self._engine.call(
                    endpoint,
                    role,
                    partial(adapter.generate, endpoint, request),
                    TokenCounts(
                        input=call_tokens(prompt, instructions, prefix),
                        output=GENERATE_OUTPUT_TOKENS,
                    ),
                )
            except ProviderError as error:
                errors.append(f"{endpoint.name}: {error}")
                continue
            if _usable(reply.output, schema):
                self._engine.store({key: _dump(reply.output)})
                self._engine.budget.check()
                return reply.output
            errors.append(f"{endpoint.name}: no usable output")
        raise NoOutputError(
            f"No model gave usable output for {role}: {'; '.join(errors)}"
        )

    async def answer(
        self,
        models: Sequence[Endpoint],
        samples: Sequence[str],
        questions: Mapping[str, Question],
        *,
        role: str,
        draw: int = 0,
    ) -> list[Answers]:
        """Answer every question about every sample.

        A question a model fails to answer goes to the next model.

        Args:
            models: Models in fallback order, reasoning or decision.
            samples: Sample texts.
            questions: Questions by id.
            role: Who is asking, recorded with usage, e.g. ``"answerer"``.
            draw: Different draws ask again; the same draw reuses answers.

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
        for n, endpoint in enumerate(models, 1):
            pass_ = _AnswerPass(
                self._engine,
                endpoint,
                samples,
                digests,
                questions,
                role,
                draw,
                can_hand_over=n < len(models),
            )
            if not await pass_.run(results, errors):
                break
        _report(results, errors, models)
        return results


class _Engine:
    """The machinery behind a session: calls, the cache, usage and the budget."""

    def __init__(
        self, usage: Usage | None, progress: Callable[[int], None] | None
    ) -> None:
        """Read the settings in effect now; open the cache if it is on."""
        self.progress = progress
        self.config = get_config()
        self.usage = usage if usage is not None else Usage()
        self.budget = Budget(self.config.max_cost_usd)
        self._cache: Cache | None = (
            open_cache(self.config.cache_path) if self.config.cache else None
        )
        self._configured: set[str] = set()
        self._served: dict[str, set[str]] = {}

    async def call[T](
        self,
        endpoint: Endpoint,
        role: str,
        call: Callable[[frozenset[str]], Awaitable[Reply[T]]],
        estimate: TokenCounts,
        stopped: Callable[[], bool] = lambda: False,
    ) -> Reply[T]:
        """One call under the budget, recorded in usage; skipped if ``stopped()``."""
        if endpoint.key not in self._configured:
            self._first_call(endpoint, estimate)
        held = self.budget.reserve(call_cost(endpoint, estimate, None)[0])
        start, cost = time.perf_counter(), None
        try:
            reply = await send(
                endpoint,
                call,
                estimated_tokens=estimate.input,
                max_retries=self.config.max_retries,
                stopped=stopped,
            )
            cost = self._record(
                endpoint,
                role,
                reply.tokens,
                start,
                served=reply.served_model,
                provider_cost=reply.provider_cost,
            )
            return reply
        except ProviderError as error:
            if not isinstance(error, Skipped):  # nothing was sent
                self._record(endpoint, role, TokenCounts(), start, error=str(error))
            if error.failure in ("auth", "quota"):
                raise ProviderAuthError(f"{endpoint.name}: {error}") from error
            raise
        finally:
            self.budget.settle(held, cost)

    def _first_call(self, endpoint: Endpoint, estimate: TokenCounts) -> None:
        apply_limits(endpoint)
        self._configured.add(endpoint.key)
        if self.budget.max_cost_usd is not None and (
            call_cost(endpoint, estimate, None)[0] is None
        ):
            logger.warning(
                f"{endpoint.name} has no known price, so max_cost_usd does not "
                "count its calls. Set one with LLM(..., price=Price(...))."
            )

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
        tokens = TokenCounts(*(max(0, n) for n in astuple(tokens)))  # some send -1
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
                cost_usd=cost,
                cost_source=source,
                served_model=served,
                input_tokens=tokens.input,
                output_tokens=tokens.output,
                cache_read_tokens=tokens.cache_read,
                cache_write_tokens=tokens.cache_write,
                reasoning_tokens=tokens.reasoning,
                latency_s=time.perf_counter() - start,
                error=error,
            )
        )
        self._check_served(endpoint, served)
        return cost

    def record_hits(self, endpoint: Endpoint, role: str, count: int) -> None:
        self.usage.add(
            UsageRecord(
                time=time.time(),
                role=role,
                model=endpoint.name,
                cost_usd=0.0,
                cost_source="cache",
                calls=count,
                cached=True,
            )
        )

    def _check_served(self, endpoint: Endpoint, served: str | None) -> None:
        """Warn once when a provider switches the model behind a name mid-run."""
        if served is None:
            return
        seen = self._served.setdefault(endpoint.name, set())
        seen.add(served)
        if len(seen) == 2:
            logger.warning(
                f"{endpoint.name} was served by {sorted(seen)} in one run; "
                "pin a dated model for reproducible results."
            )

    def cached_generation(
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

    def cached(self, keys: Iterable[str]) -> dict[str, JSON]:
        return self._cache.get_many(keys) if self._cache is not None else {}

    def store(self, replies: Mapping[str, JSON]) -> None:
        if self._cache is not None and replies:
            self._cache.put_many(replies)


@dataclass(slots=True)
class _AnswerPass:
    """One model's turn at the questions still unanswered."""

    engine: _Engine
    endpoint: Endpoint
    samples: Sequence[str]
    digests: Sequence[str]
    questions: Mapping[str, Question]
    role: str
    draw: int
    can_hand_over: bool = False
    unavailable: bool = False
    failed_in_a_row: int = 0
    fatal: ProviderAuthError | BudgetExceededError | None = None

    async def run(self, results: list[Answers], errors: list[str]) -> bool:
        """Fill what this model can. Returns whether anything was left to ask."""
        cells = [
            (i, qid) for i, r in enumerate(results) for qid, a in r.items() if a is None
        ]
        if not cells:
            return False
        keys = {cell: self._key(*cell) for cell in cells}
        stored = self.engine.cached(keys.values())
        missing: list[tuple[int, str]] = []
        for (i, qid), key in keys.items():
            answer = _load_answer(stored.get(key), self.questions[qid])
            if answer is None:
                missing.append((i, qid))
            else:
                results[i][qid] = answer
        if hits := len(cells) - len(missing):
            self.engine.record_hits(self.endpoint, self.role, hits)
            self._advance(hits)
        requests = self._requests(missing, keys)
        self._check_estimate(requests)
        try:
            await _bounded(requests, lambda r: self._send(r, results, errors), _WINDOW)
        except BaseExceptionGroup as group:
            raise group.exceptions[0] from None  # tasks do not nest groups
        if self.fatal is not None:
            raise self.fatal
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
            output=sum(
                answer_output_tokens(self.endpoint, len(r.qids)) for r in requests
            ),
        )
        cost = planned_cost(self.endpoint, len(requests), tokens)[0]
        what = f"Answering with {self.endpoint.name} ({len(requests)} calls)"
        logger.info(
            f"{what}: estimated "
            f"{format_usd(cost) if cost is not None else 'at an unknown cost'}."
        )
        if cost is not None:
            self.engine.budget.check_estimate(cost, what)

    def _estimate(self, request: _Request) -> int:
        texts = [self.questions[qid].text for qid in request.qids]
        return call_tokens(self.samples[request.sample], *texts)

    async def _send(
        self, request: _Request, results: list[Answers], errors: list[str]
    ) -> None:
        """Ask, unless the run is stopping; calls in flight finish and are kept.

        A model that is gone, or fails ``_DEAD_AFTER`` times in a row, hands the
        rest to the next model; the last model keeps trying. A bad request is
        one sample's problem and does not count.
        """
        if self.fatal is not None or self.unavailable:
            return
        try:
            await self._ask(request, results)
        except Skipped:
            return
        except ProviderError as error:
            errors.append(f"{self.endpoint.name}: {error}")
            if error.failure != "bad_request":
                self.failed_in_a_row += 1
            gone = error.failure == "unavailable" or self.failed_in_a_row >= _DEAD_AFTER
            self.unavailable = self.can_hand_over and gone
        except (ProviderAuthError, BudgetExceededError) as error:
            self.fatal = error
        else:
            self.failed_in_a_row = 0

    async def _ask(self, request: _Request, results: list[Answers]) -> None:
        sample = self.samples[request.sample]
        asked = {qid: self.questions[qid] for qid in request.qids}
        adapter, endpoint = ADAPTERS[self.endpoint.protocol], self.endpoint
        reply = await self.engine.call(
            endpoint,
            self.role,
            lambda dropped: adapter.answer(endpoint, sample, asked, dropped),
            TokenCounts(
                input=self._estimate(request),
                output=answer_output_tokens(endpoint, len(asked)),
            ),
            stopped=lambda: self.unavailable or self.fatal is not None,
        )
        stored: dict[str, JSON] = {}
        for qid, key in zip(request.qids, request.keys, strict=True):
            if (answer := reply.output.get(qid)) is None or not answer.is_valid:
                continue
            probabilities: dict[str, JSON] = {**answer.probabilities}
            stored[key] = {"label": answer.label, "probabilities": probabilities}
            for i in request.targets:
                results[i][qid] = answer
        self.engine.store(stored)
        self._advance(len(stored) * len(request.targets))
        self.engine.budget.check()

    def _advance(self, count: int) -> None:
        if self.engine.progress is not None and count:
            self.engine.progress(count)


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


def _report(
    results: Sequence[Answers], errors: Sequence[str], models: Sequence[Endpoint]
) -> None:
    """Raise if nothing was answered; else warn about the answers missing."""
    answers = [a for r in results for a in r.values()]
    if answers and answers.count(None) == len(answers):
        first = errors[0] if errors else "no usable answers"
        raise NoOutputError(
            f"No answers from {[m.name for m in models]}. First error: {first}"
        )
    if errors:
        logger.warning(
            f"{len(errors)} answer calls failed and {answers.count(None)} answers "
            f"are missing; the first error: {errors[0]}"
        )


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
    answer = Answer(
        label,
        {k: float(v) for k, v in probabilities.items() if isinstance(v, (int, float))},
    )
    return answer if answer.is_valid else None
