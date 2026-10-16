"""The limiter, prices and usage, the reply cache and the library's event loop."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import time
from pathlib import Path

import pytest

from think_reason_learn.exceptions import BudgetExceededError
from think_reason_learn.llm import (
    LLM,
    Budget,
    LLMSession,
    Price,
    TokenCounts,
    Usage,
    YesNoQuestion,
    config_context,
    format_usd,
    get_config,
    resolve,
    resolve_models,
    run_async,
    run_sync,
    set_config,
    set_limits,
)
from think_reason_learn.llm._cache import Cache, cache_key
from think_reason_learn.llm._config import Limits, limits_for
from think_reason_learn.llm._limiter import (
    Limiter,
    _Bucket,
    _rate_headers,
    _reset_seconds,
)
from think_reason_learn.llm._usage import call_cost
from think_reason_learn.testing import FakeLLM, fake_llms


class TestLimiter:
    @pytest.mark.asyncio
    async def test_concurrency_never_passes_the_limit(self) -> None:
        limiter = Limiter(Limits(max_concurrency=2), local=False)
        peak = 0

        async def call() -> None:
            nonlocal peak
            await limiter.acquire(1)
            peak = max(peak, limiter._in_flight)
            await asyncio.sleep(0.01)
            await limiter.release(estimated=1, used=1, headers={}, outcome="ok")

        async def many() -> None:
            await asyncio.gather(*(call() for _ in range(10)))

        await many()
        assert peak == 2

    @pytest.mark.asyncio
    async def test_rate_limit_halves_and_pauses(self) -> None:
        limiter = Limiter(Limits(max_concurrency=64), local=False)

        async def hit() -> None:
            await limiter.acquire(1)
            await limiter.release(
                estimated=1,
                used=None,
                headers={},
                outcome="rate_limited",
                retry_after=5,
            )

        await hit()
        assert limiter._allowed == 4
        assert limiter._delay(1) == pytest.approx(5, abs=0.1)

    @pytest.mark.asyncio
    async def test_success_raises_concurrency_to_the_ceiling(self) -> None:
        limiter = Limiter(Limits(max_concurrency=9), local=False)

        async def ok() -> None:
            for _ in range(3):
                await limiter.acquire(1)
                await limiter.release(
                    estimated=1,
                    used=1,
                    headers={},
                    outcome="ok",
                )

        await ok()
        assert limiter._allowed == 9

    def test_local_servers_start_low(self) -> None:
        assert Limiter(Limits(), local=True)._allowed == 4

    def test_bucket_math(self) -> None:
        bucket = _Bucket.per_minute(60)  # one a second, ten seconds of burst
        assert bucket.capacity == 10
        bucket.take(10)
        assert bucket.wait(1, bucket.stamp) == pytest.approx(1.0)
        assert bucket.wait(1, bucket.stamp + 1) == 0.0

    def test_spent_window_pauses_until_reset(self) -> None:
        limiter = Limiter(Limits(), local=False)
        limiter._observe(
            {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2s"}
        )
        assert limiter._delay(1) == pytest.approx(2, abs=0.1)

    def test_limits_are_learned_from_headers(self) -> None:
        limiter = Limiter(Limits(), local=False)
        limiter._observe(
            {
                "x-ratelimit-limit-requests": "600",
                "x-ratelimit-remaining-requests": "599",
            }
        )
        assert limiter._requests is not None and limiter._requests.per_second == 10

    def test_header_formats(self) -> None:
        assert _reset_seconds("6m0s") == 360 and _reset_seconds("20ms") == 0.02
        future = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30))
        assert 25 < (_reset_seconds(future) or 0) <= 30
        anthropic = {
            "anthropic-ratelimit-tokens-limit": "8000",
            "anthropic-ratelimit-tokens-remaining": "0",
        }
        assert _rate_headers(anthropic, "tokens")[:2] == (8000, 0)

    def test_long_waits_are_capped(self) -> None:
        limiter = Limiter(Limits(), local=False)
        limiter._pause(86_400)  # a daily quota's Retry-After
        assert limiter._delay(1) == pytest.approx(120, abs=0.1)

    def test_a_burst_of_rate_limits_halves_once(self) -> None:
        async def burst() -> int:
            limiter = Limiter(Limits(max_concurrency=64), local=False)
            for _ in range(3):
                limiter._in_flight += 1
                await limiter.release(
                    estimated=1, used=None, headers={}, outcome="rate_limited"
                )
            return limiter._allowed

        assert run_sync(burst()) == 4

    def test_reapplying_the_same_limits_keeps_the_spent_window(self) -> None:
        limiter = Limiter(Limits(rpm=60), local=False)
        assert limiter._requests is not None
        limiter._requests.take(10)
        limiter.configure(Limits(rpm=60), local=False)  # a new run, same limits
        assert limiter._delay(1) > 0.5

    def test_limits_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="above 0"):
            set_limits("openai", rpm=0)
        with pytest.raises(ValueError, match=">= 1"):
            set_limits("openai", max_concurrency=0)

    def test_set_limits_by_model_or_provider(self) -> None:
        set_limits("openai", max_concurrency=3)
        set_limits("openai:gpt-5-mini", rpm=100)
        openai_model = resolve(LLM("openai:gpt-5-mini", api_key="k"))
        assert limits_for(openai_model) == Limits(rpm=100)
        assert (
            limits_for(resolve(LLM("openai:gpt-4.1", api_key="k"))).max_concurrency == 3
        )


class TestUsageAndPrices:
    def test_price_counts_cache_reads_cheaper(self) -> None:
        price = Price(input=2.0, output=8.0, cached_input=0.5)
        tokens = TokenCounts(input=1_000_000, output=100_000, cache_read=400_000)
        assert price.cost(tokens) == pytest.approx(
            600_000 * 2 / 1e6 + 400_000 * 0.5 / 1e6 + 0.8
        )

    def test_cost_comes_from_provider_then_price_then_table(self) -> None:
        tokens = TokenCounts(input=1_000_000, output=1_000_000)
        priced = resolve(LLM("openai:gpt-5-mini", api_key="k", price=Price(1.0, 1.0)))
        assert call_cost(priced, tokens, 0.5) == (0.5, "provider")
        assert call_cost(priced, tokens, None) == (2.0, "price")
        listed = call_cost(resolve(LLM("openai:gpt-5-mini", api_key="k")), tokens, None)
        assert listed[1] == "price_table" and (listed[0] or 0) > 0
        unknown = resolve(LLM("ollama:my-own-model"))
        assert call_cost(unknown, tokens, None) == (None, "unknown")

    def test_budget(self) -> None:
        budget = Budget(1.0)
        held = budget.reserve(0.6)
        with pytest.raises(BudgetExceededError):
            budget.reserve(0.6)
        budget.settle(held, 1.2)
        with pytest.raises(BudgetExceededError, match="cap"):
            budget.check()
        assert Budget(None).reserve(1e9) == 1e9

    def test_usage_summary_and_export(self, tmp_path: Path) -> None:
        with fake_llms(judge=FakeLLM(answer=lambda s, q: "YES")):
            session = LLMSession()
            models = resolve_models(
                LLM("fake:judge", price=Price(1.0)), param="a", task="answer"
            )
            from think_reason_learn.llm import YesNoQuestion

            run_sync(
                session.answer(
                    models, ["a", "b"], {"q": YesNoQuestion("Q?")}, role="answerer"
                )
            )
        usage = session.usage
        assert "calls=2" in repr(usage)
        assert usage.to_frame().shape[0] == 2
        usage.to_json(tmp_path / "run.json")
        assert len(json.loads((tmp_path / "run.json").read_text())) == 2
        assert Usage().by("model").empty

    def test_format_usd(self) -> None:
        assert (format_usd(0), format_usd(1.234), format_usd(0.000123)) == (
            "$0.00",
            "$1.23",
            "$0.00012",
        )


class TestCache:
    def test_round_trip_and_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "c.sqlite"
        keys = [cache_key(n=i) for i in range(1200)]
        Cache(path).put_many({key: {"i": i} for i, key in enumerate(keys)})
        found = Cache(path).get_many(keys + ["missing"])
        assert len(found) == 1200 and found[keys[7]] == {"i": 7}

    def test_keys_depend_on_every_part(self) -> None:
        assert cache_key(a=1, b=2) == cache_key(b=2, a=1)
        assert cache_key(a=1, draw=0) != cache_key(a=1, draw=1)


class TestConfig:
    def test_a_cap_can_be_lifted(self) -> None:
        with config_context(max_cost_usd=5):
            with config_context(max_cost_usd=None, timeout=10):
                assert (get_config().max_cost_usd, get_config().timeout) == (None, 10)
            assert get_config().max_cost_usd == 5

    def test_set_config_changes_only_what_is_passed(self) -> None:
        before = get_config()
        try:
            set_config(max_retries=1)
            assert get_config().cache_dir == before.cache_dir
        finally:
            set_config(max_retries=before.max_retries)


class TestLoop:
    @pytest.mark.asyncio
    async def test_run_async_from_another_loop(self) -> None:
        async def three() -> int:
            await asyncio.sleep(0)
            return 3

        async def user() -> int:
            return await run_async(three())

        assert await user() == 3

    def test_errors_reach_the_caller(self) -> None:
        async def boom() -> None:
            raise TimeoutError("the call timed out")

        with pytest.raises(TimeoutError, match="timed out"):
            run_sync(boom())

    def test_sync_inside_the_library_loop_is_refused(self) -> None:
        async def nested() -> None:
            async def inner() -> None:
                return None

            run_sync(inner())

        with pytest.raises(RuntimeError, match="async API"):
            run_sync(nested())

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork with threads
    def test_a_forked_child_calls_models_on_its_own_loop(self) -> None:
        with fake_llms(judge=FakeLLM(answer=lambda sample, q: "YES")):
            assert _ask_once() == "YES"  # the parent's loop, limiter and cache
            with multiprocessing.get_context("fork").Pool(1) as pool:
                assert pool.apply(_ask_once) == "YES"


def _ask_once() -> str | None:
    with config_context(cache=False):
        models = resolve_models("fake:judge", param="answerer", task="answer")
        [answers] = run_sync(
            LLMSession().answer(models, ["x"], {"q": YesNoQuestion("Q?")}, role="a")
        )
    answer = answers["q"]
    return answer.label if answer is not None else None
