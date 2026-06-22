"""Offline tests for the vanilla classifier -- no real API calls."""

from __future__ import annotations

from typing import Any, List, Sequence, Type

import pandas as pd
import pytest

from think_reason_learn.core.exceptions import DataError
from think_reason_learn.core.llms import GoogleChoice
from think_reason_learn.core.llms._schemas import LLMChoice, LLMResponse
from think_reason_learn.vanilla import (
    FLASH_5POINT,
    PRO_PRECISION,
    FounderScore,
    SCALES,
    VanillaClassifier,
    build_system_prompt,
    compute_metrics,
)


_PROVIDER = GoogleChoice(model="gemini-2.5-flash")


class FakeLLM:
    """Returns a canned :class:`FounderScore` keyed by the profile text.

    ``score_map`` maps a substring of the profile to the integer score to
    return. Profiles not matched fall back to ``default_score``.
    """

    def __init__(self, score_map: dict[str, int], default_score: int = 1) -> None:
        self.score_map = score_map
        self.default_score = default_score
        self.calls: List[dict[str, Any]] = []

    async def respond(
        self,
        query: str,
        llm_priority: Sequence[LLMChoice],
        response_format: Type[Any],
        instructions: Any = None,
        temperature: Any = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        self.calls.append(
            {"query": query, "temperature": temperature, "kwargs": kwargs}
        )
        score = self.default_score
        for needle, value in self.score_map.items():
            if needle in query:
                score = value
                break
        result = FounderScore(
            reasoning="canned", score=score, confidence=0.9, key_signals=["x"]
        )
        return LLMResponse(
            response=result,
            logprobs=[],
            total_tokens=42,
            provider_model=_PROVIDER,
        )


def _data() -> tuple[pd.DataFrame, List[int]]:
    persons = [
        ("strong founder, two prior exits", 1),
        ("weak founder, no experience", 0),
        ("excellent technical founder", 1),
        ("mediocre first-timer", 0),
    ]
    X = pd.DataFrame({"data": [p for p, _ in persons]})
    y = [label for _, label in persons]
    return X, y


# ---------------------------------------------------------------------------
# Pure helpers (no LLM)
# ---------------------------------------------------------------------------


def test_compute_metrics_basic() -> None:
    m = compute_metrics([1, 0, 1, 0], [1, 0, 0, 0], threshold=4)
    assert (m.tp, m.fp, m.tn, m.fn) == (1, 0, 2, 1)
    assert m.precision == 1.0
    assert m.recall == 0.5
    # F0.5 with p=1, r=0.5 -> 1.25 * 0.5 / (0.25 + 0.5) = 0.8333...
    assert round(m.f_beta, 4) == 0.8333
    assert m.threshold == 4


def test_default_threshold_per_scale() -> None:
    assert SCALES["binary"].default_threshold == 1
    assert SCALES["ternary"].default_threshold == 2
    assert SCALES["5point"].default_threshold == 4


def test_build_system_prompt_varies_by_format() -> None:
    assert "1 to 5" in build_system_prompt("5point")
    assert "0 or 1" in build_system_prompt("binary")
    with pytest.raises(ValueError):
        build_system_prompt("septenary")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_preset_applies_cookbook_defaults() -> None:
    clf = VanillaClassifier.from_preset(FLASH_5POINT, _llm=FakeLLM({}))
    assert clf.scoring_format == "5point"
    assert clf.threshold == 4
    assert clf.temperature == 0.0


def test_from_preset_overrides() -> None:
    clf = VanillaClassifier.from_preset(
        PRO_PRECISION, temperature=1.0, threshold=1, _llm=FakeLLM({})
    )
    assert clf.scoring_format == "ternary"
    assert clf.temperature == 1.0
    assert clf.threshold == 1


def test_invalid_scoring_format_raises() -> None:
    with pytest.raises(ValueError):
        VanillaClassifier(scoring_format="nope")  # type: ignore[arg-type]


def test_invalid_semaphore_raises() -> None:
    with pytest.raises(ValueError):
        VanillaClassifier(llm_semaphore_limit=0)


# ---------------------------------------------------------------------------
# Scoring / prediction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_returns_aligned_frame() -> None:
    X, _ = _data()
    fake = FakeLLM({"strong": 5, "weak": 1, "excellent": 4, "mediocre": 2})
    clf = VanillaClassifier(
        scoring_format="5point", threshold=4, model="gemini-2.5-flash", _llm=fake
    )
    scored = await clf.score(X)

    assert list(scored.index) == list(X.index)
    assert scored.loc[0, "score"] == 5 and scored.loc[0, "predicted"] == 1
    assert scored.loc[1, "score"] == 1 and scored.loc[1, "predicted"] == 0
    assert scored.loc[3, "score"] == 2 and scored.loc[3, "predicted"] == 0
    assert (scored["status"] == "OK").all()
    # token usage accumulated
    assert clf.token_usage.token_counts


@pytest.mark.asyncio
async def test_predict_and_evaluate() -> None:
    X, y = _data()
    fake = FakeLLM({"strong": 5, "weak": 1, "excellent": 4, "mediocre": 2})
    clf = VanillaClassifier(scoring_format="5point", threshold=4, _llm=fake)

    preds = await clf.predict(X)
    assert preds == [1, 0, 1, 0]

    metrics = await clf.evaluate(X, y)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f_beta == 1.0


@pytest.mark.asyncio
async def test_failed_row_is_isolated() -> None:
    class BoomLLM(FakeLLM):
        async def respond(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("boom")

    X, _ = _data()
    clf = VanillaClassifier(scoring_format="5point", _llm=BoomLLM({}))
    scored = await clf.score(X)
    assert (scored["status"] == "FAIL").all()
    assert (scored["predicted"] == 0).all()
    assert scored["error"].notna().to_numpy().all()


@pytest.mark.asyncio
async def test_temperature_and_seed_forwarded() -> None:
    X, _ = _data()
    fake = FakeLLM({})
    clf = VanillaClassifier(
        scoring_format="binary", temperature=1.0, seed=7, _llm=fake
    )
    await clf.score(X.head(1))
    assert fake.calls[0]["temperature"] == 1.0
    assert fake.calls[0]["kwargs"].get("seed") == 7


@pytest.mark.asyncio
async def test_missing_column_raises() -> None:
    clf = VanillaClassifier(_llm=FakeLLM({}))
    with pytest.raises(DataError):
        await clf.score(pd.DataFrame({"profile": ["x"]}))


@pytest.mark.asyncio
async def test_evaluate_length_mismatch_raises() -> None:
    X, _ = _data()
    clf = VanillaClassifier(_llm=FakeLLM({}))
    with pytest.raises(DataError):
        await clf.evaluate(X, [1, 0])


@pytest.mark.asyncio
async def test_sweep_thresholds_single_call_set() -> None:
    X, y = _data()
    fake = FakeLLM({"strong": 5, "weak": 1, "excellent": 4, "mediocre": 2})
    clf = VanillaClassifier(scoring_format="5point", threshold=4, _llm=fake)

    sweep = await clf.sweep_thresholds(X, y)
    # Only one LLM call per row, regardless of how many thresholds are swept.
    assert len(fake.calls) == len(X)
    assert set(sweep["threshold"]) == {2, 3, 4, 5}
    # Sorted by f_beta descending.
    assert sweep["f_beta"].is_monotonic_decreasing
