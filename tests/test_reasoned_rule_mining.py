"""Tests for ReasonedRuleMining.

Uses a dedicated deterministic fake LLM (rather than the RRF-specific
``tests/fake_llm.py``) so the new module's distinct call sites are stubbed
without coupling to or risking the existing RRF test double.
"""

from __future__ import annotations

import math
from typing import Any, List, Type

import numpy as np
import pandas as pd
import pytest

from think_reason_learn.core.exceptions import DataError
from think_reason_learn.core.llms import LLMResponse, OpenAIChoice
from think_reason_learn.reasoned_rule_mining import (
    ExtractedRule,
    ReasonedRuleMining,
    RRMConfig,
    Vote,
)

_PROVIDER = OpenAIChoice(model="gpt-4.1-nano")
_DUMMY_LLMC: List[Any] = [OpenAIChoice(model="gpt-4.1-nano")]


class FakeRRMLLM:
    """Deterministic stand-in for the LLM singleton for RRM tests.

    Votes are content-driven: a sample whose text contains "good" votes YES with
    high confidence, otherwise NO. This produces separable ensemble
    probabilities so Platt scaling + the combiner grid can fit.
    """

    def __init__(self, vote_logprob: float = -0.05, rule_logprob: float = 0.0) -> None:
        self.vote_logprob = vote_logprob
        self.rule_logprob = rule_logprob
        self.calls: List[dict] = []

    async def respond(
        self,
        query: str,
        llm_priority: List[Any],
        response_format: Type[Any],
        instructions: str = "",
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        """Route to a canned response based on response_format/instructions."""
        self.calls.append(
            {
                "query": query,
                "instructions": instructions,
                "response_format": response_format,
            }
        )
        if response_format is Vote:
            return self._vote(query)
        # str-typed sites, distinguished by instruction content.
        if "into a SINGLE structured logical rule" in instructions:
            return self._rule(query)
        if "compiling a decision policy" in instructions:
            return self._str(
                "For YES: IF good THEN label = YES\nFor NO: IF bad THEN label = NO"
            )
        if "Summarise the key recurring patterns" in instructions:
            return self._str("summary: good -> YES; bad -> NO")
        return self._str("Reasoning: the sample's attributes explain its label.")

    def _vote(self, query: str) -> LLMResponse[Vote]:
        sample_part = query.split("Sample:")[-1].lower()
        vote = "YES" if "good" in sample_part else "NO"
        return LLMResponse(
            response=Vote(vote=vote),
            logprobs=[(vote, self.vote_logprob)],
            total_tokens=10,
            provider_model=_PROVIDER,
        )

    def _rule(self, query: str) -> LLMResponse[str]:
        return LLMResponse(
            response="IF the sample is strong THEN label = YES",
            logprobs=[("t", self.rule_logprob), ("u", self.rule_logprob)],
            total_tokens=20,
            provider_model=_PROVIDER,
        )

    def _str(self, text: str) -> LLMResponse[str]:
        return LLMResponse(
            response=text,
            logprobs=[("t", -0.1)],
            total_tokens=15,
            provider_model=_PROVIDER,
        )


def _fast_config(**overrides: Any) -> RRMConfig:
    """A small config so the combiner grid search runs quickly in tests."""
    base: dict[str, Any] = dict(
        ensemble_size=3,
        weights_n_points=4,
        weights_thresh_points=12,
        recall_floors=(0.1,),
        use_memory=False,
    )
    base.update(overrides)
    return RRMConfig(**base)


def _make_rrm(fake: FakeRRMLLM | None = None, **kwargs: Any) -> ReasonedRuleMining:
    return ReasonedRuleMining(
        reason_llmc=kwargs.pop("reason_llmc", _DUMMY_LLMC),
        _llm=fake,
        **kwargs,
    )


def _toy_data(n_per_class: int = 4) -> tuple[pd.DataFrame, list[str]]:
    rows = [{"text": "strong good profile"}] * n_per_class
    rows += [{"text": "weak bad profile"}] * n_per_class
    y = ["YES"] * n_per_class + ["NO"] * n_per_class
    return pd.DataFrame(rows), y


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        rrm = _make_rrm()
        assert rrm.extract_llmc == rrm.reason_llmc
        assert rrm.predict_llmc == rrm.reason_llmc
        assert isinstance(rrm.config, RRMConfig)

    def test_empty_reason_llmc(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ReasonedRuleMining(reason_llmc=[])

    def test_bad_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            _make_rrm(reason_temperature=3.0)

    def test_bad_name(self) -> None:
        with pytest.raises(ValueError, match="alphanumeric"):
            _make_rrm(name="bad name!")

    def test_bad_semaphore(self) -> None:
        with pytest.raises(ValueError, match="llm_semaphore_limit"):
            _make_rrm(llm_semaphore_limit=0)


class TestConfigValidation:
    def test_bad_ensemble_size(self) -> None:
        with pytest.raises(ValueError, match="ensemble_size"):
            RRMConfig(ensemble_size=0)

    def test_bad_beta(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            RRMConfig(beta=0.0)

    def test_bad_harsh_level(self) -> None:
        with pytest.raises(ValueError, match="harsh_level"):
            RRMConfig(harsh_level="brutal")  # type: ignore[arg-type]

    def test_bad_recall_floor(self) -> None:
        with pytest.raises(ValueError, match="recall floor"):
            RRMConfig(recall_floors=(0.0, 1.5))


# ---------------------------------------------------------------------------
# Pure numeric helpers (no LLM)
# ---------------------------------------------------------------------------


class TestNumericHelpers:
    def test_perplexity_zero_logprob(self) -> None:
        assert ReasonedRuleMining._perplexity(
            [("a", 0.0), ("b", 0.0)]
        ) == pytest.approx(1.0)

    def test_perplexity_value(self) -> None:
        assert ReasonedRuleMining._perplexity([("a", -1.0)]) == pytest.approx(
            math.e, rel=1e-3
        )

    def test_perplexity_empty(self) -> None:
        assert ReasonedRuleMining._perplexity([]) == 1000.0

    def test_logit_half_is_zero(self) -> None:
        assert ReasonedRuleMining._logit(np.array([0.5]))[0] == pytest.approx(0.0)

    def test_platt_monotonic(self) -> None:
        rrm = _make_rrm()
        p = np.array([0.1, 0.2, 0.8, 0.9])
        y = np.array([0, 0, 1, 1])
        lr = rrm._platt_fit(p, y)
        out = rrm._platt_apply(lr, np.array([0.1, 0.9]))
        assert out[1] > out[0]

    def test_combiner_score_basic(self) -> None:
        rrm = _make_rrm()
        s = rrm._combiner_score(np.array([0.5]), np.array([np.nan]), 2.0, 0.0, 1.0)
        assert s[0] == pytest.approx(1.0)  # logit(0.5)=0 -> s = bias

    def test_combiner_score_ignores_nan_stream(self) -> None:
        rrm = _make_rrm()
        s = rrm._combiner_score(np.array([0.7311]), np.array([np.nan]), 1.0, 5.0, 0.0)
        assert s[0] == pytest.approx(1.0, abs=1e-3)  # beta term skipped (all nan)

    def test_best_fbeta_separable(self) -> None:
        rrm = _make_rrm(config=_fast_config())
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        y = np.array([0, 0, 1, 1])
        res = rrm._best_fbeta_at_recall(scores, y, 0.1)
        assert res is not None
        fbeta, thr, precision, recall = res
        assert recall >= 0.1 and precision > 0

    def test_best_fbeta_flat_scores(self) -> None:
        rrm = _make_rrm(config=_fast_config())
        res = rrm._best_fbeta_at_recall(np.array([0.5, 0.5]), np.array([0, 1]), 0.1)
        assert res is None

    def test_optimize_weights_separable(self) -> None:
        rrm = _make_rrm(config=_fast_config())
        ens = np.array([0.05, 0.05, 0.95, 0.95])
        mod = np.array([np.nan, np.nan, np.nan, np.nan])
        y = np.array([0, 0, 1, 1])
        best = rrm._optimize_weights(ens, mod, y, 0.1)
        assert best is not None
        assert "threshold" in best and best["recall"] >= 0.1

    def test_align_outcome_replaces(self) -> None:
        out = ReasonedRuleMining._align_outcome("IF x THEN label = NO", "YES")
        assert out == "IF x THEN label = YES"

    def test_align_outcome_appends(self) -> None:
        out = ReasonedRuleMining._align_outcome("IF x is true", "YES")
        assert out.endswith("THEN label = YES")


# ---------------------------------------------------------------------------
# Core algorithm path (fake LLM)
# ---------------------------------------------------------------------------


class TestFit:
    @pytest.mark.asyncio
    async def test_fit_sets_state(self) -> None:
        fake = FakeRRMLLM()
        rrm = _make_rrm(fake, config=_fast_config())
        await rrm.set_task("Predict whether a sample is good (YES) or bad (NO).")
        X, y = _toy_data()
        await rrm.fit(X, y)

        assert rrm.policy.startswith("For YES")
        assert len(rrm.rules) == len(X)
        assert all(isinstance(r, ExtractedRule) for r in rrm.rules)
        alpha, beta, bias = rrm.weights
        assert isinstance(alpha, float) and isinstance(bias, float)
        assert 0.0 <= rrm.threshold or rrm.threshold <= 0.0  # threshold is set
        assert rrm.validation_result["n_reviewed"] >= 0
        # token usage accumulated across calls
        assert sum(tc.value for tc in rrm.token_usage.token_counts.values()) > 0

    @pytest.mark.asyncio
    async def test_fit_requires_set_task(self) -> None:
        rrm = _make_rrm(FakeRRMLLM(), config=_fast_config())
        X, y = _toy_data()
        with pytest.raises(ValueError, match="set_task"):
            await rrm.fit(X, y)

    @pytest.mark.asyncio
    async def test_fit_empty_data(self) -> None:
        rrm = _make_rrm(FakeRRMLLM(), config=_fast_config())
        await rrm.set_task("task")
        with pytest.raises(DataError):
            await rrm.fit(pd.DataFrame({"text": []}), [])

    @pytest.mark.asyncio
    async def test_fit_length_mismatch(self) -> None:
        rrm = _make_rrm(FakeRRMLLM(), config=_fast_config())
        await rrm.set_task("task")
        with pytest.raises(DataError, match="same number"):
            await rrm.fit(pd.DataFrame({"text": ["a", "b"]}), ["YES"])

    @pytest.mark.asyncio
    async def test_fit_bad_labels(self) -> None:
        rrm = _make_rrm(FakeRRMLLM(), config=_fast_config())
        await rrm.set_task("task")
        with pytest.raises(DataError, match="YES"):
            await rrm.fit(pd.DataFrame({"text": ["a", "b"]}), ["MAYBE", "YES"])

    @pytest.mark.asyncio
    async def test_fit_single_class(self) -> None:
        rrm = _make_rrm(FakeRRMLLM(), config=_fast_config())
        await rrm.set_task("task")
        with pytest.raises(DataError, match="both"):
            await rrm.fit(pd.DataFrame({"text": ["a", "b"]}), ["YES", "YES"])

    @pytest.mark.asyncio
    async def test_set_task_empty(self) -> None:
        rrm = _make_rrm(FakeRRMLLM())
        with pytest.raises(ValueError, match="non-empty"):
            await rrm.set_task("   ")


class TestPolicyFilter:
    @pytest.mark.asyncio
    async def test_perplexity_filter_excludes_high(self) -> None:
        fake = FakeRRMLLM()
        rrm = _make_rrm(fake, config=_fast_config(perplexity_threshold=1.6))
        await rrm.set_task("task")
        rrm._rules = [
            ExtractedRule(
                rule="IF KEEPME THEN label = YES", outcome="YES", perplexity=1.0
            ),
            ExtractedRule(
                rule="IF DROPME THEN label = NO", outcome="NO", perplexity=9.9
            ),
        ]
        await rrm._compile_policy()
        policy_calls = [
            c for c in fake.calls if "compiling a decision policy" in c["instructions"]
        ]
        assert len(policy_calls) == 1
        query = policy_calls[0]["query"]
        assert "KEEPME" in query
        assert "DROPME" not in query


class TestMemory:
    @pytest.mark.asyncio
    async def test_memory_update_called(self) -> None:
        fake = FakeRRMLLM()
        rrm = _make_rrm(
            fake, config=_fast_config(use_memory=True, memory_update_interval=2)
        )
        await rrm.set_task("task")
        X, y = _toy_data(n_per_class=2)  # 4 rows, interval 2 -> one mid update
        await rrm.fit(X, y)
        mem_calls = [
            c
            for c in fake.calls
            if "Summarise the key recurring patterns" in c["instructions"]
        ]
        assert len(mem_calls) >= 1
        assert rrm._memory  # rolling summary populated


class TestPredict:
    @pytest.mark.asyncio
    async def test_predict_shape_and_labels(self) -> None:
        fake = FakeRRMLLM()
        rrm = _make_rrm(fake, config=_fast_config())
        await rrm.set_task("task")
        X, y = _toy_data()
        await rrm.fit(X, y)

        test_X = pd.DataFrame(
            [{"text": "strong good profile"}, {"text": "weak bad profile"}]
        )
        preds: dict[Any, str] = {}
        async for idx, scores, label, counter in rrm.predict(test_X):
            assert label in ("YES", "NO")
            assert "ensemble_prob" in scores and "score" in scores
            preds[idx] = label
        assert len(preds) == 2
        assert preds[0] == "YES"
        assert preds[1] == "NO"

    @pytest.mark.asyncio
    async def test_predict_before_fit_raises(self) -> None:
        rrm = _make_rrm(FakeRRMLLM(), config=_fast_config())
        with pytest.raises(RuntimeError, match="not fitted"):
            async for _ in rrm.predict(pd.DataFrame({"text": ["a"]})):
                pass

    @pytest.mark.asyncio
    async def test_predict_empty_samples(self) -> None:
        fake = FakeRRMLLM()
        rrm = _make_rrm(fake, config=_fast_config())
        await rrm.set_task("task")
        X, y = _toy_data()
        await rrm.fit(X, y)
        with pytest.raises(DataError, match="non-empty"):
            async for _ in rrm.predict(pd.DataFrame({"text": []})):
                pass


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, tmp_path: Any) -> None:
        fake = FakeRRMLLM()
        rrm = _make_rrm(fake, config=_fast_config())
        await rrm.set_task("task")
        X, y = _toy_data()
        await rrm.fit(X, y)
        rrm.save(tmp_path)

        loaded = ReasonedRuleMining.load(tmp_path)
        assert loaded.policy == rrm.policy
        assert loaded.weights == rrm.weights
        assert loaded.threshold == rrm.threshold
        assert len(loaded.rules) == len(rrm.rules)

        # Loaded model predicts using the same fake (re-injected).
        loaded._llm_instance = fake
        test_X = pd.DataFrame([{"text": "strong good profile"}])
        out = [label async for _, _, label, _ in loaded.predict(test_X)]
        assert out == ["YES"]
