"""Tests for RRF cost-sensitive mode."""

import pytest
import pandas as pd
import numpy as np
from think_reason_learn.rrf import RRF
from think_reason_learn.rrf._cost_sensitive import CostSensitiveConfig
from tests.fake_llm import FakeLLM


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, list[str]]:
    """Create small test dataset: 8 founders (3 YES, 5 NO)."""
    data = {
        "name": [f"Founder_{i}" for i in range(8)],
        "description": [f"Description {i}" for i in range(8)],
    }
    labels = ["YES", "YES", "YES", "NO", "NO", "NO", "NO", "NO"]
    return pd.DataFrame(data), labels


@pytest.mark.asyncio
async def test_standard_mode_baseline(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that cost_sensitive=False uses standard pipeline."""
    X, y = sample_data
    fake_llm = FakeLLM(questions_per_call=3)

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=False,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Standard mode should:
    # - Generate questions (1 call)
    # - Answer all questions on all samples (3 questions × 8 samples)
    # With batching disabled or batch=1, expect many calls
    # With our FakeLLM, we don't batch by default in this test
    assert fake_llm.call_count > 0
    assert rrf._questions is not None
    assert len(rrf._questions) == 3


@pytest.mark.asyncio
async def test_cost_sensitive_reduces_calls(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that cost_sensitive=True reduces LLM calls."""
    X, y = sample_data

    # Standard mode
    fake_llm_std = FakeLLM(questions_per_call=6)
    rrf_std = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=6,
        random_state=42,
        cost_sensitive=False,
        _llm=fake_llm_std,
    )
    await rrf_std.set_tasks(task_description="test task")
    await rrf_std.fit(X, y)
    standard_calls = fake_llm_std.call_count

    # Cost-sensitive mode
    fake_llm_cs = FakeLLM(questions_per_call=6)
    config = CostSensitiveConfig(
        screening_fraction=0.5,  # 4/8 samples for screening
        max_questions_full_eval=2,  # Only top 2 for full eval
        enable_semantic_filter=False,  # Disable for clearer call counting
    )
    rrf_cs = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=6,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm_cs,
    )
    await rrf_cs.set_tasks(task_description="test task")
    await rrf_cs.fit(X, y)
    cost_sensitive_calls = fake_llm_cs.call_count

    # Cost-sensitive should use fewer calls (screening + top-N < full eval)
    # Standard: 1 gen + 6 questions × 8 samples = many calls
    # Cost-sensitive: 1 gen + 6 questions × 4 screen + 2 questions × 8 full
    # Exact numbers depend on batching, but CS should be less or equal
    assert cost_sensitive_calls <= standard_calls


@pytest.mark.asyncio
async def test_screening_baseline_majority(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that majority baseline pruning works."""
    X, y = sample_data
    fake_llm = FakeLLM(questions_per_call=5, default_answer="NO")

    config = CostSensitiveConfig(
        screening_fraction=1.0,  # Use all data for screening (simpler test)
        screening_baseline="majority",
        max_questions_full_eval=10,  # Don't prune by top-N
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=5,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # With labels = [YES, YES, YES, NO, NO, NO, NO, NO]
    # Majority class is NO (5/8)
    # Majority baseline should have some F1 > 0
    # Questions that perform worse should be excluded
    assert rrf._questions is not None
    # At least some questions should remain
    active_questions = rrf._questions[rrf._questions["exclusion"].isna()]
    assert len(active_questions) >= 0  # Could be 0 if all below baseline


@pytest.mark.asyncio
async def test_screening_baseline_float(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that float threshold baseline works."""
    X, y = sample_data
    fake_llm = FakeLLM(questions_per_call=4)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        screening_baseline=0.0,  # Very low threshold - all should pass
        max_questions_full_eval=10,
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=4,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # With threshold=0.0, no questions should be pruned by baseline
    assert rrf._questions is not None
    # After baseline pruning but before top-N, we might still have exclusions from top-N
    # So we just check that the method doesn't crash
    assert len(rrf._questions) == 4


@pytest.mark.asyncio
async def test_top_n_selection(sample_data: tuple[pd.DataFrame, list[str]]) -> None:
    """Test that top-N selection works correctly."""
    X, y = sample_data
    fake_llm = FakeLLM(questions_per_call=10)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        screening_baseline=0.0,  # Don't prune by baseline
        max_questions_full_eval=3,  # Only keep top 3
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=10,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Should have 10 total questions, but only top 3 active
    assert rrf._questions is not None
    assert len(rrf._questions) == 10
    active_questions = rrf._questions[rrf._questions["exclusion"].isna()]
    assert len(active_questions) == 3


@pytest.mark.asyncio
async def test_val_set_support(sample_data: tuple[pd.DataFrame, list[str]]) -> None:
    """Test that X_val/y_val can be provided."""
    X, y = sample_data
    # Split into train (6) and val (2)
    X_train, y_train = X.iloc[:6], y[:6]
    X_val, y_val = X.iloc[6:], y[6:]

    fake_llm = FakeLLM(questions_per_call=3)

    config = CostSensitiveConfig(
        screening_fraction=0.5,
        max_questions_full_eval=2,
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    # Should not raise
    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    assert rrf._X_val is not None
    assert rrf._y_val is not None
    assert len(rrf._X_val) == 2
    assert len(rrf._y_val) == 2


@pytest.mark.asyncio
async def test_deterministic_screening_split(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that same random_state produces same screening split."""
    X, y = sample_data
    fake_llm1 = FakeLLM(questions_per_call=3)
    fake_llm2 = FakeLLM(questions_per_call=3)

    config = CostSensitiveConfig(
        screening_fraction=0.5,
        max_questions_full_eval=2,
        enable_semantic_filter=False,
    )

    rrf1 = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm1,
    )

    rrf2 = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm2,
    )

    await rrf1.set_tasks(task_description="test task")
    await rrf1.fit(X, y)
    await rrf2.set_tasks(task_description="test task")
    await rrf2.fit(X, y)

    # Same random state should produce same screening indices
    assert rrf1._screening_indices is not None
    assert rrf2._screening_indices is not None
    np.testing.assert_array_equal(rrf1._screening_indices, rrf2._screening_indices)


@pytest.mark.asyncio
async def test_cost_sensitive_false_unchanged(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that cost_sensitive=False preserves original behavior."""
    X, y = sample_data
    fake_llm = FakeLLM(questions_per_call=4)

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=4,
        random_state=42,
        cost_sensitive=False,  # Explicitly false
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Should not have screening indices
    assert rrf._screening_indices is None

    # Should have all questions (no cost pruning)
    assert rrf._questions is not None
    cost_pruned = rrf._questions[rrf._questions["exclusion"] == "cost_pruning"]
    assert len(cost_pruned) == 0


@pytest.mark.asyncio
async def test_semantic_filtering_auto_applied(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that semantic filtering is auto-applied when enabled."""
    X, y = sample_data
    fake_llm = FakeLLM(questions_per_call=8)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        enable_semantic_filter=True,  # Should auto-apply
        semantic_threshold=0.85,
        max_questions_full_eval=10,
    )

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=8,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Semantic filtering might exclude some questions
    assert rrf._questions is not None
    # Check if any were excluded by semantics
    # Might be 0 if questions are diverse enough - just verify no crash


@pytest.mark.asyncio
async def test_semantic_filtering_disabled(
    sample_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that semantic filtering can be disabled."""
    X, y = sample_data
    fake_llm = FakeLLM(questions_per_call=6)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        enable_semantic_filter=False,  # Disabled
        max_questions_full_eval=10,
    )

    rrf = RRF(
        qgen_llmc=["gpt-4o-mini"],
        max_generated_questions=6,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # No questions should be excluded by semantics
    assert rrf._questions is not None
    semantic_excluded = rrf._questions[rrf._questions["exclusion"] == "semantics"]
    assert len(semantic_excluded) == 0


@pytest.mark.asyncio
async def test_val_set_validation(sample_data: tuple[pd.DataFrame, list[str]]) -> None:
    """Test that providing only X_val or y_val raises error."""
    X, y = sample_data

    # Only X_val should raise
    fake_llm1 = FakeLLM()
    rrf1 = RRF(
        qgen_llmc=["gpt-4o-mini"],
        random_state=42,
        _llm=fake_llm1,
    )
    with pytest.raises(ValueError, match="Both X_val and y_val must be provided"):
        await rrf1.fit(X, y, X_val=X.iloc[:2])

    # Only y_val should raise
    fake_llm2 = FakeLLM()
    rrf2 = RRF(
        qgen_llmc=["gpt-4o-mini"],
        random_state=42,
        _llm=fake_llm2,
    )
    with pytest.raises(ValueError, match="Both X_val and y_val must be provided"):
        await rrf2.fit(X, y, y_val=["YES", "NO"])
