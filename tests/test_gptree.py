"""Tests for GPTree class-weighted Gini, threshold predictions, and integration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fake_llm import FakeLLM
from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.gptree import GPTree, Node

# Dummy LLM choices (never called — we test pure computation methods)
_DUMMY_LLM: list[Any] = [OpenAIChoice(model="gpt-4.1-nano")]


def _make_tree(**kwargs: Any) -> GPTree:
    """Create a GPTree with dummy LLMs and optional overrides."""
    return GPTree(
        qgen_llmc=kwargs.pop("qgen_llmc", _DUMMY_LLM),
        critic_llmc=kwargs.pop("critic_llmc", _DUMMY_LLM),
        qgen_instr_llmc=kwargs.pop("qgen_instr_llmc", _DUMMY_LLM),
        **kwargs,
    )


def _tiny_dataset() -> tuple[pd.DataFrame, list[str]]:
    """6 samples with 1 prose column and binary labels."""
    data = {
        "prose": [
            "Alice is a software engineer with 10 years experience at Google",
            "Bob is a marketing intern with no startup experience",
            "Charlie is a serial entrepreneur with 2 prior exits",
            "Diana is a recent graduate from a local college",
            "Eve is a VP of engineering at a Fortune 500 company",
            "Frank is a freelance graphic designer with varied clients",
        ]
    }
    labels = ["successful", "failed", "successful", "failed", "successful", "failed"]
    return pd.DataFrame(data), labels


def _make_integration_tree(
    fake: FakeLLM, *, tmp_path: Any = None, **kwargs: Any
) -> GPTree:
    """Create a GPTree wired to FakeLLM for integration tests."""
    defaults: dict[str, Any] = {
        "max_depth": 2,
        "min_samples_leaf": 1,
        "min_question_candidates": 2,
        "max_question_candidates": 2,
        "n_samples_as_context": 3,
        "random_state": 42,
    }
    defaults.update(kwargs)
    if tmp_path is not None:
        defaults.setdefault("save_path", str(tmp_path))
        defaults.setdefault("name", "test_tree")
    return _make_tree(_llm=fake, **defaults)


def _set_y(tree: GPTree, y: list[str]) -> None:
    """Manually set _y on a tree (bypasses full fit() flow)."""
    tree._y = np.array(y, dtype=np.str_)


def _add_node(tree: GPTree, node: Node) -> None:
    """Add a node to the tree's internal node store."""
    tree._nodes[node.id] = node


# ---------------------------------------------------------------------------
# Gini tests
# ---------------------------------------------------------------------------


class TestGiniUnweighted:
    """Test standard (unweighted) Gini computation."""

    def test_pure_node(self) -> None:
        tree = _make_tree()
        _set_y(tree, ["failed"] * 10)
        indices = np.arange(10, dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.0)

    def test_balanced_node(self) -> None:
        tree = _make_tree()
        _set_y(tree, ["failed"] * 5 + ["successful"] * 5)
        indices = np.arange(10, dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.5)

    def test_imbalanced_9010(self) -> None:
        """90/10 split → gini = 1 - (0.81 + 0.01) = 0.18."""
        tree = _make_tree()
        _set_y(tree, ["failed"] * 90 + ["successful"] * 10)
        indices = np.arange(100, dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.18)


class TestGiniWeighted:
    """Test class-weighted Gini computation."""

    def test_balanced_weight_equalises(self) -> None:
        """With balanced weights, 90/10 split should have gini ≈ 0.5.

        weight(failed) = 100 / (2*90) ≈ 0.556
        weight(successful) = 100 / (2*10) = 5.0
        weighted_counts: 90*0.556=50, 10*5.0=50
        probs: 0.5, 0.5 → gini = 0.5
        """
        tree = _make_tree(class_weight="balanced")
        _set_y(tree, ["failed"] * 90 + ["successful"] * 10)
        tree._compute_class_weights()
        indices = np.arange(100, dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.5)

    def test_custom_weights(self) -> None:
        """Custom weights: {failed: 1, successful: 9} on 90/10 split.

        weighted_counts: 90*1=90, 10*9=90
        probs: 0.5, 0.5 → gini = 0.5
        """
        tree = _make_tree(class_weight={"failed": 1.0, "successful": 9.0})
        _set_y(tree, ["failed"] * 90 + ["successful"] * 10)
        tree._compute_class_weights()
        indices = np.arange(100, dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.5)

    def test_no_weight_unchanged(self) -> None:
        """class_weight=None should produce same result as unweighted."""
        tree = _make_tree(class_weight=None)
        _set_y(tree, ["failed"] * 90 + ["successful"] * 10)
        tree._compute_class_weights()
        indices = np.arange(100, dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.18)

    def test_subset_indices(self) -> None:
        """Weighted gini on a subset uses global weights, not local counts.

        Subset: 5 failed + 5 successful
        Weights (from full 90/10): w(failed)=0.556, w(successful)=5.0
        Weighted counts: 5*0.556=2.778, 5*5.0=25.0 → total=27.778
        Probs: 0.1, 0.9 → gini = 1 - (0.01 + 0.81) = 0.18
        """
        tree = _make_tree(class_weight="balanced")
        _set_y(tree, ["failed"] * 90 + ["successful"] * 10)
        tree._compute_class_weights()
        # Pick 5 failed + 5 successful
        indices = np.array([0, 1, 2, 3, 4, 90, 91, 92, 93, 94], dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# Leaf prediction tests
# ---------------------------------------------------------------------------


def _tree_with_leaf(
    dist: dict[str, int],
    *,
    decision_threshold: float | None = None,
    root_dist: dict[str, int] | None = None,
) -> GPTree:
    """Create a tree with root (node 0) and a leaf (node 1)."""
    tree = _make_tree(decision_threshold=decision_threshold)

    root = Node(
        id=0,
        label="root",
        class_distribution=root_dist or {"failed": 91, "successful": 9},
    )
    _add_node(tree, root)

    leaf = Node(id=1, label="test_leaf", class_distribution=dist)
    _add_node(tree, leaf)

    return tree


class TestLeafProba:
    """Test get_leaf_proba."""

    def test_basic(self) -> None:
        tree = _tree_with_leaf({"failed": 7, "successful": 3})
        proba = tree.get_leaf_proba(1)
        assert proba == pytest.approx({"failed": 0.7, "successful": 0.3})

    def test_pure_leaf(self) -> None:
        tree = _tree_with_leaf({"failed": 10, "successful": 0})
        proba = tree.get_leaf_proba(1)
        assert proba == pytest.approx({"failed": 1.0, "successful": 0.0})

    def test_node_not_found(self) -> None:
        tree = _tree_with_leaf({"failed": 5, "successful": 5})
        with pytest.raises(ValueError, match="not found"):
            tree.get_leaf_proba(999)


class TestLeafPredictionMajority:
    """Test get_leaf_prediction with default majority vote."""

    def test_majority_failed(self) -> None:
        tree = _tree_with_leaf({"failed": 6, "successful": 4})
        assert tree.get_leaf_prediction(1) == "failed"

    def test_majority_successful(self) -> None:
        tree = _tree_with_leaf({"failed": 3, "successful": 7})
        assert tree.get_leaf_prediction(1) == "successful"


class TestLeafPredictionThreshold:
    """Test get_leaf_prediction with decision_threshold."""

    def test_threshold_flips_prediction(self) -> None:
        """40% success rate with threshold=0.3 → predict 'successful'."""
        tree = _tree_with_leaf(
            {"failed": 6, "successful": 4},
            decision_threshold=0.3,
        )
        assert tree.get_leaf_prediction(1) == "successful"

    def test_threshold_not_met(self) -> None:
        """10% success rate with threshold=0.3 → still 'failed'."""
        tree = _tree_with_leaf(
            {"failed": 9, "successful": 1},
            decision_threshold=0.3,
        )
        assert tree.get_leaf_prediction(1) == "failed"

    def test_threshold_exact_boundary(self) -> None:
        """30% success rate with threshold=0.3 → predict 'successful' (>=)."""
        tree = _tree_with_leaf(
            {"failed": 7, "successful": 3},
            decision_threshold=0.3,
        )
        assert tree.get_leaf_prediction(1) == "successful"

    def test_pure_leaf_with_threshold(self) -> None:
        """Pure leaf (only one class) returns that class regardless of threshold."""
        tree = _tree_with_leaf(
            {"failed": 10},
            decision_threshold=0.3,
        )
        assert tree.get_leaf_prediction(1) == "failed"

    def test_minority_detected_from_root(self) -> None:
        """Minority class is determined from root distribution."""
        tree = _tree_with_leaf(
            {"failed": 4, "successful": 6},
            decision_threshold=0.3,
            root_dist={"failed": 91, "successful": 9},
        )
        # successful is minority in root → threshold applies to successful
        # P(successful|leaf) = 0.6 >= 0.3 → predict "successful"
        assert tree.get_leaf_prediction(1) == "successful"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Test input validation for new parameters."""

    def test_invalid_class_weight_string(self) -> None:
        with pytest.raises(ValueError, match="class_weight"):
            _make_tree(class_weight="invalid")  # type: ignore

    def test_invalid_class_weight_negative(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _make_tree(class_weight={"a": -1.0})

    def test_invalid_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            _make_tree(decision_threshold=0.0)

    def test_invalid_threshold_one(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            _make_tree(decision_threshold=1.0)

    def test_invalid_threshold_negative(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            _make_tree(decision_threshold=-0.5)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Ensure defaults match original behaviour."""

    def test_defaults(self) -> None:
        tree = _make_tree()
        assert tree.class_weight is None
        assert tree.decision_threshold is None
        assert tree._class_weights is None

    def test_gini_unchanged_without_weights(self) -> None:
        """Without class_weight, gini should match the original formula."""
        tree = _make_tree()
        _set_y(tree, ["failed"] * 90 + ["successful"] * 10)
        indices = np.arange(100, dtype=np.intp)
        assert tree._gini(indices) == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# Integration tests (FakeLLM)
# ---------------------------------------------------------------------------


class TestSetTasksIntegration:
    """Test set_tasks() with FakeLLM."""

    @pytest.mark.asyncio
    async def test_generates_template_from_description(self) -> None:
        fake = FakeLLM()
        tree = _make_integration_tree(fake)
        template = await tree.set_tasks(
            task_description="Classify founders as successful or not"
        )
        assert "<number_of_questions>" in template
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_custom_template_no_llm_call(self) -> None:
        fake = FakeLLM()
        tree = _make_integration_tree(fake)
        custom = "Generate <number_of_questions> questions about startups."
        result = await tree.set_tasks(instructions_template=custom)
        assert result == custom
        assert fake.call_count == 0


class TestFitIntegration:
    """Test fit() end-to-end with FakeLLM."""

    @pytest.mark.asyncio
    async def test_fit_completes(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        nodes = []
        async for node in tree.fit(X, y):
            nodes.append(node)
        assert len(nodes) > 0
        assert tree.get_root_id() is not None
        root = tree.get_node(0)
        assert root is not None

    @pytest.mark.asyncio
    async def test_fit_produces_correct_class_distribution(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        root = tree.get_node(0)
        assert root is not None
        dist = root.class_distribution
        assert dist["successful"] == 3
        assert dist["failed"] == 3

    @pytest.mark.asyncio
    async def test_fit_at_depth_1_call_count(self, tmp_path: Any) -> None:
        """At max_depth=1, root is the only non-leaf that generates questions."""
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path, max_depth=1)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        # Calls: 1 generate_questions + (2 questions × 6 samples) answers = 13
        # But actual count depends on whether splits are valid.
        # At minimum: 1 (generate) + 12 (answers for 2 questions × 6 samples)
        assert fake.call_count >= 13

    @pytest.mark.asyncio
    async def test_leaf_nodes_have_no_question(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        for node in tree._nodes.values():
            if node.is_leaf:
                assert node.question is None

    @pytest.mark.asyncio
    async def test_fit_generates_questions_at_nodes(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        df_questions = tree.get_questions()
        assert df_questions is not None
        assert len(df_questions) >= 2  # At least root generated 2 questions


class TestPredictIntegration:
    """Test predict() after fit with FakeLLM."""

    @pytest.mark.asyncio
    async def test_predict_returns_results(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        # Predict on 2 samples
        test_samples = X.iloc[:2]
        results = []
        async for record in tree.predict(test_samples):
            results.append(record)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_predict_reaches_leaf(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        test_samples = X.iloc[:1]
        results = []
        async for record in tree.predict(test_samples):
            results.append(record)
        # Last record for a sample should be "No Question" (reached leaf)
        leaf_records = [r for r in results if r[1] == "No Question"]
        assert len(leaf_records) == 1


class TestDeterminism:
    """Test that identical inputs produce identical trees."""

    @pytest.mark.asyncio
    async def test_same_inputs_same_tree(self, tmp_path: Any) -> None:
        trees = []
        for i in range(2):
            fake = FakeLLM(questions_per_call=2)
            tree = _make_integration_tree(fake, tmp_path=tmp_path / f"run_{i}")
            await tree.set_tasks(
                instructions_template="Generate <number_of_questions> questions."
            )
            X, y = _tiny_dataset()
            async for _ in tree.fit(X, y):
                pass
            trees.append(tree)
        assert len(trees[0]._nodes) == len(trees[1]._nodes)
        assert set(trees[0]._nodes.keys()) == set(trees[1]._nodes.keys())


class TestTimingInstrumentation:
    """Test that timing stats accumulate during fit."""

    @pytest.mark.asyncio
    async def test_timing_records_calls(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        summary = tree.timing.summary()
        assert summary["generate_questions"]["count"] >= 1
        assert summary["generate_questions"]["total_s"] > 0
        assert summary["answer_question"]["count"] >= 1
        assert summary["answer_question_for_row"]["count"] >= 1
        # answer_question_for_row count should be >= answer_question count
        assert (
            summary["answer_question_for_row"]["count"]
            >= summary["answer_question"]["count"]
        )


class TestAnswerCache:
    """Test persistent answer cache."""

    def test_cache_roundtrip(self, tmp_path: Any) -> None:
        from think_reason_learn.gptree._cache import AnswerCache

        cache = AnswerCache(tmp_path / "test.db")
        assert cache.get("Q1", "Sample1", 0.0, "gemini") is None
        assert cache.misses == 1
        cache.put("Q1", "Sample1", 0.0, "gemini", "Yes")
        assert cache.get("Q1", "Sample1", 0.0, "gemini") == "Yes"
        assert cache.hits == 1
        assert len(cache) == 1
        cache.close()

    def test_different_temps_different_keys(self, tmp_path: Any) -> None:
        from think_reason_learn.gptree._cache import AnswerCache

        cache = AnswerCache(tmp_path / "test.db")
        cache.put("Q1", "S1", 0.0, "gemini", "Yes")
        cache.put("Q1", "S1", 1.0, "gemini", "No")
        assert cache.get("Q1", "S1", 0.0, "gemini") == "Yes"
        assert cache.get("Q1", "S1", 1.0, "gemini") == "No"
        cache.close()

    def test_different_models_different_keys(self, tmp_path: Any) -> None:
        from think_reason_learn.gptree._cache import AnswerCache

        cache = AnswerCache(tmp_path / "test.db")
        cache.put("Q1", "S1", 0.0, "gemini-flash", "Yes")
        cache.put("Q1", "S1", 0.0, "gpt-4o", "No")
        assert cache.get("Q1", "S1", 0.0, "gemini-flash") == "Yes"
        assert cache.get("Q1", "S1", 0.0, "gpt-4o") == "No"
        cache.close()

    @pytest.mark.asyncio
    async def test_cache_enabled_during_fit(self, tmp_path: Any) -> None:
        """Fit with cache enabled populates the cache."""
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path, use_answer_cache=True)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass
        assert tree._answer_cache is not None
        assert len(tree._answer_cache) > 0
        # All entries were misses on first run
        assert tree._answer_cache.misses > 0
        assert tree._answer_cache.hits == 0

    @pytest.mark.asyncio
    async def test_cache_disabled_by_default(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        assert tree._answer_cache is None


class TestSaveLoadIntegration:
    """Test save/load round-trip with FakeLLM-built tree."""

    @pytest.mark.asyncio
    async def test_save_load_preserves_structure(self, tmp_path: Any) -> None:
        fake = FakeLLM(questions_per_call=2)
        tree = _make_integration_tree(fake, tmp_path=tmp_path)
        await tree.set_tasks(
            instructions_template="Generate <number_of_questions> questions."
        )
        X, y = _tiny_dataset()
        async for _ in tree.fit(X, y):
            pass

        original_nodes = len(tree._nodes)
        original_root_dist = tree.get_node(0).class_distribution  # type: ignore

        # Save and reload
        tree.save()
        loaded = GPTree.load(tree.save_path / tree.name)

        assert len(loaded._nodes) == original_nodes
        assert loaded.get_node(0).class_distribution == original_root_dist  # type: ignore
