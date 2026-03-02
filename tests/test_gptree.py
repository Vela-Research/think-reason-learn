"""Tests for GPTree class-weighted Gini and threshold predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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
