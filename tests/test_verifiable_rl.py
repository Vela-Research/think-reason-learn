"""Tests for VerifiableRL (the ``verifiable_rl`` module).

The core RL path (networks, MCTS, training loop, prediction) is fully offline
and deterministic. The optional LLM action-supervisor path is exercised with a
dedicated deterministic fake LLM (mirroring the RRM test approach) so the new
module's distinct call site is stubbed without any live calls.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Type

import numpy as np
import pytest
import torch

from think_reason_learn.core.exceptions import DataError
from think_reason_learn.core.llms import LLMResponse, OpenAIChoice
from think_reason_learn.verifiable_rl import (
    LLMActionSupervisor,
    NextActionPreference,
    QueryResult,
    VerifiableRL,
    VerifiableRLConfig,
)
from think_reason_learn.verifiable_rl._mcts import (
    ObservationState,
    compute_reward,
    policy_action_probs,
    tree_value_map,
)
from think_reason_learn.verifiable_rl._networks import Classifier, PolicyNet
from think_reason_learn.verifiable_rl._supervisor import (
    apply_action_bias,
    is_uncertain,
)

_PROVIDER = OpenAIChoice(model="gpt-4.1-nano")
_DUMMY_LLMC: List[Any] = [OpenAIChoice(model="gpt-4.1-nano")]


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _fast_config(**overrides: Any) -> VerifiableRLConfig:
    """A tiny config so fit/MCTS run quickly in tests."""
    base: dict[str, Any] = dict(
        max_steps=3,
        max_depth=2,
        n_rollouts=2,
        min_queries=1,
        n_iterations=1,
        update_every=2,
        policy_batch=8,
        clf_batch=8,
        # Tiny runs never reach the default freeze/target-sync horizons, so
        # disable them here to exercise the plain co-training path. The
        # freeze + target-network behaviour gets its own dedicated tests.
        freeze_clf_updates=0,
        clf_target_update_every=0,
    )
    base.update(overrides)
    return VerifiableRLConfig(**base)


def _toy_data(n_per_class: int = 6) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Two separable slots so the classifier has signal to learn."""
    rng = np.random.default_rng(0)
    n = n_per_class
    # class 1: high "a", class 0: low "a"; "b" is noise.
    a_pos = rng.normal(2.0, 0.3, size=(n, 3))
    a_neg = rng.normal(-2.0, 0.3, size=(n, 3))
    b = rng.normal(0.0, 0.3, size=(2 * n, 2))
    X = {
        "a": np.concatenate([a_pos, a_neg], axis=0),
        "b": b,
    }
    y = np.array([1] * n + [0] * n)
    return X, y


def _slots() -> dict[str, int]:
    return {"a": 3, "b": 2}


class FakeSupervisorLLM:
    """Deterministic stand-in for the LLM singleton for supervisor tests.

    Prefers slot ``a`` whenever ``a`` is still available (mentioned in the
    prompt's available list); otherwise prefers the first available slot.
    """

    def __init__(self) -> None:
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
        self.calls.append({"query": query, "instructions": instructions})
        # Choose a slot listed in the "Available slots" section only.
        avail_section = query.split("Available slots")[-1]
        prefer: list[str] = []
        for slot in ("a", "b", "c", "d", "e"):
            if f"- {slot}:" in avail_section:
                prefer = [slot]
                break
        return LLMResponse(
            response=NextActionPreference(prefer=prefer),
            logprobs=[("x", -0.1)],
            total_tokens=12,
            provider_model=_PROVIDER,
        )


class RecordingSupervisor:
    """Sync ActionSupervisor stub that records calls and returns a fixed slot."""

    def __init__(self, preferred: str | None = "a") -> None:
        self.preferred = preferred
        self.calls: List[tuple] = []

    def prefer(
        self, observed: set[str], available: Sequence[str], profile: str
    ) -> str | None:
        self.calls.append((set(observed), list(available), profile))
        if self.preferred in available:
            return self.preferred
        return None


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class TestNetworks:
    def test_policy_forward_single(self) -> None:
        net = PolicyNet(state_dim=5, action_dim=3, hidden=16)
        out = net(torch.zeros(5))
        assert out.shape == (3,)
        assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-5)
        assert bool((out >= 0).all())

    def test_policy_forward_batch(self) -> None:
        net = PolicyNet(state_dim=5, action_dim=3, hidden=16)
        out = net(torch.zeros(4, 5))
        assert out.shape == (4, 3)
        assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_classifier_forward_single_is_scalar(self) -> None:
        clf = Classifier(state_dim=5, hidden=8)
        out = clf(torch.zeros(5))
        assert out.shape == ()

    def test_classifier_forward_batch(self) -> None:
        clf = Classifier(state_dim=5, hidden=8)
        out = clf(torch.zeros(4, 5))
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Observation state
# ---------------------------------------------------------------------------


class TestObservationState:
    def _state(self) -> ObservationState:
        feats = {"a": np.ones(3, dtype=np.float32), "b": np.full(2, 2.0, np.float32)}
        return ObservationState(["a", "b"], {"a": 3, "b": 2}, feats)

    def test_initial_vector_is_zeros(self) -> None:
        s = self._state()
        vec = s.get_state_vector()
        assert vec.shape == (5,)
        assert np.allclose(vec, 0.0)
        assert s.n_observed() == 0

    def test_query_reveals_slot(self) -> None:
        s = self._state()
        s.query("a")
        vec = s.get_state_vector()
        assert np.allclose(vec, [1, 1, 1, 0, 0])
        assert s.observed["a"] and not s.observed["b"]
        assert s.n_observed() == 1

    def test_clone_is_independent(self) -> None:
        s = self._state()
        s.query("a")
        child = s.clone()
        child.query("b")
        assert s.n_observed() == 1
        assert child.n_observed() == 2

    def test_unknown_slot_raises(self) -> None:
        s = self._state()
        with pytest.raises(KeyError):
            s.query("nope")


# ---------------------------------------------------------------------------
# MCTS / reward
# ---------------------------------------------------------------------------


class TestMCTS:
    def _setup(
        self,
    ) -> tuple[ObservationState, PolicyNet, Classifier, VerifiableRLConfig]:
        slot_dims = {"a": 3, "b": 2}
        state_dim = 5
        action_dim = 3
        feats = {"a": np.ones(3, np.float32), "b": np.full(2, 2.0, np.float32)}
        state = ObservationState(["a", "b"], slot_dims, feats)
        policy = PolicyNet(state_dim, action_dim, hidden=16)
        clf = Classifier(state_dim, hidden=8)
        return state, policy, clf, _fast_config()

    def test_policy_action_probs_shape_and_normalised(self) -> None:
        state, policy, _, _ = self._setup()
        pi = policy_action_probs(state, policy, torch.device("cpu"))
        assert pi.shape == (3,)
        assert pytest.approx(pi.sum(), abs=1e-5) == 1.0

    def test_compute_reward_returns_sample(self) -> None:
        state, _, clf, cfg = self._setup()
        state.query("a")
        reward, sample = compute_reward(state, clf, 1, cfg, torch.device("cpu"))
        assert isinstance(reward, float)
        assert sample["y"] == 1
        assert np.asarray(sample["mask"]).shape == (2,)
        assert np.asarray(sample["x"]).shape == (5,)

    def test_tree_value_map_shapes(self) -> None:
        state, policy, clf, cfg = self._setup()
        rng = np.random.default_rng(0)
        q, dataset = tree_value_map(
            state, policy, clf, 1, cfg, torch.device("cpu"), rng
        )
        assert len(q) == 2  # one Q-value per info action (n_slots)
        assert all(isinstance(v, float) for v in q)
        assert len(dataset) > 0
        assert all("x" in s and "y" in s for s in dataset)


# ---------------------------------------------------------------------------
# Supervisor helpers (pure, no LLM)
# ---------------------------------------------------------------------------


class TestSupervisorHelpers:
    def test_is_uncertain_true_when_close(self) -> None:
        assert is_uncertain(np.array([0.30, 0.301, 0.05]), 0.01)

    def test_is_uncertain_false_when_separated(self) -> None:
        assert not is_uncertain(np.array([0.9, 0.05, 0.05]), 0.01)

    def test_apply_action_bias_preserves_stop_and_info_mass(self) -> None:
        pi = np.array([0.2, 0.2, 0.6])  # 2 info + stop
        out = apply_action_bias(
            pi, n_info=2, preferred="a", slot_names=["a", "b"], bias=0.05
        )
        assert out[2] == pytest.approx(0.6)  # stop untouched
        assert out[0] > out[1]  # preferred slot boosted
        assert out[:2].sum() == pytest.approx(0.4)  # info mass preserved

    def test_apply_action_bias_noop_when_preferred_none(self) -> None:
        pi = np.array([0.2, 0.2, 0.6])
        out = apply_action_bias(
            pi, n_info=2, preferred=None, slot_names=["a", "b"], bias=0.05
        )
        assert np.allclose(out, pi)


# ---------------------------------------------------------------------------
# LLM supervisor (fake LLM)
# ---------------------------------------------------------------------------


class TestLLMActionSupervisor:
    @pytest.mark.asyncio
    async def test_aprefer_returns_available_slot(self) -> None:
        fake = FakeSupervisorLLM()
        sup = LLMActionSupervisor(llmc=_DUMMY_LLMC, _llm=fake)
        out = await sup.aprefer(observed=set(), available=["a", "b"], profile="bio")
        assert out == "a"
        assert len(fake.calls) == 1

    @pytest.mark.asyncio
    async def test_aprefer_excludes_observed(self) -> None:
        fake = FakeSupervisorLLM()
        sup = LLMActionSupervisor(llmc=_DUMMY_LLMC, _llm=fake)
        # "a" already observed -> not offered as available -> falls to "b".
        out = await sup.aprefer(observed={"a"}, available=["b"], profile="bio")
        assert out == "b"

    @pytest.mark.asyncio
    async def test_aprefer_empty_available_returns_none(self) -> None:
        fake = FakeSupervisorLLM()
        sup = LLMActionSupervisor(llmc=_DUMMY_LLMC, _llm=fake)
        out = await sup.aprefer(observed={"a", "b"}, available=[], profile="bio")
        assert out is None
        assert len(fake.calls) == 0  # short-circuits, no call

    @pytest.mark.asyncio
    async def test_token_usage_accumulates(self) -> None:
        fake = FakeSupervisorLLM()
        sup = LLMActionSupervisor(llmc=_DUMMY_LLMC, _llm=fake)
        await sup.aprefer(observed=set(), available=["a", "b"], profile="bio")
        total = sum(tc.value for tc in sup.token_usage.token_counts.values())
        assert total > 0

    def test_prefer_sync_caches(self) -> None:
        fake = FakeSupervisorLLM()
        sup = LLMActionSupervisor(llmc=_DUMMY_LLMC, _llm=fake)
        out1 = sup.prefer(observed=set(), available=["a", "b"], profile="bio")
        out2 = sup.prefer(observed=set(), available=["a", "b"], profile="bio")
        assert out1 == out2 == "a"
        assert len(fake.calls) == 1  # cached second time


# ---------------------------------------------------------------------------
# Config / construction validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_bad_n_rollouts(self) -> None:
        with pytest.raises(ValueError, match="n_rollouts"):
            VerifiableRLConfig(n_rollouts=0)

    def test_bad_max_depth(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            VerifiableRLConfig(max_depth=0)

    def test_bad_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            VerifiableRLConfig(max_steps=0)

    def test_bad_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            VerifiableRLConfig(predict_threshold=1.5)

    def test_bad_freeze_clf_updates(self) -> None:
        with pytest.raises(ValueError, match="freeze_clf_updates"):
            VerifiableRLConfig(freeze_clf_updates=-1)

    def test_bad_clf_target_update_every(self) -> None:
        with pytest.raises(ValueError, match="clf_target_update_every"):
            VerifiableRLConfig(clf_target_update_every=-1)


class TestConstruction:
    def test_empty_slots(self) -> None:
        with pytest.raises(ValueError, match="slot"):
            VerifiableRL(slots=[])

    def test_duplicate_slots(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            VerifiableRL(slots=["a", "a"])

    def test_action_dim(self) -> None:
        m = VerifiableRL(slots=["a", "b", "c"])
        assert m.action_dim == 4  # 3 slots + STOP

    def test_dims_from_dict(self) -> None:
        m = VerifiableRL(slots={"a": 3, "b": 2})
        assert m.state_dim == 5

    def test_not_fitted(self) -> None:
        m = VerifiableRL(slots=_slots())
        assert not m.is_fitted


# ---------------------------------------------------------------------------
# fit/predict input validation
# ---------------------------------------------------------------------------


class TestFitValidation:
    def test_missing_slot(self) -> None:
        m = VerifiableRL(slots=_slots(), config=_fast_config())
        with pytest.raises(DataError, match="slot"):
            m.fit({"a": np.zeros((4, 3))}, np.array([0, 1, 0, 1]))

    def test_length_mismatch(self) -> None:
        m = VerifiableRL(slots=_slots(), config=_fast_config())
        X = {"a": np.zeros((4, 3)), "b": np.zeros((4, 2))}
        with pytest.raises(DataError, match="same"):
            m.fit(X, np.array([0, 1]))

    def test_empty(self) -> None:
        m = VerifiableRL(slots=_slots(), config=_fast_config())
        X = {"a": np.zeros((0, 3)), "b": np.zeros((0, 2))}
        with pytest.raises(DataError):
            m.fit(X, np.array([]))

    def test_bad_labels(self) -> None:
        m = VerifiableRL(slots=_slots(), config=_fast_config())
        X = {"a": np.zeros((2, 3)), "b": np.zeros((2, 2))}
        with pytest.raises(DataError, match="0/1|binary|0 and 1"):
            m.fit(X, np.array([2, 3]))

    def test_single_class(self) -> None:
        m = VerifiableRL(slots=_slots(), config=_fast_config())
        X = {"a": np.zeros((2, 3)), "b": np.zeros((2, 2))}
        with pytest.raises(DataError, match="both|two classes"):
            m.fit(X, np.array([1, 1]))

    def test_predict_before_fit(self) -> None:
        m = VerifiableRL(slots=_slots(), config=_fast_config())
        with pytest.raises(RuntimeError, match="fit"):
            m.predict({"a": np.zeros((1, 3)), "b": np.zeros((1, 2))})


# ---------------------------------------------------------------------------
# End-to-end (offline)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_fit_predict_smoke(self) -> None:
        X, y = _toy_data()
        m = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        m.fit(X, y)
        assert m.is_fitted

        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({0, 1})

        proba = m.predict_proba(X)
        assert proba.shape == (len(y),)
        assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0

    def test_predict_paths(self) -> None:
        X, y = _toy_data()
        m = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        m.fit(X, y)
        paths = m.predict_paths(X)
        assert len(paths) == len(y)
        for r in paths:
            assert isinstance(r, QueryResult)
            assert 0.0 <= r.probability <= 1.0
            assert r.prediction in (0, 1)
            assert set(r.slots_used).issubset(set(_slots()))
            # decision path is a list of slot names possibly ending in "stop"
            assert all(step in set(_slots()) | {"stop"} for step in r.decision_path)

    def test_list_input_accepted(self) -> None:
        X, y = _toy_data()
        X_list = [X["a"], X["b"]]  # aligned to slot order
        m = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        m.fit(X_list, y)
        preds = m.predict(X_list)
        assert preds.shape == (len(y),)

    def test_reproducible_with_seed(self) -> None:
        X, y = _toy_data()
        m1 = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=7)
        m1.fit(X, y)
        m2 = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=7)
        m2.fit(X, y)
        assert np.allclose(m1.predict_proba(X), m2.predict_proba(X))


# ---------------------------------------------------------------------------
# Supervisor integration with the policy
# ---------------------------------------------------------------------------


class TestSupervisorIntegration:
    def test_supervisor_consulted_during_predict(self) -> None:
        X, y = _toy_data()
        sup = RecordingSupervisor(preferred="a")
        # uncertain_delta=1.0 forces "uncertain" every step -> supervisor consulted.
        cfg = _fast_config(uncertain_delta=1.0, predict_min_queries=1)
        m = VerifiableRL(slots=_slots(), config=cfg, supervisor=sup, random_state=0)
        m.fit(X, y)
        profiles = ["bio"] * len(y)
        m.predict(X, profiles=profiles)
        assert len(sup.calls) > 0

    def test_no_supervisor_no_profiles_needed(self) -> None:
        X, y = _toy_data()
        m = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        m.fit(X, y)
        # predict without profiles must work when no supervisor is attached.
        preds = m.predict(X)
        assert preds.shape == (len(y),)


# ---------------------------------------------------------------------------
# Persistence + checkpoint loading
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: Any) -> None:
        X, y = _toy_data()
        m = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        m.fit(X, y)
        before = m.predict_proba(X)

        m.save(tmp_path)
        loaded = VerifiableRL.load(tmp_path)
        after = loaded.predict_proba(X)
        assert np.allclose(before, after)
        assert loaded.slot_names == m.slot_names

    def test_from_state_dicts(self, tmp_path: Any) -> None:
        X, y = _toy_data()
        m = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        m.fit(X, y)
        # Emulate the standalone checkpoint format.
        ckpt = tmp_path / "final_model.pt"
        torch.save(
            {
                "policy_state_dict": m._pol.state_dict(),
                "clf_state_dict": m._clf.state_dict(),
            },
            ckpt,
        )
        loaded = VerifiableRL.from_state_dicts(
            ckpt, slots=_slots(), config=_fast_config()
        )
        assert loaded.is_fitted
        preds = loaded.predict(X)
        assert preds.shape == (len(y),)


# ---------------------------------------------------------------------------
# Faithful training loop: warm-start, freeze, target network
# ---------------------------------------------------------------------------


class TestFaithfulTraining:
    def _ref_clf_state(self) -> dict[str, torch.Tensor]:
        ref = Classifier(state_dim=5, hidden=VerifiableRLConfig().clf_hidden)
        return {k: v.clone() for k, v in ref.state_dict().items()}

    def test_warm_start_loads_and_freeze_keeps_classifier(self) -> None:
        """Warm-start loads weights; a huge freeze horizon keeps them fixed."""
        X, y = _toy_data()
        sd = self._ref_clf_state()
        cfg = _fast_config(freeze_clf_updates=10**9, clf_target_update_every=0)
        m = VerifiableRL(slots=_slots(), config=cfg, random_state=0)
        m.fit(X, y, pretrained_classifier=sd)
        after = m._clf.state_dict()
        for k in sd:
            assert torch.allclose(after[k], sd[k]), f"frozen clf param {k} moved"

    def test_classifier_trains_when_not_frozen(self) -> None:
        """With freeze disabled, the warm-started classifier actually updates."""
        X, y = _toy_data()
        sd = self._ref_clf_state()
        cfg = _fast_config(freeze_clf_updates=0, clf_target_update_every=0)
        m = VerifiableRL(slots=_slots(), config=cfg, random_state=0)
        m.fit(X, y, pretrained_classifier=sd)
        after = m._clf.state_dict()
        assert any(not torch.allclose(after[k], sd[k]) for k in sd)

    def test_warm_start_from_checkpoint_file(self, tmp_path: Any) -> None:
        """Warm-start accepts a path with a {'state_dict': ...} wrapper."""
        sd = self._ref_clf_state()
        ckpt = tmp_path / "pretrained_classifier.pt"
        torch.save({"state_dict": sd}, ckpt)
        X, y = _toy_data()
        cfg = _fast_config(freeze_clf_updates=10**9, clf_target_update_every=0)
        m = VerifiableRL(slots=_slots(), config=cfg, random_state=0)
        m.fit(X, y, pretrained_classifier=ckpt)
        after = m._clf.state_dict()
        for k in sd:
            assert torch.allclose(after[k], sd[k])

    def test_target_network_active_during_fit_then_cleared(self) -> None:
        """A target net is used while training and dropped afterwards."""
        X, y = _toy_data()
        cfg = _fast_config(freeze_clf_updates=0, clf_target_update_every=2)
        m = VerifiableRL(slots=_slots(), config=cfg, random_state=0)
        m.fit(X, y)
        assert m._clf_target is None  # cleared after fit
        assert m._rollout_clf is m._clf  # falls back to the live classifier
        assert m.predict(X).shape == (len(y),)

    def test_pretrain_classifier_returns_loadable_state(self) -> None:
        """Pretraining returns a state dict that loads into a Classifier."""
        X, y = _toy_data()
        cfg = _fast_config()
        m = VerifiableRL(slots=_slots(), config=cfg, random_state=0)
        sd = m.pretrain_classifier(X, y, epochs=3, batch_size=4)
        ref = Classifier(state_dim=5, hidden=cfg.clf_hidden)
        ref.load_state_dict(sd)  # must not raise

    def test_pretrain_classifier_deterministic(self) -> None:
        """Same seed -> identical pretrained weights."""
        X, y = _toy_data()
        m1 = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        m2 = VerifiableRL(slots=_slots(), config=_fast_config(), random_state=0)
        sd1 = m1.pretrain_classifier(X, y, epochs=3, batch_size=4, random_state=1)
        sd2 = m2.pretrain_classifier(X, y, epochs=3, batch_size=4, random_state=1)
        for k in sd1:
            assert torch.allclose(sd1[k], sd2[k])

    def test_pretrain_then_fit_smoke(self) -> None:
        """The pretrained classifier warm-starts a subsequent fit."""
        X, y = _toy_data()
        cfg = _fast_config(freeze_clf_updates=0, clf_target_update_every=0)
        m = VerifiableRL(slots=_slots(), config=cfg, random_state=0)
        sd = m.pretrain_classifier(X, y, epochs=3, batch_size=4)
        m.fit(X, y, pretrained_classifier=sd)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({0, 1})
