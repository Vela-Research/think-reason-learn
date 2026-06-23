"""Verifiable RL.

An adaptive information-gathering classifier: instead of consuming every feature
up front, a learned policy network decides, slot by slot, which piece of
information to reveal next (or to STOP), and a classifier predicts the binary
label from the accumulated partial state. The policy is trained on soft targets
derived from Monte-Carlo tree-search rollouts; an optional LLM supervisor can
nudge the policy toward an informative slot when it is undecided.

This is a library port of the standalone VCBench reference implementation. The
algorithm (networks, MCTS targets, two-network training loop, optional LLM
supervision) is preserved; the run-script scaffolding (CLI, CSV/`.npy` I/O,
plotting, baselines, multi-seed drivers, and the VCBench-specific founder data
store) is intentionally omitted. The caller supplies generic, named information
*slots* as arrays and binary 0/1 labels.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Self

import numpy as np
import orjson
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from think_reason_learn.core.exceptions import DataError

from ._mcts import (
    ObservationState,
    compute_reward,
    policy_action_probs,
    tree_value_map,
)
from ._networks import Classifier, PolicyNet
from ._supervisor import ActionSupervisor, apply_action_bias, is_uncertain
from ._types import QueryResult, VerifiableRLConfig

logger = logging.getLogger(__name__)

_SlotsArg = Sequence[str] | Mapping[str, int]
_FeaturesArg = Mapping[str, np.ndarray] | Sequence[np.ndarray]


class VerifiableRL:
    """Adaptive, sequential information-gathering binary classifier.

    Args:
        slots: Either an ordered sequence of slot names (feature dimensions are
            inferred at ``fit``) or a mapping of ``name -> dimension`` (lets
            :attr:`state_dim` be known before fitting and enables
            :meth:`from_state_dicts`).
        config: Algorithm configuration. Defaults to ``VerifiableRLConfig()``.
        supervisor: Optional :class:`ActionSupervisor` consulted when the policy
            is undecided. Requires per-sample ``profiles`` at fit/predict time.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
        random_state: Seed for reproducible network init and rollouts.
    """

    def __init__(
        self,
        slots: _SlotsArg,
        config: VerifiableRLConfig | None = None,
        supervisor: ActionSupervisor | None = None,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        slot_names, slot_dims = self._verify_slots(slots)
        self.slot_names: List[str] = slot_names
        self.slot_dims: Dict[str, int] | None = slot_dims
        self.config = config if config is not None else VerifiableRLConfig()
        self.supervisor = supervisor
        self.device = device
        self.random_state = random_state

        self._policy: PolicyNet | None = None
        self._classifier: Classifier | None = None
        # Frozen target classifier used by the MCTS rollouts/rewards during
        # training (synced from the live classifier every N updates). ``None``
        # outside ``fit`` and when ``clf_target_update_every == 0``.
        self._clf_target: Classifier | None = None
        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._fitted = False

    # ------------------------------------------------------------------
    # Construction helpers / properties
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_slots(slots: _SlotsArg) -> tuple[List[str], Dict[str, int] | None]:
        if isinstance(slots, Mapping):
            names = list(slots.keys())
            dims: Dict[str, int] | None = {str(k): int(v) for k, v in slots.items()}
        else:
            names = list(slots)
            dims = None
        if len(names) == 0:
            raise ValueError("slots must define at least one information slot")
        if len(set(names)) != len(names):
            raise ValueError("slot names must be unique")
        if dims is not None and any(d < 1 for d in dims.values()):
            raise ValueError("slot dimensions must be >= 1")
        return names, dims

    @property
    def action_dim(self) -> int:
        """Number of actions: one per slot plus STOP."""
        return len(self.slot_names) + 1

    @property
    def state_dim(self) -> int | None:
        """Length of the state vector (``None`` until slot dims are known)."""
        if self.slot_dims is None:
            return None
        return sum(self.slot_dims[s] for s in self.slot_names)

    @property
    def is_fitted(self) -> bool:
        """Whether the model has trained or loaded weights."""
        return self._fitted

    @property
    def _pol(self) -> PolicyNet:
        """The policy network (raises if networks are not built yet)."""
        if self._policy is None:
            raise RuntimeError("networks are not built; call fit() first")
        return self._policy

    @property
    def _clf(self) -> Classifier:
        """The classifier network (raises if networks are not built yet)."""
        if self._classifier is None:
            raise RuntimeError("networks are not built; call fit() first")
        return self._classifier

    @property
    def _rollout_clf(self) -> Classifier:
        """Classifier the MCTS rollouts/rewards query during training.

        Returns the frozen *target* classifier when one is active, otherwise the
        live classifier. Using a slowly-updated target keeps the policy's value
        targets stable (matching the standalone reference).
        """
        return self._clf_target if self._clf_target is not None else self._clf

    def _build_networks(self) -> None:
        assert self.slot_dims is not None
        state_dim = self.state_dim
        assert state_dim is not None
        device = torch.device(self.device)
        self._policy = PolicyNet(
            state_dim, self.action_dim, hidden=self.config.policy_hidden
        ).to(device)
        self._classifier = Classifier(state_dim, hidden=self.config.clf_hidden).to(
            device
        )

    # ------------------------------------------------------------------
    # Input handling / validation
    # ------------------------------------------------------------------

    def _coerce_x(self, X: _FeaturesArg) -> Dict[str, np.ndarray]:
        if isinstance(X, Mapping):
            missing = [s for s in self.slot_names if s not in X]
            if missing:
                raise DataError(f"X is missing slot(s): {missing}")
            return {s: np.asarray(X[s]) for s in self.slot_names}
        if isinstance(X, (list, tuple)):
            if len(X) != len(self.slot_names):
                raise DataError(
                    f"X has {len(X)} arrays but there are {len(self.slot_names)} slots"
                )
            return {s: np.asarray(X[k]) for k, s in enumerate(self.slot_names)}
        raise DataError("X must be a dict of slot->array or a list/tuple of arrays")

    def _check_dims(self, Xd: Dict[str, np.ndarray]) -> None:
        for s in self.slot_names:
            if Xd[s].ndim != 2:
                raise DataError(f"slot '{s}' must be 2-D (n_samples, dim)")
        counts = {Xd[s].shape[0] for s in self.slot_names}
        if len(counts) != 1:
            raise DataError("all slots must have the same number of samples")

    def _validate_fit(self, Xd: Dict[str, np.ndarray], y: Any) -> np.ndarray:
        self._check_dims(Xd)
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise DataError("y must be 1-D")
        n = Xd[self.slot_names[0]].shape[0]
        if n != len(y_arr):
            raise DataError("X and y must have the same number of samples")
        if n == 0:
            raise DataError("X and y must be non-empty")
        uniq = set(int(v) for v in np.unique(y_arr))
        if not uniq.issubset({0, 1}):
            raise DataError("y must be binary (values 0/1)")
        if uniq != {0, 1}:
            raise DataError("y must contain both classes (0 and 1)")
        return y_arr.astype(int)

    def _validate_predict(
        self, Xd: Dict[str, np.ndarray], profiles: Sequence[str] | None
    ) -> int:
        self._check_dims(Xd)
        n = Xd[self.slot_names[0]].shape[0]
        if n == 0:
            raise DataError("X must be non-empty")
        if self.slot_dims is not None:
            inferred = {s: int(Xd[s].shape[1]) for s in self.slot_names}
            if inferred != self.slot_dims:
                raise DataError(
                    f"slot dims {inferred} do not match the model's {self.slot_dims}"
                )
        if profiles is not None and len(profiles) != n:
            raise DataError("profiles must have the same length as X")
        return n

    def _make_state(self, Xd: Dict[str, np.ndarray], i: int) -> ObservationState:
        assert self.slot_dims is not None
        features = {s: Xd[s][i] for s in self.slot_names}
        return ObservationState(self.slot_names, self.slot_dims, features)

    def _check_fitted(self) -> None:
        if not self._fitted or self._policy is None or self._classifier is None:
            raise RuntimeError("model is not fitted; call fit() first")

    # ------------------------------------------------------------------
    # Action selection (shared by training rollouts and prediction)
    # ------------------------------------------------------------------

    def _biased_probs(self, state: ObservationState, profile: str | None) -> np.ndarray:
        device = torch.device(self.device)
        pi = policy_action_probs(state, self._pol, device)
        n_info = len(self.slot_names)
        if self.supervisor is None or profile is None:
            return pi
        if not is_uncertain(pi[:n_info], self.config.uncertain_delta):
            return pi
        observed = {s for s in self.slot_names if state.observed[s]}
        available = [s for s in self.slot_names if not state.observed[s]]
        if not available:
            return pi
        preferred = self.supervisor.prefer(observed, available, profile)
        return apply_action_bias(
            pi, n_info, preferred, self.slot_names, self.config.llm_bias
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: _FeaturesArg,
        y: Any,
        *,
        profiles: Sequence[str] | None = None,
        pretrained_classifier: str | PathLike[str] | Mapping[str, Any] | None = None,
    ) -> Self:
        """Train the policy and classifier on partially observable episodes.

        Mirrors the standalone two-network training loop: an optional classifier
        warm-start, a freeze period during which only the policy trains against
        the (fixed) classifier, and a slowly-synced *target* classifier that the
        MCTS rollouts query so the policy's value targets stay stable.

        Args:
            X: Slot features (dict ``name -> (n, dim)`` array, or a list of
                arrays aligned to ``slots``).
            y: Binary labels (length ``n``, values in ``{0, 1}``).
            profiles: Optional per-sample text used by the LLM supervisor.
            pretrained_classifier: Optional warm-start for the classifier — a
                path to a checkpoint (``{"state_dict": ...}``,
                ``{"clf_state_dict": ...}``, or a raw state dict) or an
                in-memory state dict. Strongly recommended for faithful results:
                the MCTS reward depends on the classifier, so starting it from a
                pretrained model (rather than random) is what lets the policy
                learn useful queries (matches the reference implementation).

        Returns:
            ``self``.
        """
        Xd = self._coerce_x(X)
        y_arr = self._validate_fit(Xd, y)
        n = len(y_arr)
        if profiles is not None and len(profiles) != n:
            raise DataError("profiles must have the same length as y")

        inferred = {s: int(Xd[s].shape[1]) for s in self.slot_names}
        if self.slot_dims is not None and self.slot_dims != inferred:
            raise DataError(
                f"slot dims {inferred} do not match configured {self.slot_dims}"
            )
        self.slot_dims = inferred

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            self._rng = np.random.default_rng(self.random_state)
        self._build_networks()

        device = torch.device(self.device)
        cfg = self.config

        # Optional classifier warm-start. The MCTS reward queries the
        # classifier, so starting it from a pretrained model (rather than
        # random) is what makes the policy's value targets meaningful.
        if pretrained_classifier is not None:
            self._load_pretrained_classifier(pretrained_classifier, device)

        # Target classifier the rollouts query (synced from the live classifier
        # every ``clf_target_update_every`` updates). Disabled when 0.
        if cfg.clf_target_update_every > 0:
            target = Classifier(int(self.state_dim or 0), hidden=cfg.clf_hidden)
            target = target.to(device)
            target.load_state_dict(self._clf.state_dict())
            target.eval()
            for p in target.parameters():
                p.requires_grad_(False)
            self._clf_target = target
        else:
            self._clf_target = None

        policy_opt = torch.optim.Adam(self._pol.parameters(), lr=cfg.policy_lr)
        clf_opt = torch.optim.Adam(self._clf.parameters(), lr=cfg.clf_lr)

        spv_buf: List[Dict[str, Any]] = []
        clf_buf: List[Dict[str, Any]] = []
        n_updates = 0

        def _do_update() -> None:
            nonlocal n_updates
            n_updates += 1
            self._train_policy(
                self._sample(spv_buf, cfg.policy_sample), policy_opt, device
            )
            # Keep the classifier fixed for the first ``freeze_clf_updates``
            # updates so the policy adapts to the (warm-started) classifier first.
            if n_updates >= cfg.freeze_clf_updates:
                self._train_classifier(
                    self._sample(clf_buf, cfg.clf_sample), clf_opt, device
                )
            # Periodically refresh the target classifier from the live one.
            if (
                self._clf_target is not None
                and n_updates % cfg.clf_target_update_every == 0
            ):
                self._clf_target.load_state_dict(self._clf.state_dict())
                self._clf_target.eval()

        for _ in range(cfg.n_iterations):
            order = self._rng.permutation(n)
            for count, i in enumerate(order):
                state = self._make_state(Xd, int(i))
                profile = profiles[int(i)] if profiles is not None else None
                spv, samples = self._run_episode(state, int(y_arr[i]), profile)
                spv_buf.extend(spv)
                clf_buf.extend(samples)
                spv_buf = spv_buf[-cfg.policy_replay_max :]
                clf_buf = clf_buf[-cfg.clf_replay_max :]
                if (count + 1) % cfg.update_every == 0:
                    _do_update()
            # end-of-iteration flush
            _do_update()

        # The target classifier is a training-time aid only; drop it so the
        # fitted model uses the live classifier for prediction/persistence.
        self._clf_target = None
        self._fitted = True
        return self

    def _load_pretrained_classifier(
        self,
        src: str | PathLike[str] | Mapping[str, Any],
        device: torch.device,
    ) -> None:
        """Warm-start the classifier from a checkpoint or in-memory state dict.

        Accepts a path or mapping; unwraps a ``state_dict``/``clf_state_dict``
        wrapper if present, then loads into the live classifier.
        """
        if isinstance(src, Mapping):
            ckpt: Any = src
        else:
            ckpt = torch.load(Path(src), map_location=device, weights_only=True)
        state = ckpt
        for key in ("state_dict", "clf_state_dict"):
            if isinstance(state, Mapping) and key in state:
                state = state[key]
                break
        self._clf.load_state_dict(state)
        self._clf.train()

    # ------------------------------------------------------------------
    # Classifier pretraining (curriculum-masked) — standalone parity
    # ------------------------------------------------------------------

    def _default_curriculum(self) -> List[Dict[str, Any]]:
        """Full -> medium -> light reveal schedule, derived from #slots."""
        n = len(self.slot_names)
        return [
            {"min_k": n, "max_k": n, "frac": 0.3},
            {"min_k": 2, "max_k": max(2, n - 1), "frac": 0.4},
            {"min_k": 1, "max_k": min(2, n), "frac": 0.3},
        ]

    @staticmethod
    def _curriculum_phase(
        phases: Sequence[Mapping[str, Any]], epoch: int, total_epochs: int
    ) -> Mapping[str, Any]:
        t = epoch / max(total_epochs, 1)
        acc = 0.0
        for phase in phases:
            acc += float(phase["frac"])
            if t <= acc:
                return phase
        return phases[-1]

    @staticmethod
    def _clf_step(
        clf: Classifier,
        opt: torch.optim.Optimizer,
        criterion: nn.Module,
        bx: List[np.ndarray],
        by: List[float],
        device: torch.device,
    ) -> None:
        xb = torch.from_numpy(np.stack(bx)).float().to(device)
        yb = torch.tensor(by, dtype=torch.float32).to(device)
        loss = criterion(clf(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def pretrain_classifier(
        self,
        X: _FeaturesArg,
        y: Any,
        *,
        epochs: int = 50,
        lr: float = 1e-5,
        batch_size: int = 16,
        curriculum: Sequence[Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Pretrain a classifier on randomly-masked partial states.

        Faithfully mirrors the standalone ``pretrain_classifier.py`` curriculum:
        early epochs reveal all slots, later epochs reveal progressively fewer,
        so the classifier learns to predict from *partial* information. The
        returned ``state_dict`` is meant to warm-start training via
        ``fit(pretrained_classifier=...)`` — which is what makes the MCTS reward
        (and therefore the learned policy) meaningful.

        This does not mutate the model; it returns weights to pass to ``fit``.
        For a leakage-free cross-validation, call this on each fold's training
        data only.

        Args:
            X: Slot features (dict ``name -> (n, dim)`` or list aligned to slots).
            y: Binary labels (length ``n``, values in ``{0, 1}``).
            epochs: Number of passes over the data.
            lr: Adam learning rate.
            batch_size: Mini-batch size.
            curriculum: Optional list of ``{"min_k", "max_k", "frac"}`` phases
                (``frac`` should sum to ~1). Defaults to a full->medium->light
                schedule derived from the number of slots.
            random_state: Seed for masking/shuffling (falls back to the
                instance's ``random_state``).

        Returns:
            A classifier ``state_dict`` of CPU tensors.
        """
        Xd = self._coerce_x(X)
        y_arr = self._validate_fit(Xd, y)
        n = len(y_arr)
        inferred = {s: int(Xd[s].shape[1]) for s in self.slot_names}
        if self.slot_dims is not None and self.slot_dims != inferred:
            raise DataError(
                f"slot dims {inferred} do not match configured {self.slot_dims}"
            )
        self.slot_dims = inferred

        seed = random_state if random_state is not None else self.random_state
        if seed is not None:
            torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        device = torch.device(self.device)
        state_dim = self.state_dim
        assert state_dim is not None
        clf = Classifier(state_dim, hidden=self.config.clf_hidden).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        phases = (
            list(curriculum) if curriculum is not None else self._default_curriculum()
        )
        n_slots = len(self.slot_names)

        clf.train()
        for epoch in range(epochs):
            phase = self._curriculum_phase(phases, epoch, epochs)
            min_k = max(1, min(int(phase["min_k"]), n_slots))
            max_k = max(min_k, min(int(phase["max_k"]), n_slots))
            bx: List[np.ndarray] = []
            by: List[float] = []
            for i in rng.permutation(n):
                k = int(rng.integers(min_k, max_k + 1))
                reveal = rng.choice(n_slots, size=k, replace=False)
                state = self._make_state(Xd, int(i))
                for j in reveal:
                    state.query(self.slot_names[int(j)])
                bx.append(state.get_state_vector())
                by.append(float(y_arr[int(i)]))
                if len(bx) == batch_size:
                    self._clf_step(clf, opt, criterion, bx, by, device)
                    bx, by = [], []
            if bx:
                self._clf_step(clf, opt, criterion, bx, by, device)

        return {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    def _run_episode(
        self, state: ObservationState, label: int, profile: str | None
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cfg = self.config
        n_info = len(self.slot_names)
        spv_list: List[Dict[str, Any]] = []
        sample_list: List[Dict[str, Any]] = []

        for _ in range(cfg.max_steps):
            pi = self._biased_probs(state, profile)
            if self._rng.random() < cfg.eps:
                action = int(self._rng.integers(len(pi)))
            else:
                action = int(np.argmax(pi))

            if action == n_info:  # STOP
                break
            state.query(self.slot_names[action])
            spv, samples = self._compute_tree_targets(state, label)
            spv_list.append(spv)
            sample_list.extend(samples)

        return spv_list, sample_list

    def _compute_tree_targets(
        self, state: ObservationState, label: int
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        cfg = self.config
        device = torch.device(self.device)
        state_vec = state.get_state_vector()

        x = torch.from_numpy(state_vec).float().to(device)
        with torch.no_grad():
            pi_root = self._pol(x).detach().cpu().numpy()

        rollout_clf = self._rollout_clf
        q_info, dataset = tree_value_map(
            state,
            self._pol,
            rollout_clf,
            label,
            cfg,
            device,
            self._rng,
        )
        q_info_arr = np.asarray(q_info, dtype=np.float32)

        if state.n_observed() < cfg.min_queries:
            q_stop = -1e9
        else:
            q_stop, _ = compute_reward(state, rollout_clf, label, cfg, device)
        q_stop = float(q_stop)

        q_all = np.concatenate([q_info_arr / cfg.tau_info, [q_stop / cfg.tau_stop]])
        p_info = torch.softmax(torch.from_numpy(q_info_arr / cfg.tau_info), dim=0)
        p_stop = torch.sigmoid(torch.tensor(q_stop / cfg.tau_stop))
        p_all = torch.cat([p_info, p_stop.view(1)], dim=0)
        pai = (p_all / p_all.sum()).numpy().astype(np.float32)

        v = float(np.dot(pi_root[: len(q_all)], q_all))
        spv = {"s": state_vec.copy(), "pai": pai, "v": v}
        return spv, dataset

    def _sample(self, buf: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if k <= 0 or len(buf) <= k:
            return list(buf)
        idx = self._rng.choice(len(buf), size=k, replace=False)
        return [buf[i] for i in idx]

    def _train_policy(
        self,
        spv: List[Dict[str, Any]],
        opt: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        if not spv:
            return 0.0
        cfg = self.config
        states = torch.from_numpy(np.stack([d["s"] for d in spv])).float()
        targets = torch.from_numpy(np.stack([d["pai"] for d in spv])).float()
        loader = DataLoader(
            TensorDataset(states, targets), batch_size=cfg.policy_batch, shuffle=True
        )
        policy = self._pol
        policy.train()
        total, batches = 0.0, 0
        for _ in range(cfg.train_epochs):
            for xb, pi_tgt in loader:
                xb, pi_tgt = xb.to(device), pi_tgt.to(device)
                pi_pred = policy(xb)
                pi_pred = pi_pred[:, : pi_tgt.size(1)]
                loss = -(pi_tgt * torch.log(pi_pred + 1e-8)).sum(dim=1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                opt.step()
                total += float(loss.item())
                batches += 1
        return total / max(batches, 1)

    def _train_classifier(
        self,
        samples: List[Dict[str, Any]],
        opt: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        if not samples:
            return 0.0
        cfg = self.config
        xs = torch.from_numpy(np.stack([s["x"] for s in samples])).float()
        ys = torch.tensor([int(s["y"]) for s in samples], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(xs, ys), batch_size=cfg.clf_batch, shuffle=True
        )
        criterion = nn.BCEWithLogitsLoss()
        classifier = self._clf
        classifier.train()
        total, batches = 0.0, 0
        for _ in range(cfg.train_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = classifier(xb)
                loss = criterion(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(classifier.parameters(), cfg.grad_clip)
                opt.step()
                total += float(loss.item())
                batches += 1
        return total / max(batches, 1)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_one(
        self, state: ObservationState, profile: str | None
    ) -> tuple[float, List[str], List[str]]:
        cfg = self.config
        device = torch.device(self.device)
        n_info = len(self.slot_names)
        queried = 0
        used: List[str] = []
        path: List[str] = []

        for _ in range(cfg.max_steps):
            pi = self._biased_probs(state, profile)
            if cfg.greedy:
                action = int(np.argmax(pi))
            else:
                p = pi / pi.sum()
                action = int(self._rng.choice(len(p), p=p))

            if action == n_info and queried < cfg.predict_min_queries:
                pi = pi.copy()
                pi[n_info] = -1.0
                action = int(np.argmax(pi))

            if action == n_info:
                path.append("stop")
                break

            slot = self.slot_names[action]
            path.append(slot)
            if not state.observed[slot]:
                used.append(slot)
            state.query(slot)
            queried += 1

        x = torch.from_numpy(state.get_state_vector()).float().to(device)
        with torch.no_grad():
            prob = float(torch.sigmoid(self._clf(x)).item())
        return prob, used, path

    def predict_paths(
        self, X: _FeaturesArg, *, profiles: Sequence[str] | None = None
    ) -> List[QueryResult]:
        """Run the policy on each sample and return full decision traces."""
        self._check_fitted()
        Xd = self._coerce_x(X)
        n = self._validate_predict(Xd, profiles)
        self._pol.eval()
        self._clf.eval()

        results: List[QueryResult] = []
        for i in range(n):
            state = self._make_state(Xd, i)
            profile = profiles[i] if profiles is not None else None
            prob, used, path = self._predict_one(state, profile)
            pred = int(prob >= self.config.predict_threshold)
            results.append(
                QueryResult(
                    probability=prob,
                    prediction=pred,
                    slots_used=used,
                    decision_path=path,
                )
            )
        return results

    def predict_proba(
        self, X: _FeaturesArg, *, profiles: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return per-sample success probabilities (shape ``(n,)``)."""
        return np.array(
            [r.probability for r in self.predict_paths(X, profiles=profiles)]
        )

    def predict(
        self, X: _FeaturesArg, *, profiles: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return per-sample binary predictions (shape ``(n,)``)."""
        return np.array(
            [r.prediction for r in self.predict_paths(X, profiles=profiles)]
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | PathLike[str]) -> None:
        """Save weights + config to a directory."""
        self._check_fitted()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self._pol.state_dict(),
                "clf_state_dict": self._clf.state_dict(),
            },
            out / "model.pt",
        )
        meta = {
            "slot_names": self.slot_names,
            "slot_dims": self.slot_dims,
            "config": asdict(self.config),
            "device": self.device,
            "random_state": self.random_state,
        }
        (out / "meta.json").write_bytes(orjson.dumps(meta))

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        supervisor: ActionSupervisor | None = None,
        device: str | None = None,
    ) -> "VerifiableRL":
        """Load a model saved with :meth:`save`."""
        src = Path(path)
        meta = orjson.loads((src / "meta.json").read_bytes())
        cfg = VerifiableRLConfig(**meta["config"])
        slots = {s: meta["slot_dims"][s] for s in meta["slot_names"]}
        model = cls(
            slots=slots,
            config=cfg,
            supervisor=supervisor,
            device=device or meta["device"],
            random_state=meta["random_state"],
        )
        model._build_networks()
        ckpt = torch.load(
            src / "model.pt",
            map_location=torch.device(model.device),
            weights_only=True,
        )
        model._pol.load_state_dict(ckpt["policy_state_dict"])
        model._clf.load_state_dict(ckpt["clf_state_dict"])
        model._pol.eval()
        model._clf.eval()
        model._fitted = True
        return model

    @classmethod
    def from_state_dicts(
        cls,
        path: str | PathLike[str],
        slots: Mapping[str, int],
        *,
        config: VerifiableRLConfig | None = None,
        supervisor: ActionSupervisor | None = None,
        device: str = "cpu",
    ) -> "VerifiableRL":
        """Load a raw checkpoint of ``policy_state_dict`` + ``clf_state_dict``.

        This consumes the standalone ``runs/model_*/final_model.pt`` format.
        ``slots`` must be a ``name -> dimension`` mapping (the dimensions are
        required to rebuild the networks) whose total matches the checkpoint's
        ``state_dim`` and whose count matches ``action_dim - 1``.
        """
        if not isinstance(slots, Mapping):
            raise ValueError("slots must be a name->dim mapping for from_state_dicts")
        model = cls(
            slots=slots,
            config=config,
            supervisor=supervisor,
            device=device,
        )
        model._build_networks()
        ckpt = torch.load(
            Path(path), map_location=torch.device(model.device), weights_only=True
        )
        policy_sd = ckpt.get("policy_state_dict", ckpt.get("policy"))
        clf_sd = ckpt.get("clf_state_dict", ckpt.get("classifier", ckpt.get("clf")))
        if policy_sd is None or clf_sd is None:
            raise DataError(
                "checkpoint must contain 'policy_state_dict' and 'clf_state_dict'"
            )
        model._pol.load_state_dict(policy_sd)
        model._clf.load_state_dict(clf_sd)
        model._pol.eval()
        model._clf.eval()
        model._fitted = True
        return model
