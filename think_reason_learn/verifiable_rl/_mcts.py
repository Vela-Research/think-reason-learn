"""Observation state and Monte-Carlo tree-value estimation for Verifiable RL.

This is the generic port of the standalone ``get_observation.py`` (state) and
``Tree_value_map.py`` (MCTS). The VCBench-specific slot names / data store are
replaced by a generic, caller-defined set of named information *slots*: a state
is the concatenation of revealed slot vectors, with unrevealed slots zeroed.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from ._networks import Classifier, PolicyNet
from ._types import VerifiableRLConfig


class ObservationState:
    """A partially revealed sample over a fixed set of named slots.

    Querying a slot marks it observed; the state vector concatenates the
    per-slot feature vectors in slot order, substituting zeros for any slot not
    yet observed.
    """

    def __init__(
        self,
        slot_names: Sequence[str],
        slot_dims: Mapping[str, int],
        features: Mapping[str, np.ndarray],
    ) -> None:
        """Build a state for one sample.

        Args:
            slot_names: Ordered slot names (defines the action/vector layout).
            slot_dims: Map of slot name -> feature dimension.
            features: Map of slot name -> this sample's 1-D feature vector.
        """
        self.slot_names = list(slot_names)
        self.slot_dims = dict(slot_dims)
        self.features = features
        self.observed: Dict[str, bool] = {s: False for s in self.slot_names}

    def query(self, slot: str) -> None:
        """Mark ``slot`` as observed.

        Raises:
            KeyError: If ``slot`` is not a known slot name.
        """
        if slot not in self.observed:
            raise KeyError(f"Unknown slot: {slot}")
        self.observed[slot] = True

    def n_observed(self) -> int:
        """Return the number of revealed slots."""
        return sum(self.observed.values())

    def get_state_vector(self) -> np.ndarray:
        """Return the concatenated state vector (unobserved slots zeroed)."""
        parts: List[np.ndarray] = []
        for s in self.slot_names:
            if self.observed[s]:
                parts.append(np.asarray(self.features[s], dtype=np.float32))
            else:
                parts.append(np.zeros(self.slot_dims[s], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def clone(self) -> "ObservationState":
        """Return a copy with independent ``observed`` flags (shared features)."""
        child = ObservationState(self.slot_names, self.slot_dims, self.features)
        child.observed = dict(self.observed)
        return child


def policy_action_probs(
    state: ObservationState, policy: PolicyNet, device: torch.device
) -> np.ndarray:
    """Return the policy's action distribution for ``state`` as a numpy array."""
    x = torch.from_numpy(state.get_state_vector()).float().to(device)
    with torch.no_grad():
        pi_t = policy(x)
    pi = pi_t.detach().cpu().numpy().astype(np.float64)
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    total = pi.sum()
    if total <= 0:
        pi[:] = 1.0 / len(pi)
    else:
        pi /= total
    return pi


def compute_reward(
    state: ObservationState,
    classifier: Classifier,
    label: int,
    config: VerifiableRLConfig,
    device: torch.device,
) -> Tuple[float, Dict[str, object]]:
    """Score a terminal state and emit a classifier training sample.

    Returns:
        A ``(reward, sample)`` pair. ``sample`` carries the state vector, label,
        and observed-slot mask for classifier training.
    """
    state_vec = state.get_state_vector()
    x = torch.from_numpy(state_vec).float().to(device)
    with torch.no_grad():
        logits = classifier(x)
        prob = torch.sigmoid(logits)
        pred_label = int((prob >= config.clf_threshold).item())

    label = int(label)
    if pred_label == 1 and label == 1:
        reward = config.reward_tp
    elif pred_label == 1 and label == 0:
        reward = config.reward_fp
    elif pred_label == 0 and label == 0:
        reward = config.reward_tn
    else:  # pred_label == 0 and label == 1
        reward = config.reward_fn

    mask = np.array(
        [1 if state.observed[s] else 0 for s in state.slot_names], dtype=np.int64
    )
    sample: Dict[str, object] = {"x": state_vec.copy(), "y": label, "mask": mask}
    return float(reward), sample


def _rollout_once(
    start_state: ObservationState,
    policy: PolicyNet,
    classifier: Classifier,
    label: int,
    config: VerifiableRLConfig,
    device: torch.device,
    dataset: List[Dict[str, object]],
    rng: np.random.Generator,
    depth_start: int = 0,
) -> float:
    """Sample one rollout to termination; append the terminal classifier sample."""
    state = start_state.clone()
    depth = depth_start
    repeat_penalty_sum = 0.0
    n_info = len(state.slot_names)

    while True:
        if depth >= config.max_depth:
            reward, sample = compute_reward(state, classifier, label, config, device)
            reward = reward + config.step_penalty * depth + repeat_penalty_sum
            dataset.append(sample)
            return float(reward)

        pi = policy_action_probs(state, policy, device)
        action = int(rng.choice(len(pi), p=pi))

        if action == n_info:  # STOP
            reward, sample = compute_reward(state, classifier, label, config, device)
            reward = reward + config.step_penalty * depth + repeat_penalty_sum
            dataset.append(sample)
            return float(reward)

        slot = state.slot_names[action]
        if state.observed[slot]:  # repeated query yields no new info
            repeat_penalty_sum += config.repeat_penalty
            depth += 1
            continue

        state.query(slot)
        depth += 1


def tree_value_map(
    root_state: ObservationState,
    policy: PolicyNet,
    classifier: Classifier,
    label: int,
    config: VerifiableRLConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[List[float], List[Dict[str, object]]]:
    """Estimate Q-values for each info action via Monte-Carlo rollouts.

    For each information slot ``a``: reveal it, then roll out the policy
    ``n_rollouts`` times and average the returns. STOP is handled separately by
    the caller.

    Returns:
        A ``(Q, dataset)`` pair: ``Q`` has one value per info action;
        ``dataset`` aggregates the terminal classifier samples from all rollouts.
    """
    dataset: List[Dict[str, object]] = []
    pi0 = policy_action_probs(root_state, policy, device)
    n_info = len(pi0) - 1

    q_values: List[float] = [0.0] * n_info
    for action in range(n_info):
        child = root_state.clone()
        slot = child.slot_names[action]
        init_repeat = config.repeat_penalty if child.observed[slot] else 0.0
        if not child.observed[slot]:
            child.query(slot)

        returns: List[float] = []
        for _ in range(int(config.n_rollouts)):
            g = _rollout_once(
                child,
                policy,
                classifier,
                label,
                config,
                device,
                dataset,
                rng,
                depth_start=1,
            )
            returns.append(g + init_repeat)
        q_values[action] = float(np.mean(returns)) if returns else 0.0

    return q_values, dataset
