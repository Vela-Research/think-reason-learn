"""Partly observed samples, rewards and Monte Carlo targets, batched.

A state is one sample with some slots revealed: its feature vector with the
unrevealed slots zeroed. Every rollout of a decision runs as one batch through
the networks, instead of one row at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from ._networks import Classifier, PolicyNet, action_probs

type Masks = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Search:
    """Settings of the rollouts and the reward."""

    n_rollouts: int
    max_depth: int
    min_queries: int
    tau_info: float
    tau_stop: float
    reward_tp: float
    reward_fp: float
    reward_tn: float
    reward_fn: float
    step_penalty: float
    repeat_penalty: float
    reward_threshold: float


@dataclass(frozen=True, slots=True)
class Sample:
    """One sample: its features, which slot each column belongs to, slot costs."""

    features: torch.Tensor
    columns: npt.NDArray[np.intp]
    costs: npt.NDArray[np.float64]

    @property
    def n_slots(self) -> int:
        """Number of slots."""
        return len(self.costs)

    def states(self, masks: Masks) -> torch.Tensor:
        """State vectors for revealed-slot ``masks`` of shape ``(n, n_slots)``."""
        revealed = torch.from_numpy(masks[:, self.columns]).to(self.features)
        return self.features.unsqueeze(0) * revealed


def rewards(
    classifier: Classifier, states: torch.Tensor, label: int, search: Search
) -> npt.NDArray[np.float64]:
    """Terminal rewards: the classifier's call scored against the label."""
    with torch.no_grad():
        called = (
            (torch.sigmoid(classifier(states)) >= search.reward_threshold).cpu().numpy()
        )
    if label == 1:
        return np.where(called, search.reward_tp, search.reward_fn)
    return np.where(called, search.reward_fp, search.reward_tn)


def policy_probs(policy: PolicyNet, states: torch.Tensor) -> npt.NDArray[np.float64]:
    """Action probabilities, one row per state."""
    with torch.no_grad():
        probs = action_probs(policy(states)).cpu().numpy().astype(np.float64)
    return probs / probs.sum(axis=1, keepdims=True)


def sample_actions(
    probs: npt.NDArray[np.float64], rng: np.random.Generator
) -> npt.NDArray[np.intp]:
    """One action per row, drawn in proportion to its (unnormalised) weights."""
    draws = rng.random((len(probs), 1)) * probs.sum(axis=1, keepdims=True)
    return np.minimum((probs.cumsum(axis=1) < draws).sum(axis=1), probs.shape[1] - 1)


def allowed_actions(masks: Masks, min_queries: int) -> Masks:
    """Per row, the unseen slots, and STOP once ``min_queries`` slots are seen."""
    return np.column_stack([~masks, masks.sum(axis=1) >= min_queries])


def greedy(probs: npt.NDArray[np.float64], allowed: Masks) -> npt.NDArray[np.intp]:
    """Per row, the most likely allowed action."""
    return np.argmax(np.where(allowed, probs, -1.0), axis=1)


def rollouts(
    sample: Sample,
    masks: Masks,
    depth: int,
    networks: tuple[PolicyNet, Classifier],
    label: int,
    search: Search,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], torch.Tensor]:
    """Play the policy from each mask to the end; the returns and final states.

    A repeated query costs ``repeat_penalty`` and reveals nothing; each step
    costs ``step_penalty``; each revealed slot costs its slot cost.
    """
    policy, classifier = networks
    masks = masks.copy()
    n = len(masks)
    returns, penalties = np.zeros(n), np.zeros(n)
    finals = sample.features.new_zeros((n, len(sample.columns)))
    active = np.ones(n, dtype=bool)
    while active.any():
        rows = np.flatnonzero(active)
        states = sample.states(masks[rows])
        done = np.ones(len(rows), dtype=bool)
        if depth < search.max_depth:
            probs = policy_probs(policy, states)
            probs[:, -1] *= masks[rows].sum(axis=1) >= search.min_queries
            actions = sample_actions(probs, rng)
            done = actions == sample.n_slots
            go, slots = rows[~done], actions[~done]
            repeated = masks[go, slots]
            penalties[go[repeated]] += search.repeat_penalty
            masks[go[~repeated], slots[~repeated]] = True
        if done.any():
            ended = rows[done]
            finals[ended] = states[torch.from_numpy(done)]
            returns[ended] = (
                rewards(classifier, states[torch.from_numpy(done)], label, search)
                + search.step_penalty * depth
                + penalties[ended]
                - masks[ended] @ sample.costs
            )
            active[ended] = False
        depth += 1
    return returns, finals


def targets(
    sample: Sample,
    mask: npt.NDArray[np.bool_],
    networks: tuple[PolicyNet, Classifier],
    label: int,
    search: Search,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], float, torch.Tensor]:
    """Policy targets at one state, from Monte Carlo values of each action.

    Each information action is revealed and then rolled out ``n_rollouts``
    times; STOP is worth the reward of stopping now, unless fewer than
    ``min_queries`` slots are revealed. The targets are a softmax over the
    information values and, separately, a sigmoid of the STOP value, as in the
    paper. So slot costs and the step penalty shape which slot is revealed, not
    when to stop: STOP is learned from whether the call made now is right.

    Returns:
        The information target, the STOP target, and the rollouts' final states
        (classifier training samples).
    """
    n_slots, k = sample.n_slots, search.n_rollouts
    children = np.repeat(mask[None, :], n_slots, axis=0) | np.eye(n_slots, dtype=bool)
    repeat = np.where(mask, search.repeat_penalty, 0.0)
    returns, finals = rollouts(
        sample, np.repeat(children, k, axis=0), 1, networks, label, search, rng
    )
    q_info = returns.reshape(n_slots, k).mean(axis=1) + repeat
    q_stop = (
        float(rewards(networks[1], sample.states(mask[None, :]), label, search)[0])
        if mask.sum() >= search.min_queries
        else -1e9
    )
    info = np.exp((q_info - q_info.max()) / search.tau_info)
    stop = float(1 / (1 + np.exp(-np.clip(q_stop / search.tau_stop, -60, 60))))
    return (info / info.sum()).astype(np.float32), stop, finals
