"""Pydantic models, configuration, and result types for Verifiable RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field


class NextActionPreference(BaseModel):
    """Structured weak preference returned by the LLM action supervisor.

    Attributes:
        prefer: Zero or one preferred slot name. The supervisor only nudges the
            policy; an empty list means "no clear preference".
    """

    prefer: List[str] = Field(default_factory=list)


@dataclass
class QueryResult:
    """The outcome of running the policy on a single sample.

    Attributes:
        probability: Classifier success probability from the final state.
        prediction: Binary label (``1`` if ``probability >= predict_threshold``).
        slots_used: Distinct information slots queried (in first-query order).
        decision_path: Ordered actions taken, each a slot name or ``"stop"``.
    """

    probability: float
    prediction: int
    slots_used: List[str]
    decision_path: List[str]


@dataclass
class VerifiableRLConfig:
    """Configuration for :class:`VerifiableRL`.

    Defaults mirror the standalone VCBench reference implementation
    (``runs/model_*`` checkpoints were trained with these network sizes and
    reward values).

    Args:
        policy_hidden: Hidden width of the PolicyNet MLP.
        clf_hidden: Hidden width of the Classifier MLP.
        max_steps: Max actions per episode during training/prediction.
        max_depth: Max rollout depth inside MCTS.
        n_rollouts: Monte-Carlo rollouts per root action.
        min_queries: Slots that must be revealed before STOP is allowed when
            building policy targets.
        predict_min_queries: Same guard, applied at prediction time.
        reward_tp/reward_fp/reward_tn/reward_fn: Asymmetric terminal rewards.
        step_penalty: Per-step penalty added to a terminal reward.
        repeat_penalty: Penalty for re-querying an already-observed slot.
        clf_threshold: Probability threshold used inside the reward function.
        tau_info: Temperature for the info-action softmax target.
        tau_stop: Temperature for the STOP-action sigmoid target.
        n_iterations: Passes over the training set in ``fit``.
        update_every: Train the networks every this-many episodes.
        policy_lr/clf_lr: Adam learning rates.
        policy_batch/clf_batch: Mini-batch sizes.
        train_epochs: Optimisation epochs per network update.
        grad_clip: Gradient-norm clip (``None``/0 disables).
        eps: Epsilon for epsilon-greedy exploration during training.
        policy_replay_max/clf_replay_max: Replay-buffer caps.
        policy_sample/clf_sample: Samples drawn per update (``0`` = use all).
        freeze_clf_updates: Number of initial network updates during which the
            classifier is held fixed (only the policy trains) so the policy can
            first adapt to a stable (warm-started) classifier. ``0`` disables.
        clf_target_update_every: Sync the frozen *target* classifier (the one
            the MCTS rollouts/rewards query) to the live classifier every this
            many updates. ``0`` makes rollouts use the live classifier directly.
        uncertain_delta: Top-2 info-prob gap below which the supervisor is asked.
        llm_bias: Probability mass added to the supervisor's preferred slot.
        predict_threshold: Probability threshold for the final binary label.
        greedy: If True, take argmax actions at prediction time.
    """

    # networks
    policy_hidden: int = 512
    clf_hidden: int = 256
    # episode / search
    max_steps: int = 5
    max_depth: int = 4
    n_rollouts: int = 10
    min_queries: int = 3
    predict_min_queries: int = 1
    # reward
    reward_tp: float = 4.0
    reward_fp: float = -16.0
    reward_tn: float = 0.0
    reward_fn: float = -0.25
    step_penalty: float = -0.1
    repeat_penalty: float = -5.0
    clf_threshold: float = 0.3
    # target temperatures
    tau_info: float = 1.0
    tau_stop: float = 4.0
    # training
    n_iterations: int = 1
    update_every: int = 25
    policy_lr: float = 5e-5
    clf_lr: float = 1e-5
    policy_batch: int = 128
    clf_batch: int = 512
    train_epochs: int = 1
    grad_clip: float = 5.0
    eps: float = 0.1
    policy_replay_max: int = 20000
    clf_replay_max: int = 40000
    policy_sample: int = 2000
    clf_sample: int = 4000
    freeze_clf_updates: int = 10
    clf_target_update_every: int = 5
    # supervisor
    uncertain_delta: float = 0.01
    llm_bias: float = 0.05
    # prediction
    predict_threshold: float = 0.3
    greedy: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any value is out of its allowed range.
        """
        if self.policy_hidden < 1 or self.clf_hidden < 1:
            raise ValueError("policy_hidden and clf_hidden must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.n_rollouts < 1:
            raise ValueError("n_rollouts must be >= 1")
        if self.min_queries < 0 or self.predict_min_queries < 0:
            raise ValueError("min_queries and predict_min_queries must be >= 0")
        if self.tau_info <= 0 or self.tau_stop <= 0:
            raise ValueError("tau_info and tau_stop must be > 0")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if self.update_every < 1:
            raise ValueError("update_every must be >= 1")
        if self.policy_batch < 1 or self.clf_batch < 1:
            raise ValueError("policy_batch and clf_batch must be >= 1")
        if self.train_epochs < 1:
            raise ValueError("train_epochs must be >= 1")
        if self.freeze_clf_updates < 0:
            raise ValueError("freeze_clf_updates must be >= 0")
        if self.clf_target_update_every < 0:
            raise ValueError("clf_target_update_every must be >= 0")
        if not (0.0 <= self.clf_threshold <= 1.0):
            raise ValueError("clf_threshold must be in [0, 1]")
        if not (0.0 <= self.predict_threshold <= 1.0):
            raise ValueError("predict_threshold must be in [0, 1]")
        if not (0.0 <= self.eps <= 1.0):
            raise ValueError("eps must be in [0, 1]")
