"""VerifiableRL: decide which information to reveal, one slot at a time.

Ye, Alican, Griffin, Ontoyin Yin and Ihlamur, "Learning What to Ask: Cost-Aware
Sequential Founder Evaluation" (MIDAS @ ECML-PKDD 2026),
https://vela.partners/research/cost-aware-sequential-evaluation.

Each sample's features come in named slots (e.g. education, jobs). A policy
network chooses which slot to reveal next, or to stop; a classifier predicts the
label from what was revealed. The policy learns from Monte Carlo rollouts scored
by cost-aware, asymmetric rewards; an optional LLM supervisor breaks near-ties.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Self

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

from think_reason_learn.core._classifier import (
    binary_decision,
    binary_proba,
    forget_fitted,
    positive_index,
)
from think_reason_learn.core._combine import best_threshold
from think_reason_learn.core._data import Classes, LabelInput, encode_labels
from think_reason_learn.core._persistence import Persistable
from think_reason_learn.core._sklearn import require_fitted
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import LLMSession, Models, Usage, resolve_models

from ._episodes import (
    Sample,
    Search,
    allowed_actions,
    greedy,
    policy_probs,
    targets,
)
from ._networks import Classifier, PolicyNet
from ._supervisor import Supervisor
from ._torch import backward, private_cpu_rng, shuffled_batches, tensor

type SlotFeatures = npt.NDArray[np.floating] | Mapping[str, npt.NDArray[np.floating]]
"""A 2-D array with the slots' columns in order, or one array per slot."""

type SlotTexts = Mapping[str, Sequence[str]]
"""Per slot, one text per sample, shown to the supervisor once revealed."""


class VerifiableRL(Persistable, ClassifierMixin, BaseEstimator):
    """A two-class classifier that learns which slots of information to reveal.

    Args:
        slots: Each slot's name and number of feature columns, in column order.
        slot_costs: Cost of revealing each slot, subtracted from the reward.
        supervisor: Reasoning model(s) that suggest a slot when the policy's top
            two are within ``uncertain_delta``. Needs ``slot_texts``.
        policy_hidden, clf_hidden: Hidden widths of the networks.
        max_steps: Most slots revealed per sample.
        max_depth, n_rollouts: Depth and number of rollouts per action.
        min_queries: Slots to reveal before STOP may be chosen in training.
        predict_min_queries: The same, when predicting.
        reward_tp, reward_fp, reward_tn, reward_fn: Rewards of the final call.
        step_penalty, repeat_penalty: Added per step and per repeated query.
        reward_threshold: Classifier probability counted as a positive call.
        tau_info, tau_stop: Temperatures of the policy targets.
        n_iterations: Passes over the training samples.
        update_every: Samples between network updates.
        policy_lr, clf_lr: Learning rates.
        policy_batch, clf_batch: Batch sizes.
        train_epochs: Passes over the replay sample per update.
        grad_clip: Gradient-norm clip; 0 to turn off.
        eps: Chance of a random action while training.
        policy_replay, clf_replay: Replay buffer sizes.
        policy_sample, clf_sample: Examples drawn per update; 0 for all.
        freeze_clf_updates: Updates before the classifier starts training.
        clf_target_update_every: Updates between syncs of the frozen classifier
            the rollouts use; 0 to use the live one.
        pretrain_epochs: Epochs of classifier pretraining on random reveals;
            0 to skip. Rewards depend on the classifier, so this helps.
        pretrain_lr, pretrain_batch: Its learning rate and batch size.
        uncertain_delta: Gap between the top two actions that asks the supervisor.
        llm_bias: The supervisor's pick is multiplied by ``1 + llm_bias``.
        threshold: Positive probability for a positive prediction. ``None`` tunes
            it for F-beta on the training samples.
        beta: F-beta for the tuned threshold.
        pos_label: The positive class; by default the second sorted label.
        device: Torch device.
        random_state: Seed for the networks, exploration and rollouts.
    """

    def __init__(
        self,
        slots: Mapping[str, int] | None = None,
        *,
        slot_costs: Mapping[str, float] | None = None,
        supervisor: Models | None = None,
        policy_hidden: int = 512,
        clf_hidden: int = 256,
        max_steps: int = 5,
        max_depth: int = 4,
        n_rollouts: int = 10,
        min_queries: int = 3,
        predict_min_queries: int = 1,
        reward_tp: float = 4.0,
        reward_fp: float = -16.0,
        reward_tn: float = 0.0,
        reward_fn: float = -0.25,
        step_penalty: float = -0.1,
        repeat_penalty: float = -5.0,
        reward_threshold: float = 0.3,
        tau_info: float = 1.0,
        tau_stop: float = 4.0,
        n_iterations: int = 1,
        update_every: int = 25,
        policy_lr: float = 5e-5,
        clf_lr: float = 1e-5,
        policy_batch: int = 128,
        clf_batch: int = 512,
        train_epochs: int = 1,
        grad_clip: float = 5.0,
        eps: float = 0.1,
        policy_replay: int = 20_000,
        clf_replay: int = 40_000,
        policy_sample: int = 2_000,
        clf_sample: int = 4_000,
        freeze_clf_updates: int = 10,
        clf_target_update_every: int = 5,
        pretrain_epochs: int = 50,
        pretrain_lr: float = 1e-5,
        pretrain_batch: int = 16,
        uncertain_delta: float = 0.01,
        llm_bias: float = 0.05,
        threshold: float | None = None,
        beta: float = 0.5,
        pos_label: str | int | None = None,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        self.slots = slots
        self.slot_costs = slot_costs
        self.supervisor = supervisor
        self.policy_hidden = policy_hidden
        self.clf_hidden = clf_hidden
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.n_rollouts = n_rollouts
        self.min_queries = min_queries
        self.predict_min_queries = predict_min_queries
        self.reward_tp = reward_tp
        self.reward_fp = reward_fp
        self.reward_tn = reward_tn
        self.reward_fn = reward_fn
        self.step_penalty = step_penalty
        self.repeat_penalty = repeat_penalty
        self.reward_threshold = reward_threshold
        self.tau_info = tau_info
        self.tau_stop = tau_stop
        self.n_iterations = n_iterations
        self.update_every = update_every
        self.policy_lr = policy_lr
        self.clf_lr = clf_lr
        self.policy_batch = policy_batch
        self.clf_batch = clf_batch
        self.train_epochs = train_epochs
        self.grad_clip = grad_clip
        self.eps = eps
        self.policy_replay = policy_replay
        self.clf_replay = clf_replay
        self.policy_sample = policy_sample
        self.clf_sample = clf_sample
        self.freeze_clf_updates = freeze_clf_updates
        self.clf_target_update_every = clf_target_update_every
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_batch = pretrain_batch
        self.uncertain_delta = uncertain_delta
        self.llm_bias = llm_bias
        self.threshold = threshold
        self.beta = beta
        self.pos_label = pos_label
        self.device = device
        self.random_state = random_state

    classes_: Classes
    slot_names_: list[str]
    columns_: npt.NDArray[np.intp]
    costs_: npt.NDArray[np.float64]
    policy_: PolicyNet
    classifier_: Classifier
    threshold_: float
    usage_: Usage

    def fit(
        self, X: SlotFeatures, y: LabelInput, *, slot_texts: SlotTexts | None = None
    ) -> Self:
        """Train the policy and the classifier.

        Args:
            X: Slot features: a 2-D array (columns in ``slots`` order) or one
                array per slot.
            y: One of two labels per sample.
            slot_texts: Text of each slot per sample, for the supervisor.
        """
        forget_fitted(self)
        try:
            features = self._layout(X, fitting=True)
            classes, codes = encode_labels(y)
            if len(classes) != 2 or len(codes) != len(features):
                raise DataError("y needs two classes and one label per row of X.")
            positive_index(classes, self.pos_label)  # a bad pos_label fails now
            self.classes_ = classes
            self._train(features, codes, slot_texts)
        except BaseException:
            forget_fitted(self, keep="usage_")  # what the failed fit spent
            raise
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted once ``fit`` has finished: the threshold is set last."""
        return hasattr(self, "threshold_")

    def _train(
        self,
        features: npt.NDArray[np.float32],
        codes: npt.NDArray[np.intp],
        slot_texts: SlotTexts | None,
    ) -> None:
        positive = (codes == self._pos_index()).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        self._build_networks(rng)
        session = LLMSession()
        self.usage_ = session.usage
        if self.pretrain_epochs:
            self._pretrain(features, positive, rng)
        _Trainer(
            self, features, positive, self._supervisor(session, slot_texts), rng
        ).run()
        proba = self._final_proba(features, slot_texts)
        self.threshold_ = (
            self.threshold
            if self.threshold is not None
            else best_threshold(proba, positive == 1, self.beta)[0]
        )

    def predict_proba(
        self, X: SlotFeatures, *, slot_texts: SlotTexts | None = None
    ) -> npt.NDArray[np.float64]:
        """Class probabilities, columns in ``classes_`` order."""
        require_fitted(self)
        positive = self._final_proba(self._layout(X), slot_texts)
        return binary_proba(positive, self._pos_index())

    def predict(
        self, X: SlotFeatures, *, slot_texts: SlotTexts | None = None
    ) -> Classes:
        """The predicted label of each sample."""
        proba = self.predict_proba(X, slot_texts=slot_texts)
        return self.classes_[binary_decision(proba, self._pos_index(), self.threshold_)]

    def decision_paths(
        self, X: SlotFeatures, *, slot_texts: SlotTexts | None = None
    ) -> list[list[str]]:
        """The slots revealed for each sample, in order."""
        require_fitted(self)
        _, paths = self._reveal(self._layout(X), slot_texts)
        return [[self.slot_names_[s] for s in path] for path in paths]

    def _final_proba(
        self, features: npt.NDArray[np.float32], slot_texts: SlotTexts | None
    ) -> npt.NDArray[np.float64]:
        masks, _ = self._reveal(features, slot_texts)
        with torch.no_grad():
            logits = self.classifier_(self._states(features, masks))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def _reveal(
        self, features: npt.NDArray[np.float32], slot_texts: SlotTexts | None
    ) -> tuple[npt.NDArray[np.bool_], list[list[int]]]:
        """Reveal slots greedily for every sample at once; never a slot already seen."""
        supervisor = self._supervisor(LLMSession(usage=self.usage_), slot_texts)
        n, n_slots = len(features), len(self.slot_names_)
        masks = np.zeros((n, n_slots), dtype=bool)
        paths: list[list[int]] = [[] for _ in range(n)]
        active = np.ones(n, dtype=bool)
        for _ in range(self.max_steps):
            rows = np.flatnonzero(active)
            if not len(rows):
                break
            allowed = allowed_actions(masks[rows], self.predict_min_queries)
            probs = policy_probs(
                self.policy_, self._states(features[rows], masks[rows])
            )
            if supervisor is not None:
                probs = supervisor.nudge(probs, allowed, masks[rows], rows)
            actions: list[int] = greedy(probs, allowed).tolist()
            picked: list[int] = rows.tolist()
            for row, action in zip(picked, actions, strict=True):
                if action == n_slots:
                    active[row] = False
                else:
                    masks[row, action] = True
                    paths[row].append(action)
            active &= ~masks.all(axis=1)
        return masks, paths

    def _states(
        self, features: npt.NDArray[np.float32], masks: npt.NDArray[np.bool_]
    ) -> torch.Tensor:
        return tensor(features * masks[:, self.columns_], torch.device(self.device))

    def _layout(
        self, X: SlotFeatures, *, fitting: bool = False
    ) -> npt.NDArray[np.float32]:
        """Features as one float array with the slots' columns in order."""
        if not self.slots:
            raise ValueError("VerifiableRL needs slots={name: n_columns, ...}.")
        names = list(self.slots)
        matrix = (
            np.hstack([np.asarray(X[s]) for s in names])
            if isinstance(X, Mapping)
            else np.asarray(X)
        )
        if (
            matrix.ndim != 2
            or matrix.shape[1] != sum(self.slots.values())
            or not len(matrix)
        ):
            raise DataError(
                f"X needs {sum(self.slots.values())} columns for the slots "
                f"{dict(self.slots)}."
            )
        if fitting:
            self.slot_names_ = names
            self.columns_ = np.repeat(
                np.arange(len(names)), list(self.slots.values())
            ).astype(np.intp)
            costs = self.slot_costs or {}
            self.costs_ = np.array([costs.get(s, 0.0) for s in names], dtype=float)
        return matrix.astype(np.float32)

    def _pos_index(self) -> int:
        return positive_index(self.classes_, self.pos_label)

    def _supervisor(
        self, session: LLMSession, slot_texts: SlotTexts | None
    ) -> Supervisor | None:
        if self.supervisor is None or slot_texts is None:
            return None
        models = resolve_models(self.supervisor, param="supervisor", task="generate")
        return Supervisor(
            models,
            session,
            self.slot_names_,
            slot_texts,
            self.uncertain_delta,
            self.llm_bias,
        )

    def _build_networks(self, rng: np.random.Generator) -> None:
        state_dim = len(self.columns_)
        with private_cpu_rng(int(rng.integers(2**31))):
            self.policy_ = PolicyNet(
                state_dim, len(self.slot_names_), self.policy_hidden
            ).to(self.device)
            self.classifier_ = Classifier(state_dim, self.clf_hidden).to(self.device)

    def _pretrain(
        self,
        features: npt.NDArray[np.float32],
        positive: npt.NDArray[np.float32],
        rng: np.random.Generator,
    ) -> None:
        """Train the classifier on random reveals: all slots first, then fewer."""
        n_slots = len(self.slot_names_)
        optimiser = torch.optim.Adam(self.classifier_.parameters(), lr=self.pretrain_lr)
        loss = nn.BCEWithLogitsLoss()
        for epoch in range(self.pretrain_epochs):
            low, high = _curriculum(epoch / self.pretrain_epochs, n_slots)
            for batch in np.array_split(
                rng.permutation(len(features)),
                max(1, len(features) // self.pretrain_batch),
            ):
                masks = _random_masks(len(batch), n_slots, low, high, rng)
                _step(
                    optimiser,
                    loss(
                        self.classifier_(self._states(features[batch], masks)),
                        tensor(positive[batch], torch.device(self.device)),
                    ),
                    self.classifier_,
                    self.grad_clip,
                )


@dataclass(slots=True)
class _Trainer:
    """The two-network training loop of one ``fit``."""

    model: VerifiableRL
    features: npt.NDArray[np.float32]
    positive: npt.NDArray[np.float32]
    supervisor: Supervisor | None
    rng: np.random.Generator
    shuffle: torch.Generator = field(default_factory=torch.Generator)
    policy_examples: list[tuple[torch.Tensor, npt.NDArray[np.float32], float]] = field(
        default_factory=list[tuple[torch.Tensor, npt.NDArray[np.float32], float]]
    )
    clf_examples: list[tuple[torch.Tensor, float]] = field(
        default_factory=list[tuple[torch.Tensor, float]]
    )
    search: Search = field(init=False)
    updates: int = 0

    def __post_init__(self) -> None:
        self.search = Search(
            **{f.name: getattr(self.model, f.name) for f in fields(Search)}
        )

    def run(self) -> None:
        model = self.model
        self.shuffle.manual_seed(int(self.rng.integers(2**31)))
        target = self._frozen_copy() if model.clf_target_update_every else None
        optimisers = (
            torch.optim.Adam(model.policy_.parameters(), lr=model.policy_lr),
            torch.optim.Adam(model.classifier_.parameters(), lr=model.clf_lr),
        )
        for _ in range(model.n_iterations):
            for count, i in enumerate(self.rng.permutation(len(self.features)), 1):
                self._episode(int(i), target or model.classifier_)
                if count % model.update_every == 0:
                    self._update(optimisers, target)
            if len(self.features) % model.update_every:
                self._update(optimisers, target)

    def _episode(self, i: int, rollout_classifier: Classifier) -> None:
        """Visit states of one sample, recording policy targets at each."""
        model, n_slots = self.model, len(self.model.slot_names_)
        sample = Sample(
            tensor(self.features[i], torch.device(model.device)),
            model.columns_,
            model.costs_,
        )
        mask, label = np.zeros(n_slots, dtype=bool), int(self.positive[i])
        for _ in range(model.max_steps):
            info, stop, finals = targets(
                sample,
                mask,
                (model.policy_, rollout_classifier),
                label,
                self.search,
                self.rng,
            )
            state = sample.states(mask[None, :])[0]
            self.policy_examples.append((state, info, stop))
            self.clf_examples.extend((row, float(label)) for row in finals)
            action = self._explore(sample, mask, i)
            if action == n_slots:
                break
            mask[action] = True
            if mask.all():
                break
        self.policy_examples = self.policy_examples[-model.policy_replay :]
        self.clf_examples = self.clf_examples[-model.clf_replay :]

    def _explore(self, sample: Sample, mask: npt.NDArray[np.bool_], i: int) -> int:
        """An unseen slot, or STOP after ``min_queries``: at random or greedy."""
        masks = mask[None, :]
        allowed = allowed_actions(masks, self.model.min_queries)
        if self.rng.random() < self.model.eps:
            return int(self.rng.choice(np.flatnonzero(allowed[0])))
        probs = policy_probs(self.model.policy_, sample.states(masks))
        if self.supervisor is not None:
            probs = self.supervisor.nudge(probs, allowed, masks, [i])
        return int(greedy(probs, allowed)[0])

    def _update(
        self,
        optimisers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        target: Classifier | None,
    ) -> None:
        model = self.model
        self.updates += 1
        self._train_policy(optimisers[0])
        if self.updates >= model.freeze_clf_updates:
            self._train_classifier(optimisers[1])
        if target is not None and self.updates % model.clf_target_update_every == 0:
            target.load_state_dict(model.classifier_.state_dict())

    def _train_policy(self, optimiser: torch.optim.Optimizer) -> None:
        model = self.model
        examples = _sample(self.policy_examples, model.policy_sample, self.rng)
        if not examples:
            return
        states = torch.stack([s for s, _, _ in examples])
        info = tensor(np.stack([t for _, t, _ in examples]), states)
        stop = torch.tensor([t for _, _, t in examples]).to(states)
        for _ in range(model.train_epochs):
            for batch in shuffled_batches(
                len(states), model.policy_batch, self.shuffle
            ):
                logits = model.policy_(states[batch])
                loss = (
                    -(info[batch] * torch.log_softmax(logits[:, :-1], dim=1))
                    .sum(dim=1)
                    .mean()
                )
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    logits[:, -1], stop[batch]
                )
                _step(optimiser, loss, model.policy_, model.grad_clip)

    def _train_classifier(self, optimiser: torch.optim.Optimizer) -> None:
        model = self.model
        examples = _sample(self.clf_examples, model.clf_sample, self.rng)
        if not examples:
            return
        states = torch.stack([s for s, _ in examples])
        labels = torch.tensor([y for _, y in examples]).to(states)
        for _ in range(model.train_epochs):
            for batch in shuffled_batches(len(states), model.clf_batch, self.shuffle):
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model.classifier_(states[batch]), labels[batch]
                )
                _step(optimiser, loss, model.classifier_, model.grad_clip)

    def _frozen_copy(self) -> Classifier:
        """A frozen copy for the rollouts (a deep copy draws no random numbers)."""
        return (
            copy.deepcopy(self.model.classifier_)
            .requires_grad_(requires_grad=False)
            .eval()
        )


def _sample[E](examples: list[E], size: int, rng: np.random.Generator) -> list[E]:
    if size <= 0 or len(examples) <= size:
        return examples
    return [examples[i] for i in rng.choice(len(examples), size=size, replace=False)]


def _step(
    optimiser: torch.optim.Optimizer,
    loss: torch.Tensor,
    network: nn.Module,
    grad_clip: float,
) -> None:
    optimiser.zero_grad(set_to_none=True)
    backward(loss)
    if grad_clip:
        nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optimiser.step()


def _random_masks(
    n: int, n_slots: int, low: int, high: int, rng: np.random.Generator
) -> npt.NDArray[np.bool_]:
    """``n`` masks, each revealing a random number in [low, high] of random slots."""
    counts = rng.integers(max(1, low), max(1, high) + 1, size=n)
    order = np.argsort(rng.random((n, n_slots)), axis=1)
    return order < counts[:, None]


def _curriculum(progress: float, n_slots: int) -> tuple[int, int]:
    """Slots revealed in pretraining: all, then a middle range, then one or two."""
    if progress < 0.3:
        return n_slots, n_slots
    if progress < 0.7:
        return min(2, n_slots), max(min(2, n_slots), n_slots - 1)
    return 1, min(2, n_slots)
