"""VerifiableRL: the contract, the fixes from the audit, episodes and the supervisor."""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_set_params,
)

from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import LLMSession, resolve_models
from think_reason_learn.testing import FakeLLM, GenerateRequest, fake_llms
from think_reason_learn.verifiable_rl import VerifiableRL
from think_reason_learn.verifiable_rl._episodes import Sample, Search, rollouts, targets
from think_reason_learn.verifiable_rl._networks import Classifier, PolicyNet
from think_reason_learn.verifiable_rl._supervisor import Supervisor, boost, is_uncertain

SLOTS = {"signal": 2, "noise": 2, "more_noise": 2}
rng = np.random.default_rng(0)
X = rng.normal(size=(60, 6)).astype(np.float32)
Y = np.where(X[:, 0] > 0, "hit", "miss")

fast = partial(
    VerifiableRL,
    SLOTS,
    policy_hidden=16,
    clf_hidden=16,
    max_steps=3,
    max_depth=2,
    n_rollouts=2,
    min_queries=1,
    update_every=10,
    policy_lr=1e-2,
    clf_lr=1e-2,
    pretrain_epochs=30,
    pretrain_lr=1e-2,
    freeze_clf_updates=0,
    pos_label="hit",
    random_state=0,
)


def test_contract(tmp_path: Path) -> None:
    for check in (
        check_no_attributes_set_in_init,
        check_get_params_invariance,
        check_set_params,
    ):
        check("VerifiableRL", fast())
    with pytest.raises(NotFittedError):
        fast().predict(X)
    model = fast()
    clone(model)
    assert model.fit(X, Y) is model
    proba = model.predict_proba(X)
    assert proba.shape == (60, 2) and np.allclose(proba.sum(axis=1), 1)
    predicted = model.predict(X)
    assert set(predicted) <= {"hit", "miss"}
    assert (pickle.loads(pickle.dumps(model)).predict(X) == predicted).all()
    assert (VerifiableRL.load(model.save(tmp_path / "m")).predict(X) == predicted).all()


def test_learns_from_the_informative_slot() -> None:
    model = fast(n_iterations=3).fit(X, Y)
    assert (model.predict(X) == Y).mean() >= 0.8


def test_never_reveals_a_slot_twice() -> None:
    paths = fast().fit(X, Y).decision_paths(X)
    assert all(len(path) == len(set(path)) for path in paths)
    assert all(path for path in paths)  # predict_min_queries=1


def test_a_saturated_stop_still_never_repeats_a_slot() -> None:
    model = fast(predict_min_queries=3).fit(X, Y)
    _always_stop(model.policy_)  # P(STOP) rounds to 1 in float32
    assert all(sorted(path) == sorted(SLOTS) for path in model.decision_paths(X[:5]))


def _always_stop(policy: PolicyNet) -> PolicyNet:
    head = policy.net[-1]
    assert isinstance(head, nn.Linear)
    with torch.no_grad():
        head.weight[-1] = 0.0
        head.bias[-1] = 40.0
    return policy


def test_rollouts_reveal_min_queries_before_stopping() -> None:
    search = Search(2, 20, 2, 1.0, 4.0, 4.0, -16.0, 0.0, -0.25, -0.1, -5.0, 0.3)
    sample = Sample(torch.ones(6), np.repeat(np.arange(3), 2), np.zeros(3))
    start = np.array([[True, False, False]] * 8)
    _, finals = rollouts(
        sample,
        start,
        1,
        (_always_stop(PolicyNet(6, 3, 16)), Classifier(6, 16)),
        1,
        search,
        np.random.default_rng(0),
    )
    assert ((finals != 0).sum(dim=1) >= 4).all()  # two slots of two columns


def test_supervisor_is_not_asked_when_stop_wins_anyway() -> None:
    advisor = FakeLLM(generate=lambda r: {"slot": "noise"})
    with fake_llms(advisor=advisor):
        models = resolve_models("fake:advisor", param="supervisor", task="generate")
        texts = {name: ["text"] for name in SLOTS}
        supervisor = Supervisor(models, LLMSession(), list(SLOTS), texts, 1.0, 0.5)
        masks = np.zeros((2, 3), dtype=bool)
        allowed = np.ones((2, 4), dtype=bool)
        probs = np.array([[0.1, 0.1, 0.1, 0.7], [0.3, 0.3, 0.3, 0.1]])
        nudged = supervisor.nudge(probs, allowed, masks, [0, 0])
    assert advisor.n_calls == 1  # only the second row
    assert np.array_equal(nudged[0], probs[0]) and nudged[1, 1] > nudged[1, 0]


def test_deterministic_without_touching_global_torch_state() -> None:
    torch.manual_seed(123)
    before = torch.get_rng_state()
    first, second = fast().fit(X, Y), fast().fit(X, Y)
    assert torch.equal(before, torch.get_rng_state())
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))


def test_dict_and_array_inputs_agree() -> None:
    model = fast().fit(X, Y)
    by_slot = {"signal": X[:, :2], "noise": X[:, 2:4], "more_noise": X[:, 4:]}
    assert np.allclose(model.predict_proba(by_slot), model.predict_proba(X))


def test_input_checks() -> None:
    with pytest.raises(DataError, match="columns"):
        fast().fit(X[:, :5], Y)
    with pytest.raises(ValueError, match="slots"):
        VerifiableRL().fit(X, Y)
    with pytest.raises(DataError, match="two classes"):
        fast().fit(X, np.arange(60) % 3)


SEARCH = Search(2, 2, 1, 1.0, 4.0, 4.0, -16.0, 0.0, -0.25, -0.1, -5.0, 0.3)


def _networks() -> tuple[PolicyNet, Classifier]:
    return PolicyNet(6, 3, 16), Classifier(6, 16)


def test_slot_costs_lower_the_return() -> None:
    columns = np.repeat(np.arange(3), 2)
    free = Sample(torch.ones(6), columns, np.zeros(3))
    costly = Sample(torch.ones(6), columns, np.array([1.0, 0.0, 0.0]))
    masks = np.array([[True, False, False]])
    gen = np.random.default_rng(0)
    paid = rollouts(costly, masks, 2, _networks(), 1, SEARCH, gen)[0]
    unpaid = rollouts(free, masks, 2, _networks(), 1, SEARCH, np.random.default_rng(0))[
        0
    ]
    assert paid[0] == pytest.approx(unpaid[0] - 1.0)


def test_targets_are_two_part() -> None:
    sample = Sample(torch.ones(6), np.repeat(np.arange(3), 2), np.zeros(3))
    info, stop, finals = targets(
        sample,
        np.zeros(3, dtype=bool),
        _networks(),
        1,
        SEARCH,
        np.random.default_rng(0),
    )
    assert info.shape == (3,) and info.sum() == pytest.approx(1.0)
    assert stop < 1e-6  # no slot revealed yet: min_queries forbids stopping
    assert finals.shape == (3 * SEARCH.n_rollouts, 6)


def test_supervisor_boost_is_multiplicative_and_keeps_stop() -> None:
    probs = np.array([0.3, 0.3, 0.2, 0.2])
    boosted = boost(probs, 3, 1, 0.5)
    assert boosted[3] == 0.2 and boosted[:3].sum() == pytest.approx(0.8)
    assert boosted[1] / boosted[0] == pytest.approx(1.5)
    assert is_uncertain(probs, 3, 0.01) and not is_uncertain(
        np.array([0.6, 0.1, 0.1, 0.2]), 3, 0.01
    )


def test_supervisor_sees_only_revealed_slots() -> None:
    prompts: list[str] = []

    def pick(request: GenerateRequest) -> dict[str, object]:
        prompts.append(request.prompt)
        return {"slot": "noise"}

    texts = {name: [f"{name} text {i}" for i in range(60)] for name in SLOTS}
    with fake_llms(advisor=FakeLLM(generate=pick)):
        model = fast(supervisor="fake:advisor", uncertain_delta=1.0, max_steps=2).fit(
            X[:20], Y[:20]
        )
        model.predict(X[:5], slot_texts=texts)
    assert prompts
    for prompt in prompts:
        revealed, _, offered = prompt.partition("\n\nSlots you may reveal next: ")
        shown = {line.split(":")[0] for line in revealed.splitlines()[1:]}
        assert not shown & set(offered.split(", "))


def test_a_failed_refit_leaves_no_model() -> None:
    model = fast().fit(X, Y)
    with pytest.raises(DataError):
        model.fit(X, np.arange(60) % 3)
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_integer_labels() -> None:
    labels = (Y == "hit").astype(int)
    model = fast(pos_label=1).fit(X, labels)
    assert model.predict(X).dtype.kind == "i"
