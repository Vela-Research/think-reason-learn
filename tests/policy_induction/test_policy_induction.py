"""Policy Induction offline: the memory step, rounds, both predictors, the contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import pytest

from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import Question
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.testing import (
    FakeLLM,
    GenerateRequest,
    check_classifier,
    fake_llms,
)

TRAITS = ["engineer", "lawyer", "engineer, exit", "chef", "engineer", "banker"]
FOUNDERS = [f"founder {i}: {TRAITS[i % len(TRAITS)]}" for i in range(36)]
LABELS = ["hit" if "engineer" in f else "miss" for f in FOUNDERS]
GOOD = ("Founders who are engineers succeed.",)
BAD = ("Founders who are lawyers succeed.",)


def judge_answer(sample: str, question: Question) -> float:
    """A decision model: positive when the sample has the policy's trait."""
    trait = "engineer" if "engineer" in question.text else "lawyer"
    return 0.9 if trait in sample else 0.1


def writer(
    policy: tuple[str, ...],
) -> Callable[[GenerateRequest], str | dict[str, object]]:
    def reply(request: GenerateRequest) -> str | dict[str, object]:
        return (
            {"rules": list(policy)}
            if request.schema is not None
            else "Judge the founder."
        )

    return reply


policy_induction = partial(
    PolicyInduction,
    task="Will the founder succeed?",
    generator="fake:writer",
    answerer="fake:judge",
    n_samples=12,
    n_rounds=1,
    batch_size=4,
    cv=3,
    pos_label="hit",
)


@pytest.fixture
def fakes() -> Iterator[tuple[FakeLLM, FakeLLM]]:
    write, judge = (
        FakeLLM(generate=writer(GOOD)),
        FakeLLM(answer=judge_answer, kind="decision"),
    )
    with fake_llms(writer=write, judge=judge):
        yield write, judge


def test_contract(fakes: tuple[FakeLLM, FakeLLM], tmp_path: Path) -> None:
    check_classifier(policy_induction, FOUNDERS, LABELS, tmp_path)


def test_a_better_refinement_is_kept(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    model = policy_induction(initial_policy=BAD).fit(FOUNDERS, LABELS)
    assert model.policy_ == GOOD
    kept = model.history_["kept"].tolist()
    assert kept[0] and not any(kept[1:])
    assert list(model.predict(["an engineer", "a poet"])) == ["hit", "miss"]


def test_a_worse_refinement_is_rejected() -> None:
    with fake_llms(
        writer=FakeLLM(generate=writer(BAD)),
        judge=FakeLLM(answer=judge_answer, kind="decision"),
    ):
        model = policy_induction(initial_policy=GOOD).fit(FOUNDERS, LABELS)
    assert model.policy_ == GOOD
    assert not any(model.history_["kept"].tolist())


def test_rounds_and_batches_set_the_refinements(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    write, _ = fakes
    model = policy_induction(n_rounds=2, batch_size=3).fit(FOUNDERS, LABELS)
    assert len(model.history_) == 2 * 12 // 3
    first_policy_calls = sum(
        c.schema is not None and "Current policy" not in (c.prompt or "")
        for c in write.calls
    )
    assert first_policy_calls == 1


def test_expert_policy_skips_the_first_draft(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    write, _ = fakes
    policy_induction(initial_policy=GOOD, n_rounds=0).fit(FOUNDERS, LABELS)
    assert not any(c.schema is not None for c in write.calls)


def test_rules_predictor_weighs_each_rule(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    model = policy_induction(
        initial_policy=GOOD + BAD, n_rounds=0, predictor="rules"
    ).fit(FOUNDERS, LABELS)
    assert model.combiner_ is not None and model.combiner_.coef[0] > 0
    assert list(model.predict(["an engineer", "a lawyer"])) == ["hit", "miss"]


def test_two_classes_only(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    with pytest.raises(DataError, match="two classes"):
        policy_induction().fit(FOUNDERS, [str(i % 3) for i in range(36)])


def test_paper_defaults() -> None:
    model = PolicyInduction()
    assert (model.n_rounds, model.n_samples, model.batch_size, model.n_examples) == (
        4,
        240,
        1,
        40,
    )
    assert (model.max_rules, model.predictor, model.beta) == (20, "policy", 0.5)
