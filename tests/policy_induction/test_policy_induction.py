"""Policy Induction offline: the memory step, rounds, both predictors, the contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial

import pytest

from think_reason_learn.exceptions import DataError, LLMError
from think_reason_learn.llm import Question
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.testing import (
    FakeLLM,
    GenerateRequest,
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


def test_rounds_and_batches_set_the_refinements() -> None:
    write = FakeLLM(generate=writer(BAD))  # never precise, so refining never stops
    with fake_llms(writer=write, judge=FakeLLM(answer=judge_answer, kind="decision")):
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


def test_rules_predictor_warns_when_it_weighs_no_rule(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with fake_llms(
        writer=FakeLLM(generate=writer(GOOD)),
        judge=FakeLLM(answer=lambda sample, question: 0.5, kind="decision"),
    ):
        model = policy_induction(
            initial_policy=GOOD, n_rounds=0, predictor="rules"
        ).fit(FOUNDERS, LABELS)
    assert model.weighted_rules_ == () and "Every weight is zero" in caplog.text


def test_paper_defaults() -> None:
    model = PolicyInduction()
    assert (model.n_rounds, model.n_samples, model.batch_size, model.n_examples) == (
        4,
        240,
        1,
        40,
    )
    assert (model.max_rules, model.predictor, model.beta) == (20, "policy", 0.5)


def test_policy_predictor_takes_the_answerers_call(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    assert policy_induction().fit(FOUNDERS, LABELS).threshold_ == 0.5


def test_each_round_asks_for_fresh_refinements() -> None:
    write = FakeLLM(generate=writer(BAD))
    with fake_llms(writer=write, judge=FakeLLM(answer=judge_answer, kind="decision")):
        policy_induction(initial_policy=BAD, n_rounds=2, batch_size=1).fit(
            FOUNDERS, LABELS
        )
    assert sum(c.schema is not None for c in write.calls) == 2 * 12


def test_a_perfectly_precise_policy_stops_refining(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    write, _ = fakes
    model = policy_induction(initial_policy=GOOD, n_rounds=4).fit(FOUNDERS, LABELS)
    assert model.history_.empty  # nothing could beat it, so nothing was paid for
    assert not any(c.schema is not None for c in write.calls)


def test_rules_predictor_drops_unweighted_rules(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    model = policy_induction(
        initial_policy=GOOD + ("Founders who like tea succeed.",),
        n_rounds=0,
        predictor="rules",
    ).fit(FOUNDERS, LABELS)
    assert model.weighted_rules_ and set(model.weighted_rules_) <= set(model.policy_)


def test_an_empty_first_policy_is_an_error() -> None:
    with (
        fake_llms(
            writer=FakeLLM(generate=writer(())),
            judge=FakeLLM(answer=judge_answer, kind="decision"),
        ),
        pytest.raises(LLMError, match="empty policy"),
    ):
        policy_induction().fit(FOUNDERS, LABELS)


def test_a_failed_refinement_keeps_the_policy() -> None:
    def refuse(request: GenerateRequest) -> str | dict[str, object] | None:
        if request.schema is None:
            return "Judge the founder."
        return (
            None if "Current policy" in (request.prompt or "") else {"rules": list(BAD)}
        )

    with fake_llms(
        writer=FakeLLM(generate=refuse),
        judge=FakeLLM(answer=judge_answer, kind="decision"),
    ):
        model = policy_induction().fit(FOUNDERS, LABELS)
    assert model.policy_ == BAD and not any(model.history_["kept"])


def test_a_refinement_that_cannot_be_judged_keeps_the_policy() -> None:
    def no_engineers(sample: str, question: Question) -> float | None:
        return None if "engineers" in question.text else judge_answer(sample, question)

    with fake_llms(
        writer=FakeLLM(generate=writer(GOOD)),
        judge=FakeLLM(answer=no_engineers, kind="decision"),
    ):
        model = policy_induction(initial_policy=BAD).fit(FOUNDERS, LABELS)
    assert model.policy_ == BAD and not any(model.history_["kept"])
