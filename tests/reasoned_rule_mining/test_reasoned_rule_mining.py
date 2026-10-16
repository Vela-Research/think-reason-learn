"""Reasoned Rule Mining offline: mining, calibration, the cascade, the contract."""

from __future__ import annotations

import re
from functools import partial

import pytest

from think_reason_learn.llm import Question, config_context
from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining
from think_reason_learn.testing import (
    FakeLLM,
    GenerateRequest,
    fake_llms,
)

TRAITS = ["engineer, exit", "lawyer", "engineer", "chef", "engineer, exit", "banker"]
FOUNDERS = [f"founder {i}: {TRAITS[i % len(TRAITS)]}" for i in range(48)]
LABELS = ["YES" if "engineer" in f else "NO" for f in FOUNDERS]


def write(request: GenerateRequest) -> str | dict[str, object]:
    if request.schema is None:
        return "YES when the founder is an engineer; otherwise NO."
    label = re.search(r"True class: (\w+)", request.prompt)
    assert label is not None
    return {
        "reasoning": "Engineers build.",
        "rule": f"IF engineer THEN class = {label[1]}",
    }


def judge(sample: str, question: Question) -> float:
    if question.text.startswith("Re-examine"):
        return 0.9 if "exit" in sample else 0.2
    return 0.97 if "engineer" in sample else 0.05


rule_mining = partial(
    ReasonedRuleMining,
    task="Will the founder succeed?",
    generator="fake:writer",
    answerer="fake:judge",
    n_samples=12,
    cv=3,
)


@pytest.fixture
def scripted() -> tuple[FakeLLM, FakeLLM]:
    return FakeLLM(generate=write), FakeLLM(answer=judge, kind="decision")


def test_mines_one_rule_per_sample_from_its_true_class(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    model = rule_mining().fit(FOUNDERS, LABELS)
    assert len(model.mined_rules_) == 12
    assert {label for label, _ in model.mined_rules_} == {"YES", "NO"}
    assert all(rule.endswith(f"= {label}") for label, rule in model.mined_rules_)
    assert model.policy_.startswith("YES when the founder is an engineer")
    assert list(model.predict(["an engineer", "a poet"])) == ["YES", "NO"]


def test_calibrates_on_samples_the_miner_never_saw(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    writer, answerer = fakes
    rule_mining().fit(FOUNDERS, LABELS)
    mined = {
        line.removeprefix("Sample:\n")
        for c in writer.calls
        if c.schema is not None
        for line in [(c.prompt or "").split("\n\n")[1]]
    }
    judged = {c.sample for c in answerer.calls}
    assert len(mined) == 12 and not mined & judged


def test_only_confident_samples_get_the_harsh_look() -> None:
    first, harsh = (
        FakeLLM(answer=judge, kind="decision"),
        FakeLLM(answer=judge, kind="decision"),
    )
    with fake_llms(writer=FakeLLM(generate=write), judge=first, harsh=harsh):
        rule_mining(harsh_answerer="fake:harsh").fit(FOUNDERS, LABELS)
    reviewed = {c.sample for c in harsh.calls}
    assert reviewed and all("engineer" in (s or "") for s in reviewed)
    assert not any(c.questions[0].startswith("Re-examine") for c in first.calls)


def test_no_harsh_look_when_off(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    _, answerer = fakes
    rule_mining(use_harsh=False).fit(FOUNDERS, LABELS)
    assert not any(c.questions[0].startswith("Re-examine") for c in answerer.calls)


def test_reasoning_answerers_vote_in_an_ensemble() -> None:
    voter = FakeLLM(answer=lambda s, q: "YES" if "engineer" in s else "NO")
    with fake_llms(writer=FakeLLM(generate=write), judge=voter):
        rule_mining(ensemble_size=3, use_harsh=False).fit(FOUNDERS, LABELS)
    assert voter.n_calls == 3 * (len(FOUNDERS) - 12)


def test_paper_defaults() -> None:
    model = ReasonedRuleMining()
    assert (model.harsh_gate, model.harsh_level, model.beta) == (0.95, "strict", 0.5)


def test_rare_positives_still_calibrate_on_unmined_samples() -> None:
    founders = [
        f"founder {i}: {'engineer' if i % 10 == 0 else 'chef'}" for i in range(60)
    ]
    labels = ["YES" if "engineer" in f else "NO" for f in founders]
    writer, answerer = FakeLLM(generate=write), FakeLLM(answer=judge, kind="decision")
    with fake_llms(writer=writer, judge=answerer):
        model = rule_mining(n_samples=500, cv=2).fit(founders, labels)
    mined = {
        (c.prompt or "").split("\n\n")[1].removeprefix("Sample:\n")
        for c in writer.calls
        if c.schema
    }
    judged = {c.sample for c in answerer.calls}
    assert {label for label, _ in model.mined_rules_} == {"YES", "NO"}
    assert (
        len([m for m in mined if "engineer" in m]) <= 3
    )  # at most half of 6 positives
    assert not mined & judged


def test_probabilities_are_not_pulled_to_one_half(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    founders = [
        f"founder {i}: {'engineer' if i % 10 == 0 else 'chef'}" for i in range(100)
    ]
    labels = ["YES" if "engineer" in f else "NO" for f in founders]
    model = rule_mining(n_samples=10, cv=3).fit(founders, labels)
    mean_positive = model.predict_proba(founders)[:, 1].mean()
    assert abs(mean_positive - 0.1) < 0.1


def test_an_unweighted_second_look_is_not_paid_for_at_predict(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    _, answerer = fakes
    model = rule_mining().fit(FOUNDERS, LABELS)
    model.combiner_ = type(model.combiner_)(
        model.combiner_.coef * [1, 0, 0], model.combiner_.intercept, model.threshold_
    )
    asked = answerer.n_calls
    model.predict(["founder: engineer, exit"])  # rated highly: would be promoted
    assert answerer.n_calls == asked + 1  # the first pass only
    assert model.estimate_cost(["founder: engineer, exit"]).calls == 1


def test_a_sample_the_miner_skips_does_not_stop_the_fit() -> None:
    mining_calls = iter(range(100))

    def flaky(request: GenerateRequest) -> str | dict[str, object] | None:
        if request.schema is not None and next(mining_calls) == 0:
            return None  # the first mined sample gets no usable rule
        return write(request)

    with fake_llms(
        writer=FakeLLM(generate=flaky), judge=FakeLLM(answer=judge, kind="decision")
    ):
        model = rule_mining().fit(FOUNDERS, LABELS)
    assert len(model.mined_rules_) == 11


def test_the_plan_covers_a_second_look_at_negatives() -> None:
    labels = ["YES" if "exit" in f else "NO" for f in FOUNDERS]  # rated by "engineer"
    with (
        fake_llms(
            writer=FakeLLM(generate=write), judge=FakeLLM(answer=judge, kind="decision")
        ),
        config_context(cache=False),
    ):
        model = rule_mining()
        planned = model.estimate_cost(FOUNDERS, labels).calls
        model.fit(FOUNDERS, labels)
    assert model.usage_.calls <= planned


def test_mining_every_sample_calibrates_on_all_without_warning(
    fakes: tuple[FakeLLM, FakeLLM], caplog: pytest.LogCaptureFixture
) -> None:
    rule_mining(n_samples=None).fit(FOUNDERS, LABELS)
    assert "Too few" not in caplog.text
