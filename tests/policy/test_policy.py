"""Policy decisions offline: will the recipient love this gift? NO < MAYBE < YES."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    get_scorer,  # pyright: ignore[reportUnknownVariableType]  # partial annotation
    roc_auc_score,  # pyright: ignore[reportUnknownVariableType]  # partial annotation
)
from sklearn.model_selection import (
    cross_val_predict,  # pyright: ignore[reportUnknownVariableType]  # partial annotation
)

from think_reason_learn import LLM, Price
from think_reason_learn.exceptions import BudgetExceededError, DataError
from think_reason_learn.llm import Question, Usage
from think_reason_learn.policy import (
    Criterion,
    Mistake,
    Outcome,
    Policy,
    PolicyClassifier,
    TeacherVerdict,
    answer_checks,
    label_cases,
    write_checks,
)
from think_reason_learn.policy._checks import policy_checks, weight_bounds
from think_reason_learn.policy._labels import outcome_codes
from think_reason_learn.testing import (
    FakeLLM,
    GenerateRequest,
    check_classifier,
    fake_llms,
)

HOBBY = Criterion("C1", "The gift matches a hobby the recipient has.", "positive")
OWNS = Criterion(
    "C2", "The recipient already owns something like the gift.", "negative"
)
HANDMADE = Criterion("C3", "The gift is handmade.", "positive")
LOCAL = Criterion("C4", "The gift was bought from a local shop.", "neutral")
OUTCOMES = (
    Outcome("NO", "the recipient will not like the gift"),
    Outcome("MAYBE", "the recipient may like the gift"),
    Outcome("YES", "the recipient will love the gift"),
)
GIFTS = Policy(
    text="Will the recipient love this gift?",
    outcomes=OUTCOMES,
    criteria=(HOBBY, OWNS, HANDMADE, LOCAL),
)
MARKERS = {
    HOBBY.text: "matches a hobby",
    OWNS.text: "already owns",
    HANDMADE.text: "handmade",
}
NEGATIVE = {OWNS.text}


def case(i: int, traits: str) -> str:
    """A gift idea: ``traits`` has h if it matches a hobby, o if the recipient
    owns one already, m if it is handmade."""
    return (
        f"Gift idea {i}: a scarf. "
        + ("It matches a hobby the recipient has. " if "h" in traits else "")
        + ("The recipient already owns one. " if "o" in traits else "")
        + ("It is handmade." if "m" in traits else "It was mass-produced.")
    )


def outcome_of(text: str) -> str:
    score = sum(marker in text for marker in ("matches a hobby", "handmade"))
    score -= "already owns" in text
    return OUTCOMES[min(max(score, 0), 2)].name


TRAITS = [
    "".join(t for t, on in zip("hom", bits, strict=False) if on)
    for bits in itertools.product([0, 1], repeat=3)
]
CASES = [case(i, TRAITS[i % len(TRAITS)]) for i in range(48)]
LABELS = [outcome_of(text) for text in CASES]


def answer(sample: str, question: Question) -> float | None:
    """A literal answerer: yes when the case states the criterion."""
    if "unanswerable" in sample:
        return None
    for text, marker in MARKERS.items():
        if text in question.text:
            has = marker in sample
            judged_alone = question.text.startswith("Policy:")
            if judged_alone and text in NEGATIVE:
                has = not has  # judged by "owns one already" alone, it falls short
            return 0.9 if has else 0.1
    return 0.5


def teach(request: GenerateRequest) -> dict[str, object]:
    text = request.prompt.removeprefix("Case:\n")
    return {
        "outcome": outcome_of(text).lower(),  # matched without regard to case
        "confidence": 0.9,
        "criteria": ["C1", "C3", "not-a-criterion"],
        "unjudged": ["C4"],
        "why": "It fits what the recipient likes.",
    }


def write(request: GenerateRequest) -> dict[str, object]:
    return {
        "checks": [
            {"text": "The gift is handmade.", "kind": "positive"},
            {"text": "the gift is handmade.", "kind": "positive"},  # a repeat
            {"text": " ", "kind": "negative"},  # blank
        ]
    }


@pytest.fixture
def fakes() -> Iterator[tuple[FakeLLM, FakeLLM, FakeLLM]]:
    teacher, writer = FakeLLM(generate=teach), FakeLLM(generate=write)
    judge = FakeLLM(answer=answer, kind="decision")
    with fake_llms(teacher=teacher, writer=writer, judge=judge):
        yield teacher, writer, judge


def model(**settings: object) -> PolicyClassifier:
    defaults: dict[str, object] = {
        "policy": GIFTS,
        "answerer": "fake:judge",
        "teacher": "fake:teacher",
        "cv": 3,
    }
    return PolicyClassifier(**{**defaults, **settings})  # pyright: ignore[reportArgumentType]  # settings of the right types


def test_a_policy_is_checked() -> None:
    with pytest.raises(ValueError, match="2 to 4 outcomes"):
        Policy("t", outcomes=(Outcome("ONLY"),))
    with pytest.raises(ValueError, match="2 to 4 outcomes"):
        Policy("t", outcomes=tuple(Outcome(str(i)) for i in range(5)))
    with pytest.raises(ValueError, match="must differ"):
        Policy("t", outcomes=(Outcome("A"), Outcome("A")))
    with pytest.raises(ValueError, match="ids must differ"):
        Policy(outcomes=OUTCOMES, criteria=(HOBBY, HOBBY))
    with pytest.raises(ValueError, match="text, criteria"):
        Policy(outcomes=OUTCOMES)
    with pytest.raises(ValueError, match="kind"):
        Criterion("C9", "text", "sideways")  # pyright: ignore[reportArgumentType]  # the wrong kind on purpose
    assert Policy(outcomes=OUTCOMES, criteria=(HOBBY,)).names == ("NO", "MAYBE", "YES")
    missing = pd.Series(["YES", None, np.nan, pd.NA], dtype=object)
    assert outcome_codes(GIFTS, missing, 4).tolist() == [2, -1, -1, -1]


def test_templates_build_checks_per_criterion_and_boundary() -> None:
    checks = policy_checks(GIFTS, ("reach", "applies", "met"))
    ids = [c.id for c in checks if c.group == "C2"]
    assert ids == ["C2.reach1", "C2.reach2", "C2.applies", "C2.met"]
    assert "Judged by this policy alone: the recipient will love the gift." in next(
        c.question for c in checks if c.id == "C1.reach2"
    )
    bounds = dict(
        zip([c.id for c in checks], weight_bounds(checks, "criteria"), strict=False)
    )
    assert bounds["C2.reach1"] == (0.0, np.inf)  # reaching an outcome only raises
    assert bounds["C2.met"] == (-np.inf, 0.0)  # a negative criterion only lowers
    assert bounds["C1.met"] == (0.0, np.inf)
    assert bounds["C2.applies"] == bounds["C4.met"] == (-np.inf, np.inf)
    assert set(weight_bounds(checks, "free")) == {(-np.inf, np.inf)}
    whole = policy_checks(Policy("Love it or not.", OUTCOMES), ("reach",))
    assert [c.id for c in whole] == ["policy.reach1", "policy.reach2"]


def test_it_learns_the_gift_policy_from_the_teacher(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    clf = model().fit(CASES)
    assert list(clf.outcomes_) == ["NO", "MAYBE", "YES"]  # the policy's order
    assert (clf.predict(CASES) == np.array(LABELS)).mean() == 1.0
    proba = clf.predict_proba(CASES)
    assert proba.shape == (48, 3) and np.allclose(proba.sum(axis=1), 1)
    reach = np.cumsum(proba[:, ::-1], axis=1)[:, ::-1]
    assert (np.diff(reach, axis=1) <= 1e-12).all()  # P(>=NO) >= P(>=MAYBE) >= P(YES)
    assert set(clf.labels_["source"]) == {"teacher"}
    assert clf.labels_["criteria"][0] == ("C1", "C3")  # unknown ids dropped
    report = clf.report_
    assert report["cases"] == {"gold": 0, "teacher": 48, "unlabelled": 0}
    oof = report["out_of_fold"]
    assert isinstance(oof, dict) and oof["gold"] is None
    assert oof["teacher"]["exact"] == 1.0
    assert report["teacher_consistency"] is None


def test_your_labels_are_never_sent_to_the_teacher(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    teacher, _, _ = fakes
    y = [label if i % 3 == 0 else None for i, label in enumerate(LABELS)]
    clf = model(teacher_check=0).fit(CASES, y)
    judged = {c.prompt for c in teacher.calls}
    gold = {f"Case:\n{text}" for text, label in zip(CASES, y, strict=True) if label}
    assert len(judged) == 32 and not judged & gold
    assert (clf.labels_["source"] == "gold").sum() == 16
    report = clf.report_
    assert report["cases"] == {"gold": 16, "teacher": 32, "unlabelled": 0}
    assert "policy_vs_labels" not in report
    assert all(c.prefix and "C1 [positive]" in c.prefix for c in teacher.calls)


def test_the_teacher_is_checked_against_your_labels(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    teacher, _, _ = fakes
    y = [label if i % 3 == 0 else None for i, label in enumerate(LABELS)]
    y[0] = "YES"  # the teacher says NO here
    priced = LLM("fake:teacher", price=Price(1.0, 4.0))
    usage = model(teacher=priced, teacher_check=10).estimate_cost(CASES, y)
    assert {(r.role, r.calls) for r in usage} >= {("teacher", 32 + 10)}
    clf = model(teacher_check=10).fit(CASES, y)
    assert teacher.n_calls == 32 + 10
    assert clf.policy_vs_labels_ is not None and clf.policy_vs_labels_["n"] == 10
    assert clf.report_["policy_vs_labels"] == clf.policy_vs_labels_
    # Measured only: your labels stay, and the teacher's verdicts are not used.
    assert (clf.labels_["source"] == "gold").sum() == 16
    assert clf.labels_["outcome"][0] == "YES"
    everything = model(teacher_check=100).fit(CASES, y)
    assert everything.report_["policy_vs_labels"] == {
        "n": 16,
        "exact": 15 / 16,
        "kappa": everything.policy_vs_labels_["kappa"],  # pyright: ignore[reportOptionalSubscript]  # set above
    }
    assert model(teacher=None).fit(CASES, LABELS).policy_vs_labels_ is None


def test_a_criterion_the_teacher_cannot_judge_is_dropped(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    _, _, judge = fakes
    clf = model().fit(CASES)
    assert "C4" not in set(clf.checks_["group"])
    assert clf.dropped_criteria_ == ("C4",) and clf.unjudged_["C4"] == 1.0
    # Its own checks quote it; the whole-policy checks still read the policy as written.
    quoted = f'"{LOCAL.text}"'
    assert not any(quoted in q for c in judge.calls for q in c.questions)
    assert clf.explain(CASES[:1])[0].unassessed == ("C4",)


def test_reasons_add_up_to_the_points(fakes: tuple[FakeLLM, FakeLLM, FakeLLM]) -> None:
    clf = model().fit(CASES)
    for explained, predicted in zip(
        clf.explain(CASES), clf.predict(CASES), strict=True
    ):
        assert explained.baseline + sum(explained.reasons.values()) == explained.points
        assert np.isclose(sum(explained.checks.values()), explained.points)
        assert explained.outcome == predicted
        assert set(explained.reasons) <= {"C1", "C2", "C3", "policy"}
        assert list(explained.cutoffs) == ["MAYBE", "YES"]
    [loved] = clf.explain([case(99, "hm")])
    assert loved.outcome == "YES" and loved.reasons["C1"] > 0
    _, unknown = clf.explain([CASES[0], "unanswerable"])
    assert set(unknown.unassessed) >= {"C4", "C1", "C2", "C3"}
    assert unknown.points == 0.0  # no answers: an average case


def test_refit_calls_no_model(fakes: tuple[FakeLLM, FakeLLM, FakeLLM]) -> None:
    teacher, _, judge = fakes
    clf = model().fit(CASES)
    calls, cost = teacher.n_calls + judge.n_calls, clf.usage_.calls
    points = clf.checks_.set_index("id")["points"]
    clf.refit(mute=["C1"], lock=["C3.applies"])
    after = clf.checks_.set_index("id")["points"]
    assert teacher.n_calls + judge.n_calls == calls and clf.usage_.calls == cost
    assert (after[["C1.reach1", "C1.reach2", "C1.applies"]] == 0).all()
    assert np.isclose(after["C3.applies"], points["C3.applies"])
    assert clf.report_["muted"] == ["C1"] and clf.report_["locked"] == ["C3.applies"]
    relabelled = ["YES"] + [None] * 47
    clf.refit(labels=[*relabelled[:1], *LABELS[1:]])
    assert clf.labels_["source"][0] == "gold" and clf.labels_["outcome"][0] == "YES"
    assert clf.labels_["source"][1] == "teacher"
    with pytest.raises(ValueError, match="ids"):
        clf.refit(mute=["C99"])
    labels = clf.labels_
    with pytest.raises(DataError, match="two different outcomes"):
        clf.refit(labels=["NO"] * 48)
    assert clf.labels_ is labels  # a failed refit changes nothing
    # Locked while other cases are left out: its points stay exactly as they were.
    before = clf.checks_.set_index("id")["points"]["C3.applies"]
    handmade = [i for i, text in enumerate(CASES) if "handmade" in text][:10]
    fewer = [None if i in handmade else label for i, label in enumerate(LABELS)]
    clf.refit(labels=fewer, lock=["C3.applies"])  # the handmade answers' spread moves
    assert clf.checks_.set_index("id")["points"]["C3.applies"] == before


def test_refit_says_which_new_labels_it_cannot_use(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM], caplog: pytest.LogCaptureFixture
) -> None:
    def stuck(request: GenerateRequest) -> dict[str, object]:
        verdict = teach(request)
        if "Gift idea 0:" in request.prompt or "Gift idea 1:" in request.prompt:
            verdict["outcome"] = "NOT AN OUTCOME"  # no usable verdict
        return verdict

    with fake_llms(teacher=FakeLLM(generate=stuck)):
        clf = model(teacher_check=0).fit(CASES)
    assert clf.labels_["outcome"][:2].isna().all()
    with caplog.at_level(logging.WARNING):
        clf.refit(labels=LABELS)
    assert "2 labelled cases are left out" in caplog.text


def test_sample_weights_multiply_the_teachers(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    pool = [case(100 + i, "hm") for i in range(5)]
    labels = label_cases(
        GIFTS, CASES[:24], "fake:teacher", pool=pool, min_per_outcome=6
    )
    clf = model().fit(labels["case"], labels["outcome"], sample_weight=labels["weight"])
    assert np.allclose(clf.labels_["sample_weight"], labels["weight"])
    assert np.allclose(clf.labels_["weight"], labels["weight"])
    assert (clf.labels_["weight"] < 1).any()
    weights = np.linspace(0.5, 1.5, 48)
    clf = model(teacher_check=0).fit(CASES, sample_weight=weights)
    assert np.allclose(clf.labels_["weight"], weights)
    clf.refit(labels=["YES", *LABELS[1:]])
    assert clf.labels_["weight"][0] == clf.labels_["sample_weight"][0] == 0.5
    with pytest.raises(DataError, match="sample_weight"):
        model().fit(CASES, sample_weight=[1.0] * 3)
    with pytest.raises(DataError, match="sample_weight"):
        model().fit(CASES, sample_weight=[-1.0] * 48)


def test_it_warns_when_one_outcome_has_nearly_every_label(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM], caplog: pytest.LogCaptureFixture
) -> None:
    lopsided = ["NO"] * 45 + ["MAYBE"] * 3
    with caplog.at_level(logging.WARNING):
        model(teacher=None).fit(CASES, lopsided)
    assert "94% of the training labels are NO" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model(teacher=None).fit(CASES, LABELS)
    assert "training labels" not in caplog.text


def test_a_check_with_one_answer_for_every_case_gets_no_weight(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    wrapped = Criterion("Q1", "The gift is wrapped in paper.", "positive")
    clf = model(checks=(wrapped,)).fit(CASES)  # the answerer says 0.5 to all
    [row] = clf.checks_[clf.checks_["id"] == "Q1"].to_dict("records")
    assert row["flat"] and not row["used"] and row["points"] == 0
    assert not clf.checks_[clf.checks_["group"] == "C1"]["flat"].any()


def test_scikit_learn_scorers_read_the_probabilities(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    clf = model(teacher=None).fit(CASES, LABELS)
    assert list(clf.classes_) == ["MAYBE", "NO", "YES"]  # sorted, as scikit-learn's
    assert list(clf.outcomes_) == ["NO", "MAYBE", "YES"]  # the policy's, lowest first
    proba = clf.predict_proba(CASES)
    assert (clf.classes_[proba.argmax(axis=1)] == clf.predict(CASES)).all()
    [explained] = clf.explain(CASES[5:6])
    assert list(explained.probabilities) == ["NO", "MAYBE", "YES"]
    assert explained.probabilities["MAYBE"] == proba[5, 0]
    assert get_scorer("roc_auc_ovr")(clf, CASES, LABELS) == 1.0
    out_of_fold = np.asarray(
        cross_val_predict(  # pyright: ignore[reportUnknownArgumentType]  # partial annotation
            model(teacher=None), CASES, LABELS, cv=2, method="predict_proba"
        ),
        dtype=float,
    )
    assert roc_auc_score(LABELS, out_of_fold, multi_class="ovr") > 0.99
    keep = Policy(GIFTS.text, (Outcome("RETURN"), Outcome("KEEP")), (HOBBY, OWNS))
    kept = ["KEEP" if "matches a hobby" in text else "RETURN" for text in CASES]
    two = model(policy=keep, teacher=None).fit(CASES, kept)
    assert list(two.classes_) == ["KEEP", "RETURN"]
    assert get_scorer("roc_auc")(two, CASES, kept) == 1.0


@pytest.mark.parametrize("decision", ["likely", "balanced"])
def test_each_outcome_falls_in_its_cutoffs(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM], decision: str
) -> None:
    clf = model(decision=decision).fit(CASES)
    rng = np.random.default_rng(0)
    answers = rng.uniform(0, 1, (300, len(clf.checks_)))  # cases of every score
    fitted = clf.scorer_
    # Close cut-points: MAYBE is never the most probable, so P(>= k) = 0.5 is no
    # longer where the most probable outcome changes.
    close = replace(fitted, thresholds=np.array([0.0, 0.3]) + fitted.thresholds[0])
    for scorer in (fitted, close):
        clf.scorer_ = scorer
        for why in [*clf.explain(CASES), *clf._explanations(answers)]:
            cutoffs = list(why.cutoffs.values())
            band = sum(why.points >= c for c in cutoffs)
            assert clf.outcomes_[band] == why.outcome
            assert cutoffs == sorted(cutoffs)


@pytest.mark.asyncio
async def test_async_twins(fakes: tuple[FakeLLM, FakeLLM, FakeLLM]) -> None:
    clf = await model().afit(CASES)
    [explained] = await clf.aexplain(CASES[5:6])
    assert [explained.outcome] == list(await clf.apredict(CASES[5:6]))


def test_save_and_load_round_trip(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM], tmp_path: Path
) -> None:
    clf = model(decision="balanced").fit(CASES)
    loaded = PolicyClassifier.load(clf.save(tmp_path / "gifts"))
    assert loaded.get_params()["policy"] == GIFTS
    assert (loaded.predict_proba(CASES) == clf.predict_proba(CASES)).all()
    assert (loaded.predict(CASES) == clf.predict(CASES)).all()
    assert loaded.labels_.equals(clf.labels_) and loaded.report_ == clf.report_
    assert loaded.explain(CASES[:2]) == clf.explain(CASES[:2])
    loaded.refit()  # the answers came back too


def test_it_keeps_the_library_contract(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM], tmp_path: Path
) -> None:
    def make() -> PolicyClassifier:
        judge = LLM("fake:judge", api_key="sk-policy-secret")
        return PolicyClassifier(GIFTS, judge, teacher="fake:teacher", cv=3)

    check_classifier(make, CASES, LABELS, tmp_path)


def test_the_writer_adds_checks_from_the_mistakes(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    _, writer, _ = fakes
    partial = Policy(GIFTS.text, OUTCOMES, (HOBBY, OWNS))  # handmade is missing
    without = model(policy=partial).fit(CASES)
    clf = model(policy=partial, writer="fake:writer").fit(CASES)
    [call] = writer.calls
    assert call.prompt and "model said" in call.prompt and "Gift idea" in call.prompt
    written = clf.checks_[clf.checks_["form"] == "written"]
    assert list(written["id"]) == ["W1-01"] and bool(written["used"].iloc[0])
    assert (clf.predict(CASES) == np.array(LABELS)).mean() == 1.0
    assert (without.predict(CASES) == np.array(LABELS)).mean() < 1.0


def test_two_draws_measure_the_teacher(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    draws = itertools.count()

    def unsure(request: GenerateRequest) -> dict[str, object]:
        verdict = teach(request)
        if next(draws) % 5 == 0:
            verdict["outcome"] = "MAYBE"
        return verdict

    with fake_llms(teacher=FakeLLM(generate=unsure)):
        clf = model(teacher_draws=2).fit(CASES)
    labels = clf.labels_
    disagree = labels["second"] != labels["outcome"]
    assert disagree.any() and (labels["weight"][disagree] == 0.5).all()
    consistency = clf.report_["teacher_consistency"]
    assert isinstance(consistency, float) and 0.5 < consistency < 1


def test_a_teacher_of_your_own(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def mine(policy: Policy, text: str) -> TeacherVerdict:
        return TeacherVerdict(outcome_of(text), why="by the rule")

    clf = model(teacher=mine).fit(CASES)
    assert (clf.predict(CASES) == np.array(LABELS)).all()
    with caplog.at_level(logging.WARNING):
        folder = clf.save(tmp_path / "mine")
    assert "cannot be saved" in caplog.text
    assert PolicyClassifier.load(folder).teacher is None

    def unknown_outcome(policy: Policy, text: str) -> TeacherVerdict:
        return TeacherVerdict("LOVE")

    with pytest.raises(DataError, match="TeacherVerdict"):
        model(teacher=unknown_outcome).fit(CASES)


def test_balanced_cut_offs(fakes: tuple[FakeLLM, FakeLLM, FakeLLM]) -> None:
    clf = model(decision="balanced").fit(CASES)
    assert clf.cut_points_ is not None and len(clf.cut_points_) == 2
    expected = clf.predict_proba(CASES) @ np.arange(3)
    reached = (expected[:, None] >= clf.cut_points_[None, :] - 1e-12).sum(axis=1)
    assert (clf.classes_[reached] == clf.predict(CASES)).all()


def test_two_outcomes(fakes: tuple[FakeLLM, FakeLLM, FakeLLM]) -> None:
    yes_no = Policy(
        "Will the recipient love this gift?",
        (Outcome("NO"), Outcome("YES")),
        (HOBBY, OWNS, HANDMADE),
    )
    labels = ["YES" if label == "YES" else "NO" for label in LABELS]
    clf = model(policy=yes_no, teacher=None).fit(CASES, labels)
    assert (clf.predict(CASES) == np.array(labels)).all()
    assert clf.predict_proba(CASES).shape == (48, 2)


def test_problems_are_caught_before_any_call(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    teacher, _, judge = fakes
    with pytest.raises(DataError, match="no label"):
        model(teacher=None).fit(CASES, [None] * 48)
    with pytest.raises(DataError, match="not the policy's outcomes"):
        model().fit(CASES, ["LOVE"] * 48)
    with pytest.raises(ValueError, match="templates"):
        model(templates=("vibes",)).fit(CASES)
    with pytest.raises(ValueError, match="needs policy"):
        PolicyClassifier(answerer="fake:judge").fit(CASES, LABELS)
    with pytest.raises(ValueError, match="writer"):
        model(writer_rounds=1).fit(CASES)
    with pytest.raises(ValueError, match="kept for whole-policy"):
        model(checks=(Criterion("policy", "Is it wrapped?"),)).fit(CASES)
    assert teacher.n_calls == judge.n_calls == 0


def test_cost_estimates_and_the_cap(fakes: tuple[FakeLLM, FakeLLM, FakeLLM]) -> None:
    teacher, _, judge = fakes
    priced = model(
        teacher=LLM("fake:teacher", price=Price(1.0, 4.0)),
        answerer=LLM("fake:judge", price=Price(1.0)),
    )
    usage = priced.estimate_cost(CASES, [None] * 48)
    calls = {(r.role, r.calls) for r in usage}
    assert ("teacher", 48) in calls and ("answerer", 48) in calls
    assert teacher.n_calls == judge.n_calls == 0
    priced.max_cost_usd = 1e-6
    with pytest.raises(BudgetExceededError):
        priced.fit(CASES)


def test_label_cases_tops_up_a_short_outcome(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    few = [text for text in CASES if outcome_of(text) != "YES"][:20]
    pool = [case(100 + i, "hm") for i in range(5)] + CASES[:20]
    usage = Usage()
    labels = label_cases(
        GIFTS,
        few,
        "fake:teacher",
        y=[outcome_of(few[0]), *[None] * 19],
        pool=pool,
        min_per_outcome=3,
        usage=usage,
    )
    assert labels["source"][0] == "gold" and usage.calls > 0
    added = labels[labels["pool"]]
    assert len(added) >= 3 and set(added["outcome"]) == {"YES"}
    yes = labels[labels["outcome"] == "YES"]
    assert (yes["sampling_weight"] < 1).all()
    assert (labels[labels["outcome"] == "NO"]["sampling_weight"] == 1).all()
    assert list(labels.columns[:2]) == ["case", "outcome"]


def test_answer_checks_returns_a_matrix(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    answers = answer_checks(
        [CASES[5], "unanswerable"], [HOBBY.text, HANDMADE.text], "fake:judge"
    )
    assert answers.shape == (2, 2) and np.isnan(answers[1]).all()
    assert np.allclose(
        answers[0],
        [0.9 if m in CASES[5] else 0.1 for m in ("matches a hobby", "handmade")],
    )


def test_write_checks_names_and_dedupes(
    fakes: tuple[FakeLLM, FakeLLM, FakeLLM],
) -> None:
    mistakes = [Mistake(CASES[5], "YES", "MAYBE", "Handmade gifts are loved.")]
    checks = write_checks(GIFTS, mistakes, "fake:writer", prefix="N")
    assert checks == (Criterion("N01", "The gift is handmade.", "positive"),)
    again = write_checks(
        GIFTS, mistakes, "fake:writer", current=["The gift is handmade."]
    )
    assert again == ()
    assert write_checks(GIFTS, [], "fake:writer") == ()
