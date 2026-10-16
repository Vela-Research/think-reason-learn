"""RRF offline: generation set, filters, the vote, combiners and the contract."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import numpy as np
import pytest

from think_reason_learn.exceptions import DataError, LLMError
from think_reason_learn.llm import Question, config_context
from think_reason_learn.rrf import RRF
from think_reason_learn.rrf._selection import question_stats, rank, tune_vote, vote_grid
from think_reason_learn.testing import (
    FakeLLM,
    GenerateRequest,
    fake_llms,
)

KEYWORDS = {
    "Is the founder an engineer?": "engineer",
    "Does the founder write code?": "engineer",  # same answers as the first: redundant
    "Is the founder a lawyer?": "lawyer",
    "Did the founder exit a company?": "exit",
}
TRAITS = [
    "engineer",
    "engineer exit",
    "lawyer",
    "chef",
    "engineer",
    "banker",
    "lawyer exit",
    "poet",
]
FOUNDERS = [f"founder {i}: {TRAITS[i % len(TRAITS)]}" for i in range(48)]
LABELS = ["YES" if "engineer" in f else "NO" for f in FOUNDERS]


def write(request: GenerateRequest) -> str | dict[str, object]:
    if request.schema is None:
        return "Look at the founder's background."
    return {
        "questions": [*KEYWORDS, "is the founder an engineer?"]
    }  # a repeat, other case


def answer(sample: str, question: Question) -> str:
    keyword = KEYWORDS.get(question.text, question.text.lower())
    return "YES" if keyword in sample else "NO"


rrf = partial(
    RRF,
    task="Will the founder succeed?",
    generator="fake:writer",
    answerer="fake:judge",
    n_prompts=3,
    cv=3,
)


@pytest.fixture
def judge() -> Iterator[FakeLLM]:
    fake = FakeLLM(answer=answer)
    with fake_llms(writer=FakeLLM(generate=write), judge=fake):
        yield fake


def test_the_vote_uses_the_question_that_predicts(judge: FakeLLM) -> None:
    model = rrf().fit(FOUNDERS, LABELS)
    assert model.rules_ == ("Is the founder an engineer?",)
    assert (model.min_votes_, model.threshold_) == (1, 1.0)
    assert list(model.predict(["an engineer in Paris", "a chef"])) == ["YES", "NO"]
    assert "Does the founder write code?" not in model.rules_  # Hamming duplicate
    assert len(model.questions_) == 4  # the case-insensitive repeat is merged


def test_questions_are_written_and_scored_on_different_samples(judge: FakeLLM) -> None:
    writer = FakeLLM(generate=write)
    with fake_llms(writer=writer, judge=judge):
        rrf(generation_size=16).fit(FOUNDERS, LABELS)
    shown = {
        line
        for c in writer.calls
        if c.schema
        for line in (c.prompt or "").splitlines()
        if line.startswith("founder")
    }
    scored = {c.sample for c in judge.calls}
    assert shown and scored and not shown & scored
    assert len(scored) == len(FOUNDERS) - 16


def test_weak_questions_are_dropped(judge: FakeLLM) -> None:
    model = rrf().fit(FOUNDERS, LABELS)
    [lawyer] = model.questions_[
        model.questions_["question"] == "Is the founder a lawyer?"
    ].to_dict("records")
    assert lawyer["precision"] == 0 and not lawyer["used"]


def test_predict_asks_only_the_rules(judge: FakeLLM) -> None:
    model = rrf().fit(FOUNDERS, LABELS)
    before = judge.n_calls
    model.predict(["new one", "new two"])
    assert judge.n_calls - before == 2 * len(model.rules_)


def test_elasticnet_combiner(judge: FakeLLM) -> None:
    model = rrf(combiner="elasticnet").fit(FOUNDERS, LABELS)
    assert model.combiner_ is not None and "Is the founder an engineer?" in model.rules_
    assert list(model.predict(["an engineer", "a poet"])) == ["YES", "NO"]


def test_screening_answers_fewer_questions(judge: FakeLLM) -> None:
    rrf().fit(FOUNDERS, LABELS)
    full = judge.n_calls
    screening = FakeLLM(answer=answer)
    with fake_llms(writer=FakeLLM(generate=write), judge=screening):
        model = rrf(screening_size=8, screened_questions=1, random_state=1).fit(
            FOUNDERS, LABELS
        )
    assert screening.n_calls < full
    assert len(model.questions_) == 1


def test_expert_questions_are_scored_too(judge: FakeLLM) -> None:
    model = rrf(
        expert_questions=("Did the founder exit a company?", "Is the founder a chef?")
    ).fit(FOUNDERS, LABELS)
    assert "Is the founder a chef?" in set(model.questions_["question"])


def test_positive_class_and_two_classes_only(judge: FakeLLM) -> None:
    flipped = rrf(pos_label="NO").fit(FOUNDERS, LABELS)
    assert flipped.rules_ and "engineer" not in flipped.rules_[0]
    with pytest.raises(DataError, match="two classes"):
        rrf().fit(FOUNDERS, [str(i % 3) for i in range(48)])


def test_bad_data_and_settings_fail_before_any_call(judge: FakeLLM) -> None:
    with pytest.raises(DataError, match="No samples to write questions from"):
        rrf().fit(["an engineer", "a chef"], ["YES", "NO"])
    with pytest.raises(ValueError, match="example_class_ratio names unknown"):
        rrf(example_class_ratio={"yes": 1.0}).fit(FOUNDERS, LABELS)
    assert judge.n_calls == 0


def test_elasticnet_warns_when_it_weighs_no_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with fake_llms(
        writer=FakeLLM(generate=write),
        judge=FakeLLM(answer=lambda sample, question: 0.5, kind="decision"),
    ):
        model = rrf(combiner="elasticnet", drop_below_base_rate=False).fit(
            FOUNDERS, LABELS
        )
    assert model.rules_ == () and "Every weight is zero" in caplog.text


class TestSelection:
    y = np.array([True, True, False, False])
    answers = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)

    def test_stats(self) -> None:
        precision, recall, scores = question_stats(self.answers, self.y, 1.0)
        assert precision.tolist() == [1.0, 0.0, 0.5] and recall.tolist() == [
            1.0,
            0.0,
            1.0,
        ]
        assert scores[0] == 1.0

    def test_rank_drops_weak_and_redundant(self) -> None:
        ranked = rank(
            self.answers,
            self.y,
            beta=0.5,
            drop_below_base_rate=True,
            hamming_threshold=0.15,
        )
        assert ranked == [0]

    def test_vote_grid_and_tuning(self) -> None:
        grid = vote_grid(self.answers[:, [0, 2]], self.y, 1.0)
        assert grid[0, 0] == 1.0 and grid[1, 1] == 1.0
        ranked, n, t = tune_vote(
            self.answers,
            self.y,
            beta=0.5,
            drop_below_base_rate=True,
            hamming_threshold=0.15,
            max_rules=None,
            cv=5,
            random_state=0,
        )
        assert (ranked, n, t) == ([0], 1, 1)


def test_integer_labels_and_pos_label(judge: FakeLLM) -> None:
    labels = [1 if label == "YES" else 0 for label in LABELS]
    model = rrf(pos_label=1).fit(FOUNDERS, labels)
    assert list(model.predict(["an engineer", "a chef"])) == [1, 0]


def test_screening_reuses_screened_answers() -> None:
    screening = FakeLLM(answer=answer)
    with (
        fake_llms(writer=FakeLLM(generate=write), judge=screening),
        config_context(cache=False),
    ):
        model = rrf(screening_size=8, screened_questions=1, random_state=1).fit(
            FOUNDERS, LABELS
        )
    scored = len(FOUNDERS) - 24  # the generation set is not scored
    assert screening.n_calls == 8 * 4 + (scored - 8) * len(model.questions_)


def test_no_questions_is_an_error_not_a_silent_no() -> None:
    with (
        fake_llms(
            writer=FakeLLM(generate=lambda r: {"questions": []} if r.schema else "x"),
            judge=FakeLLM(answer=answer),
        ),
        pytest.raises(LLMError, match="no usable questions"),
    ):
        rrf().fit(FOUNDERS, LABELS)


def test_one_failed_prompt_is_skipped() -> None:
    replies = iter([None])  # the first question prompt writes nothing usable

    def flaky(request: GenerateRequest) -> str | dict[str, object] | None:
        if request.schema is None:
            return "x"
        return next(replies, None) or write(request)

    with fake_llms(writer=FakeLLM(generate=flaky), judge=FakeLLM(answer=answer)):
        model = rrf().fit(FOUNDERS, LABELS)
    assert model.rules_ == ("Is the founder an engineer?",)


def test_integer_labels_work_with_scikit_learn_metrics(judge: FakeLLM) -> None:
    y = [int(label == "YES") for label in LABELS]
    model = rrf().fit(FOUNDERS, y)
    assert model.classes_.dtype.kind == "i"
    assert (model.predict(FOUNDERS) == np.array(y)).all()
