"""GPTree offline: growth, paper defaults, routing, the critic, the contract."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from think_reason_learn.gptree import GPTree
from think_reason_learn.gptree._gptree import _sample_weights
from think_reason_learn.gptree._split import Split, best_split, impurity
from think_reason_learn.llm import ChoiceQuestion, Question, config_context
from think_reason_learn.testing import (
    FakeLLM,
    GenerateRequest,
    fake_llms,
)

FOUNDERS = [
    "an engineer from MIT",
    "a lawyer from Yale",
    "an engineer and second-time founder",
    "a chef from Lyon",
    "an engineer and designer",
    "a banker in London",
    "an engineer who sold a company",
    "a lawyer turned investor",
]
LABELS = ["hit" if "engineer" in f else "miss" for f in FOUNDERS]
QUESTIONS: dict[str, object] = {
    "questions": [
        {"text": "Is the founder an engineer?"},
        {"text": "Did the founder study law?"},
    ]
}


def write(request: GenerateRequest) -> str | dict[str, object]:
    if request.schema is not None:
        return QUESTIONS
    return "Engineers tend to do well."


def answer(sample: str, question: Question) -> str | None:
    if isinstance(question, ChoiceQuestion):
        return "big" if "engineer" in sample else "small"
    if "engineer" in question.text:
        return "YES" if "engineer" in sample else "NO"
    return "YES" if "lawyer" in sample else "NO"


tree = partial(
    GPTree,
    task="Will the founder succeed?",
    generator="fake:writer",
    answerer="fake:judge",
)


@pytest.fixture
def scripted() -> tuple[FakeLLM, FakeLLM]:
    return FakeLLM(generate=write), FakeLLM(answer=answer)


def test_splits_on_the_question_that_separates_the_classes(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    model = tree().fit(FOUNDERS, LABELS)
    root = model.tree_.nodes[0]
    assert (
        root.question is not None
        and root.question.text == "Is the founder an engineer?"
    )
    assert model.tree_.n_leaves == 2
    assert list(model.predict(["an engineer in Berlin", "a poet"])) == ["hit", "miss"]
    assert "Is the founder an engineer?" in model.export_graphviz()
    assert model.tree_.to_frame(list(model.classes_)).loc[0, "n_samples"] == 8


def test_paper_defaults() -> None:
    model = GPTree()
    assert (model.max_depth, model.min_samples_leaf, model.max_choices) == (
        18,
        0.005,
        2,
    )
    assert model.insights and model.answerer == "jev"


def test_prompts_hold_samples_not_answers(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    writer, _ = fakes
    mixed = [
        *LABELS[:4],
        "miss",
        *LABELS[5:],
    ]  # one engineer missed: the YES child splits again
    tree(max_depth=3, min_samples_leaf=1).fit(FOUNDERS, mixed)
    node_prompts = [c.prompt or "" for c in writer.calls if c.schema is not None]
    assert len(node_prompts) >= 2
    child = node_prompts[1]
    assert "- Is the founder an engineer? -> " in child
    assert "label: " in child and ": YES" not in child.split("Examples at this node")[1]


def test_unanswered_samples_follow_the_largest_branch() -> None:
    def flaky(sample: str, question: Question) -> str | None:
        return None if sample == "a chef from Lyon" else answer(sample, question)

    with fake_llms(writer=FakeLLM(generate=write), judge=FakeLLM(answer=flaky)):
        model = tree().fit(FOUNDERS, LABELS)
    root, children = model.tree_.nodes[0], model.tree_.nodes[1:]
    assert sum(c.n_samples for c in children) == root.n_samples == 8


def test_critic_reviews_without_adding_questions() -> None:
    def critic(request: GenerateRequest) -> dict[str, object]:
        return {"questions": [{"text": "Is the founder an engineer?"}]}

    reviewer = FakeLLM(generate=critic)
    with fake_llms(
        writer=FakeLLM(generate=write), judge=FakeLLM(answer=answer), reviewer=reviewer
    ):
        model = tree(critic="fake:reviewer").fit(FOUNDERS, LABELS)
    assert reviewer.n_calls >= 1
    assert model.tree_.nodes[0].candidates == ("Is the founder an engineer?",)
    assert "critic" in set(model.usage_.by("role")["role"])


def test_choice_questions_when_allowed() -> None:
    def write_choice(request: GenerateRequest) -> str | dict[str, object]:
        if request.schema is None:
            return "notes"
        return {
            "questions": [
                {
                    "text": "How large is the founder's network?",
                    "options": ["big", "small", "none"],
                }
            ]
        }

    with fake_llms(writer=FakeLLM(generate=write_choice), judge=FakeLLM(answer=answer)):
        model = tree(max_choices=3).fit(FOUNDERS, LABELS)
    root = model.tree_.nodes[0]
    assert isinstance(root.question, ChoiceQuestion)
    assert set(root.children) == {"big", "small"}  # the empty "none" branch is dropped


def test_leaf_and_size_limits(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    assert tree(max_leaf_nodes=1).fit(FOUNDERS, LABELS).tree_.n_leaves == 1
    assert tree(min_samples_leaf=0.6).fit(FOUNDERS, LABELS).tree_.n_leaves == 1
    assert tree(max_depth=0).fit(FOUNDERS, LABELS).tree_.n_leaves == 1


def test_write_instructions_then_fit_reuses_them(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    writer, _ = fakes
    model = tree(insights=False)
    assert model.write_instructions() == "Engineers tend to do well."
    model.instructions = "Look at the founder's training."
    model.fit(FOUNDERS, LABELS)
    assert model.instructions_ == "Look at the founder's training."
    assert (
        sum(c.schema is None for c in writer.calls) == 1
    )  # no second instructions call


def test_insights_summarise_each_class(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    model = tree().fit(FOUNDERS, LABELS)
    assert model.insights_.startswith("hit:") and "miss:" in model.insights_
    assert tree(insights=False).fit(FOUNDERS, LABELS).insights_ == ""


def test_predicting_training_samples_reuses_answers(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    _, judge = fakes
    model = tree().fit(FOUNDERS, LABELS)
    before = judge.n_calls
    model.predict(FOUNDERS)
    assert judge.n_calls == before


def test_prune_and_apply(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    model = tree().fit(FOUNDERS, LABELS)
    assert set(model.apply(FOUNDERS)) == {1, 2}
    model.prune(0)
    assert model.tree_.n_leaves == 1 and set(model.apply(FOUNDERS)) == {0}


def test_task_is_required(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    with pytest.raises(ValueError, match="needs task"):
        GPTree(generator="fake:writer", answerer="fake:judge").fit(FOUNDERS, LABELS)


class TestSplit:
    def test_impurity(self) -> None:
        assert impurity(np.array([5.0, 5.0]), "gini") == 0.5
        assert impurity(np.array([5.0, 5.0]), "entropy") == 1.0
        assert impurity(np.array([4.0, 0.0]), "gini") == 0.0

    def test_best_split_and_limits(self) -> None:
        y = np.array([0, 0, 1, 1])
        w = np.ones(4)
        answers = {
            "good": ["YES", "YES", "NO", "NO"],
            "noise": ["YES", "NO", "YES", "NO"],
        }

        def split(min_leaf: int) -> Split | None:
            return best_split(
                answers,
                y,
                w,
                n_classes=2,
                criterion="gini",
                min_samples_leaf=min_leaf,
                min_impurity_decrease=0.0,
                total_weight=4.0,
            )

        best = split(1)
        assert best is not None and (best.qid, best.decrease) == ("good", 0.5)
        assert split(3) is None

    def test_balanced_class_weights(self) -> None:
        y = np.array([0, 0, 0, 1])
        weights = _sample_weights(y, np.array(["a", "b"], dtype=object), "balanced")
        assert weights.tolist() == [4 / 6, 4 / 6, 4 / 6, 2.0]


def test_prune_keeps_surviving_ids(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    mixed = [*LABELS[:4], "miss", *LABELS[5:]]
    model = tree(max_depth=3, min_samples_leaf=1).fit(FOUNDERS, mixed)
    before = [
        (n.id, n.question.text if n.question else None) for n in model.tree_.nodes
    ]
    leaf = next(n.id for n in model.tree_.nodes if n.is_leaf)
    model.prune(leaf)
    after = [(n.id, n.question.text if n.question else None) for n in model.tree_.nodes]
    assert after == before


def test_a_node_without_answers_does_not_fail(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    model = tree().fit(FOUNDERS, LABELS)
    with fake_llms(judge=FakeLLM(answer=lambda sample, question: None)):
        assert len(model.predict(["someone new"])) == 1


def test_two_classes_use_the_papers_leaf_threshold(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    model = tree().fit(FOUNDERS, LABELS)
    assert model.threshold_ is not None
    three = tree().fit(FOUNDERS, [*LABELS[:7], "other"])
    assert three.threshold_ is None


def test_leaf_budget_stops_growth_early(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    writer, _ = fakes
    mixed = [*LABELS[:4], "miss", *LABELS[5:]]
    tree(max_leaf_nodes=2, min_samples_leaf=1, insights=False).fit(FOUNDERS, mixed)
    assert sum(c.schema is not None for c in writer.calls) == 1


def test_bad_settings_fail_before_any_call(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    writer, judge = fakes
    with pytest.raises(ValueError, match="pos_label"):
        tree(pos_label="HIT").fit(FOUNDERS, LABELS)
    with pytest.raises(ValueError, match="example_class_ratio names unknown"):
        tree(example_class_ratio={"HIT": 1.0}).fit(FOUNDERS, LABELS)
    assert writer.n_calls == judge.n_calls == 0


def test_a_failed_critic_keeps_the_written_questions() -> None:
    with fake_llms(
        writer=FakeLLM(generate=write),
        judge=FakeLLM(answer=answer),
        reviewer=FakeLLM(generate=lambda request: None),
    ):
        model = tree(critic="fake:reviewer").fit(FOUNDERS, LABELS)
    assert not model.tree_.nodes[0].is_leaf


def test_a_failed_insight_is_left_out() -> None:
    def no_hit_insight(request: GenerateRequest) -> str | dict[str, object] | None:
        return None if "Samples labelled hit" in request.prompt else write(request)

    with fake_llms(
        writer=FakeLLM(generate=no_hit_insight), judge=FakeLLM(answer=answer)
    ):
        model = tree().fit(FOUNDERS, LABELS)
    assert model.insights_.startswith("miss:") and "hit:" not in model.insights_


def test_yes_no_options_make_a_yes_no_question(fakes: tuple[FakeLLM, FakeLLM]) -> None:
    options: dict[str, object] = {
        "questions": [{"text": "Is the founder an engineer?", "options": ["Yes", "No"]}]
    }
    with fake_llms(
        writer=FakeLLM(generate=lambda r: options if r.schema else "x"),
        judge=FakeLLM(answer=answer),
    ):
        model = tree().fit(FOUNDERS, LABELS)
    question = model.tree_.nodes[0].question
    assert question is not None and question.text == "Is the founder an engineer?"


def test_a_node_whose_questions_fail_stays_a_leaf() -> None:
    writer = FakeLLM(generate=lambda r: None if r.schema else "Some guidance.")
    with fake_llms(writer=writer, judge=FakeLLM(answer=answer)):
        model = tree().fit(FOUNDERS, LABELS)  # no usable questions at the root
    assert len(model.tree_.nodes) == 1 and model.tree_.nodes[0].is_leaf


def test_class_weights_by_label_and_unknown_labels() -> None:
    y = np.array([0, 0, 1])
    weights = _sample_weights(y, np.array([0, 1]), {1: 5.0})
    assert list(weights) == [1.0, 1.0, 5.0]
    with pytest.raises(ValueError, match="unknown classes"):
        _sample_weights(y, np.array(["no", "yes"]), {"YES": 5.0})


def test_a_budget_stop_inside_the_tree_stops_the_fit(
    fakes: tuple[FakeLLM, FakeLLM],
) -> None:
    from think_reason_learn.exceptions import BudgetExceededError
    from think_reason_learn.llm import LLM, Price

    priced = LLM("fake:judge", price=Price(input=1e6))  # a dollar a token
    with config_context(max_cost_usd=0.01), pytest.raises(BudgetExceededError):
        tree(answerer=priced).fit(FOUNDERS, LABELS)


def test_the_plan_bounds_a_tree_that_peels_one_sample_a_level() -> None:
    founders = [f"founder {i}" for i in range(12)]
    labels = ["hit" if i % 3 == 0 else "miss" for i in range(12)]

    def ask_each(request: GenerateRequest) -> str | dict[str, object]:
        if request.schema is None:
            return "Notes."
        return {"questions": [{"text": f"Is this {f}?"} for f in founders]}

    def is_it(sample: str, question: Question) -> str:
        return "YES" if question.text == f"Is this {sample}?" else "NO"

    judge = FakeLLM(answer=is_it)
    model = tree(max_questions=12, min_samples_leaf=1, insights=False)
    with (
        fake_llms(writer=FakeLLM(generate=ask_each), judge=judge),
        config_context(cache=False),
    ):
        planned = model.estimate_cost(founders, labels)
        model.fit(founders, labels)
    assert model.tree_.n_leaves > 3  # one hit peeled off per level
    assert judge.n_calls <= sum(r.calls for r in planned if r.role == "answerer")
