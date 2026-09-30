"""Offline tests for GPTree answering with Jev (choice questions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pandas as pd
import pytest

from think_reason_learn.core.llms import (
    ChoiceQuestion,
    JevBudget,
    JevChoice,
    LLMChoice,
    OpenAIChoice,
)
from think_reason_learn.core.llms._schemas import LLMResponse
from think_reason_learn.gptree import GPTree
from think_reason_learn.gptree import _gptree as gptree_module
from think_reason_learn.gptree._gptree import Question, Questions
from think_reason_learn.gptree._prompts import num_questions_tag
from tests.fake_jev import FakeJevLLM

pytestmark = pytest.mark.usefixtures("typesafe_key")

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]
ROWS = [f"strong founder {i}" for i in range(4)] + [
    f"weak founder {i}" for i in range(4)
]
LABELS = ["successful"] * 3 + ["failed"] * 4 + ["successful"]
TECH = Question(
    value="Does the founder have a strong technical background?",
    choices=["yes", "no"],
    question_type="INFERENCE",
)
STAGE = Question(
    value="What stage is the company at?",
    choices=["early", "late", "growth"],
    question_type="INFERENCE",
)


def _choose(state: str, question: ChoiceQuestion) -> str | None:
    if question.instructions == TECH.value:
        return "yes" if "strong" in state else "no"
    return question.labels[0]


class FakeTreeLLM(FakeJevLLM):
    """Question generation for GPTree plus Jev-like answering."""

    async def respond(
        self,
        query: str,
        llm_priority: List[Any],
        response_format: Any,
        instructions: Any = None,
        temperature: Any = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        self.calls.append({"query": query, "response_format": response_format})
        if response_format is Questions:
            response: Any = Questions(questions=[TECH, STAGE], cumulative_memory="m")
        else:  # a GPTree answer model with a Literal of the choices
            labels = response_format.model_fields["answer"].annotation.__args__
            response = response_format(answer=labels[0])
        return LLMResponse(
            response=response,
            logprobs=[],
            total_tokens=10,
            provider_model=OpenAIChoice(model="gpt-4.1-nano"),
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeTreeLLM:
    fake = FakeTreeLLM(choose=_choose)
    monkeypatch.setattr(gptree_module, "llm", fake)
    return fake


def _tree(tmp_path: Path, **kwargs: Any) -> GPTree:
    return GPTree(
        qgen_llmc=CHAT,
        critic_llmc=CHAT,
        qgen_instr_llmc=CHAT,
        qanswer_llmc=kwargs.pop("qanswer_llmc", [JevChoice(max_cost_usd=4.0)]),
        min_question_candidates=2,
        max_question_candidates=2,
        save_path=tmp_path,
        name="jev_tree",
        **kwargs,
    )


async def _fit(tree: GPTree) -> GPTree:
    await tree.set_tasks(
        instructions_template=f"Ask {num_questions_tag} questions about founders."
    )
    async for _ in tree.fit(pd.DataFrame({"text": ROWS}), LABELS):
        pass
    return tree


@pytest.mark.asyncio
async def test_node_questions_go_in_one_request_per_sample(
    tmp_path: Path, fake: FakeTreeLLM
) -> None:
    await _fit(_tree(tmp_path, max_depth=1))

    [call] = fake.answer_calls
    assert call["n_requests"] == len(ROWS)
    for request in fake.answer_requests:
        questions = [
            q for q in request.questions.values() if isinstance(q, ChoiceQuestion)
        ]
        assert len(questions) == len(request.questions)
        assert [(q.instructions, q.labels) for q in questions] == [
            (TECH.value, TECH.choices),
            (STAGE.value, STAGE.choices),
        ]
    assert not any(c["response_format"] is not Questions for c in fake.calls)


@pytest.mark.asyncio
async def test_jev_labels_split_the_node(tmp_path: Path, fake: FakeTreeLLM) -> None:
    tree = await _fit(_tree(tmp_path, max_depth=1))

    root = tree.get_node(0)
    assert root is not None and root.question is not None
    assert root.question.value == TECH.value
    assert sorted(c.label for c in root.children) == ["no", "yes"]
    assert root.split_ratios == (4, 4)


@pytest.mark.asyncio
async def test_jev_is_sent_feature_columns_only(
    tmp_path: Path, fake: FakeTreeLLM
) -> None:
    await _fit(_tree(tmp_path, max_depth=2))

    assert len(fake.answer_calls) == 3  # the root, then its two children
    assert {r.state for r in fake.answer_requests} == {f"text: {r}" for r in ROWS}


@pytest.mark.asyncio
async def test_fit_shares_one_budget_across_nodes(
    tmp_path: Path, fake: FakeTreeLLM
) -> None:
    await _fit(_tree(tmp_path, max_depth=2))

    budgets = {id(c["budget"]) for c in fake.answer_calls}
    budget = fake.answer_calls[0]["budget"]
    assert len(budgets) == 1
    assert isinstance(budget, JevBudget) and budget.max_cost_usd == 4.0


@pytest.mark.asyncio
async def test_missing_choice_leaves_the_answer_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def choose(state: str, question: ChoiceQuestion) -> str | None:
        return None if state == f"text: {ROWS[0]}" else _choose(state, question)

    fake = FakeTreeLLM(choose=choose)
    monkeypatch.setattr(gptree_module, "llm", fake)
    tree = await _fit(_tree(tmp_path, max_depth=1))

    root = tree.get_node(0)
    assert root is not None and root.split_ratios == (3, 4)


@pytest.mark.asyncio
async def test_predict_asks_each_node_once_for_all_its_samples(
    tmp_path: Path, fake: FakeTreeLLM
) -> None:
    tree = await _fit(_tree(tmp_path, max_depth=1))
    fake.answer_calls.clear()
    fake.answer_requests.clear()

    X_new = pd.DataFrame({"text": ["strong founder x", "weak founder y", "weak z"]})
    records = [r[:4] async for r in tree.predict(X_new)]

    assert len(fake.answer_calls) == 1
    assert [
        list(r.questions.values())[0].instructions for r in fake.answer_requests
    ] == [TECH.value] * 3
    root = tree.get_node(0)
    assert root is not None
    child = {c.label: c.id for c in root.children}
    assert sorted(records) == sorted(
        [
            (0, TECH.value, "yes", 0),
            (0, "No Question", "No Answer", child["yes"]),
            (1, TECH.value, "no", 0),
            (1, "No Question", "No Answer", child["no"]),
            (2, TECH.value, "no", 0),
            (2, "No Question", "No Answer", child["no"]),
        ]
    )


@pytest.mark.asyncio
async def test_chat_answerer_keeps_the_chat_path(
    tmp_path: Path, fake: FakeTreeLLM
) -> None:
    await _fit(_tree(tmp_path, max_depth=1, qanswer_llmc=CHAT))

    assert fake.answer_requests == []
    assert any(c["response_format"] is not Questions for c in fake.calls)
