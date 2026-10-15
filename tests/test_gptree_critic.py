"""Offline tests for GPTree's question critic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import pandas as pd
import pytest

from think_reason_learn.core.llms import JevChoice, LLMChoice, OpenAIChoice
from think_reason_learn.core.llms._schemas import LLMResponse
from think_reason_learn.gptree import GPTree
from think_reason_learn.gptree import _gptree as gptree_module
from think_reason_learn.gptree._gptree import Question, Questions
from think_reason_learn.gptree._prompts import CRITIC_INSTRUCTIONS, num_questions_tag

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]
CRITIC: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-mini")]
ROWS = [f"strong founder {i}" for i in range(4)] + [
    f"weak founder {i}" for i in range(4)
]
LABELS = ["successful"] * 3 + ["failed"] * 4 + ["successful"]
TECH = Question(
    value="Does the founder have a strong technical background?",
    choices=["yes", "no"],
    question_type="INFERENCE",
)
TECH_AGAIN = Question(
    value="Is the founder technical?", choices=["yes", "no"], question_type="INFERENCE"
)
LEAK = Question(
    value="Will the founder succeed?", choices=["yes", "no"], question_type="INFERENCE"
)
GENERATED = [TECH, TECH_AGAIN, LEAK]
REVISED = [TECH]


class FakeCriticLLM:
    """Generates GENERATED, answers by sample text, and critiques with ``critic``."""

    def __init__(self, critic: Questions | Exception | None) -> None:
        self.critic = critic
        self.calls: List[dict[str, Any]] = []

    @property
    def critic_calls(self) -> List[dict[str, Any]]:
        return [c for c in self.calls if c["llm_priority"] is CRITIC]

    async def respond(
        self,
        query: str,
        llm_priority: List[Any],
        response_format: Any,
        instructions: Any = None,
        temperature: Any = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        self.calls.append(
            {
                "query": query,
                "llm_priority": llm_priority,
                "instructions": instructions,
                "temperature": temperature,
            }
        )
        if llm_priority is CRITIC:
            if isinstance(self.critic, Exception):
                raise self.critic
            response: Any = self.critic
        elif response_format is Questions:
            response = Questions(questions=GENERATED, cumulative_memory="memory")
        else:  # a GPTree answer model with a Literal of the choices
            labels = response_format.model_fields["answer"].annotation.__args__
            response = response_format(
                answer=labels[0] if "Sample: text: strong" in query else labels[-1]
            )
        return LLMResponse(
            response=response,
            logprobs=[],
            total_tokens=10,
            provider_model=OpenAIChoice(model="gpt-4.1-nano"),
        )


def _tree(tmp_path: Path, **kwargs: Any) -> GPTree:
    return GPTree(
        qgen_llmc=CHAT,
        critic_llmc=CRITIC,
        qgen_instr_llmc=CHAT,
        qanswer_llmc=CHAT,
        critic_temperature=0.3,
        min_question_candidates=3,
        max_question_candidates=3,
        max_depth=1,
        save_path=tmp_path,
        name="critic_tree",
        **kwargs,
    )


async def _fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    critic: Questions | Exception | None,
    **kwargs: Any,
) -> tuple[GPTree, FakeCriticLLM]:
    fake = FakeCriticLLM(critic)
    monkeypatch.setattr(gptree_module, "llm", fake)
    tree = _tree(tmp_path, **kwargs)
    await tree.set_task(
        instructions_template=f"Ask {num_questions_tag} questions about founders."
    )
    await tree.fit(pd.DataFrame({"text": ROWS}), LABELS)
    return tree, fake


def _root_questions(tree: GPTree) -> List[str]:
    root = tree.get_node(0)
    assert root is not None
    return [q.value for q in root.questions]


@pytest.mark.asyncio
async def test_critic_off_makes_no_critic_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    critic = Questions(questions=REVISED, cumulative_memory="dropped two")
    tree, fake = await _fit(tmp_path, monkeypatch, critic)

    assert fake.critic_calls == []
    assert _root_questions(tree) == [q.value for q in GENERATED]


@pytest.mark.asyncio
async def test_critic_revision_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    critic = Questions(questions=REVISED, cumulative_memory="dropped two")
    tree, fake = await _fit(tmp_path, monkeypatch, critic, use_critic=True)

    [call] = fake.critic_calls
    assert call["instructions"] == CRITIC_INSTRUCTIONS
    assert call["temperature"] == 0.3
    assert all(q.value in call["query"] for q in GENERATED)  # the candidates
    assert ROWS[0] in call["query"]  # the samples the generator saw
    assert _root_questions(tree) == [q.value for q in REVISED]
    root = tree.get_node(0)
    assert root is not None and root.cumulative_memory == "memory"
    callers = {
        caller
        for count in tree.token_usage.token_counts.values()
        for caller in count.callers
    }
    assert "GPTree._critique_questions" in callers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "critic",
    [
        ValueError("provider down"),
        None,
        Questions(questions=[], cumulative_memory="all bad"),
        Questions(questions=[*GENERATED, LEAK], cumulative_memory="added one"),
    ],
    ids=["raises", "none", "empty", "too_many"],
)
async def test_misbehaving_critic_keeps_the_generated_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    critic: Questions | Exception | None,
) -> None:
    with caplog.at_level(logging.WARNING, logger=gptree_module.__name__):
        tree, fake = await _fit(tmp_path, monkeypatch, critic, use_critic=True)

    assert len(fake.critic_calls) == 1
    assert _root_questions(tree) == [q.value for q in GENERATED]
    assert any("critic" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_custom_critic_instructions_survive_save_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    critic = Questions(questions=REVISED, cumulative_memory="dropped two")
    tree, fake = await _fit(tmp_path, monkeypatch, critic, use_critic=True)
    tree._critic_instructions_template = "Be strict."
    tree.save()

    loaded = GPTree.load(tmp_path)
    assert loaded.use_critic and loaded.critic_instructions_template == "Be strict."
    assert _root_questions(loaded) == [q.value for q in REVISED]

    loaded.critic_llmc = CRITIC  # loading rebuilds the list; keep the fake's marker
    loaded.prune_tree(0)
    async for _ in loaded.resume_fit(0):
        pass
    assert fake.critic_calls[-1]["instructions"] == "Be strict."


@pytest.mark.parametrize("param", ["qgen_llmc", "qgen_instr_llmc", "critic_llmc"])
def test_jev_is_rejected_for_generation_when_the_tree_is_made(
    tmp_path: Path, param: str
) -> None:
    """Jev only answers; finding out at the first node (or never, for the critic)
    is too late."""
    llmc: dict[str, Any] = {
        "qgen_llmc": CHAT,
        "critic_llmc": CHAT,
        "qgen_instr_llmc": CHAT,
        param: [JevChoice()],
    }
    with pytest.raises(ValueError, match=f"{param}.*cannot generate"):
        GPTree(qanswer_llmc=CHAT, save_path=tmp_path, **llmc)


def test_critic_and_instruction_models_default_to_the_generation_models(
    tmp_path: Path,
) -> None:
    tree = GPTree(qgen_llmc=CHAT, qanswer_llmc=CHAT, save_path=tmp_path, name="t")

    assert tree.critic_llmc == CHAT
    assert tree.qgen_instr_llmc == CHAT
    tree.save(tmp_path / "saved")
    loaded = GPTree.load(tmp_path / "saved")
    assert loaded.critic_llmc == loaded.qgen_llmc
    assert loaded.qgen_instr_llmc == loaded.qgen_llmc
