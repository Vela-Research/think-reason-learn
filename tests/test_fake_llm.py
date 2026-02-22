"""Tests for FakeLLM dispatch — ensures both RRF and GPTree call sites work."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import create_model

from tests.fake_llm import FakeLLM
from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.rrf._rrf import Answer as RRFAnswer
from think_reason_learn.rrf._rrf import Questions as RRFQuestions
from think_reason_learn.gptree._gptree import Questions as GPTreeQuestions

_LLM_PRIORITY: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]


# ---------------------------------------------------------------------------
# RRF dispatch (backward compatibility)
# ---------------------------------------------------------------------------


class TestRRFDispatch:
    """Verify existing RRF behaviour is preserved after refactor."""

    @pytest.mark.asyncio
    async def test_rrf_questions(self) -> None:
        fake = FakeLLM(questions_per_call=3)
        resp = await fake.respond(
            query="samples",
            llm_priority=_LLM_PRIORITY,
            response_format=RRFQuestions,
        )
        assert isinstance(resp.response, RRFQuestions)
        assert len(resp.response.questions) == 3
        assert isinstance(resp.response.questions[0], str)

    @pytest.mark.asyncio
    async def test_rrf_answer(self) -> None:
        fake = FakeLLM(default_answer="YES")
        resp = await fake.respond(
            query="question",
            llm_priority=_LLM_PRIORITY,
            response_format=RRFAnswer,
        )
        assert isinstance(resp.response, RRFAnswer)
        assert resp.response.answer == "YES"

    @pytest.mark.asyncio
    async def test_rrf_set_tasks(self) -> None:
        fake = FakeLLM()
        resp = await fake.respond(
            query="Generate YES/NO questions for: classifying founders",
            llm_priority=_LLM_PRIORITY,
            response_format=str,
        )
        assert "<number_of_questions>" in str(resp.response)

    @pytest.mark.asyncio
    async def test_rrf_batch_answer(self) -> None:
        fake = FakeLLM(default_answer="YES")
        resp = await fake.respond(
            query="You are a VC analyst. Sample 0: Alice Sample 1: Bob",
            llm_priority=_LLM_PRIORITY,
            response_format=str,
        )
        assert '"answer"' in str(resp.response)


# ---------------------------------------------------------------------------
# GPTree dispatch
# ---------------------------------------------------------------------------


class TestGPTreeDispatch:
    """Verify GPTree call sites dispatch correctly."""

    @pytest.mark.asyncio
    async def test_gptree_questions(self) -> None:
        fake = FakeLLM(questions_per_call=2)
        resp = await fake.respond(
            query="samples with label successful...",
            llm_priority=_LLM_PRIORITY,
            response_format=GPTreeQuestions,
        )
        assert isinstance(resp.response, GPTreeQuestions)
        assert len(resp.response.questions) == 2
        q = resp.response.questions[0]
        assert hasattr(q, "value")
        assert hasattr(q, "choices")
        assert q.question_type == "INFERENCE"

    @pytest.mark.asyncio
    async def test_gptree_dynamic_answer(self) -> None:
        """Dynamic Answer model from _make_answer_model (Literal choices)."""
        DynAnswer = create_model("Answer", answer=(Literal["Yes", "No"], ...))
        fake = FakeLLM()
        resp = await fake.respond(
            query="Query: Q?\n\nSample: some text",
            llm_priority=_LLM_PRIORITY,
            response_format=DynAnswer,
        )
        assert resp.response.answer in ("Yes", "No")  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_gptree_dynamic_answer_three_choices(self) -> None:
        """Dynamic Answer with 3 choices cycles through them."""
        DynAnswer = create_model(
            "Answer", answer=(Literal["High", "Medium", "Low"], ...)
        )
        fake = FakeLLM()
        answers = []
        for _ in range(6):
            resp = await fake.respond(
                query="Query: Q?\n\nSample: text",
                llm_priority=_LLM_PRIORITY,
                response_format=DynAnswer,
            )
            answers.append(resp.response.answer)  # type: ignore[union-attr]
        # Should cycle through all 3 choices
        assert set(answers) == {"High", "Medium", "Low"}

    @pytest.mark.asyncio
    async def test_gptree_set_tasks(self) -> None:
        fake = FakeLLM()
        resp = await fake.respond(
            query="Build a decision tree for:\nClassifying founders",
            llm_priority=_LLM_PRIORITY,
            response_format=str,
        )
        assert "<number_of_questions>" in str(resp.response)

    @pytest.mark.asyncio
    async def test_gptree_answer_cycles_choices(self) -> None:
        """Answers cycle through choices for non-degenerate splits."""
        DynAnswer = create_model("Answer", answer=(Literal["Yes", "No"], ...))
        fake = FakeLLM()
        answers = []
        for _ in range(4):
            resp = await fake.respond(
                query="Query: Q?\n\nSample: text",
                llm_priority=_LLM_PRIORITY,
                response_format=DynAnswer,
            )
            answers.append(resp.response.answer)  # type: ignore[union-attr]
        # With cycling, should see both values
        assert "Yes" in answers
        assert "No" in answers


# ---------------------------------------------------------------------------
# Call tracking
# ---------------------------------------------------------------------------


class TestCallTracking:
    """Verify call count and reset work across both RRF and GPTree calls."""

    @pytest.mark.asyncio
    async def test_call_count_increments(self) -> None:
        fake = FakeLLM()
        assert fake.call_count == 0
        await fake.respond("q", _LLM_PRIORITY, RRFQuestions)
        assert fake.call_count == 1
        await fake.respond("q", _LLM_PRIORITY, GPTreeQuestions)
        assert fake.call_count == 2

    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        fake = FakeLLM()
        await fake.respond("q", _LLM_PRIORITY, RRFQuestions)
        fake.reset()
        assert fake.call_count == 0
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_unknown_format_raises(self) -> None:
        fake = FakeLLM()
        with pytest.raises(TypeError, match="unknown response_format"):
            await fake.respond("q", _LLM_PRIORITY, int)  # type: ignore
