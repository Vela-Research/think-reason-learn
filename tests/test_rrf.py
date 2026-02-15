"""Offline tests for RRF using FakeLLM -- no real API calls."""

from __future__ import annotations

import pytest
import pandas as pd

from think_reason_learn.rrf import RRF, QuestionExclusion
from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.core.exceptions import DataError
from tests.fake_llm import FakeLLM


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

LLM_CHOICE: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]


def _sample_data() -> tuple[pd.DataFrame, list[str]]:
    """Minimal 8-row dataset (trimmed from examples/rrf.py)."""
    persons = [
        "A: 30yo woman, CS Stanford, 6yr Google, AI healthcare startup, $2M seed.",
        "B: 25yo man, marketing NYU, 3yr Apple marketing mgr, social media app.",
        "C: LA doctor, UCLA, remote monitoring platform, no tech/startup exp.",
        "D: 40yo man, law UChicago, corporate lawyer, legal-tech idea, no tech bg.",
        "E: 28yo woman, CE Berkeley, fintech YC startup, own fintech product.",
        "F: 32yo man, MBA Columbia, marketing Apple/Spotify, subscription box.",
        "G: 27yo woman, IE MIT, PM Amazon logistics, 2 cofounders, accelerator.",
        "H: 7 companies 3 industries, PM fintech startup, edutech idea.",
    ]
    X = pd.DataFrame({"data": persons})
    y = ["YES", "NO", "NO", "YES", "NO", "NO", "YES", "NO"]
    return X, y


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, list[str]]:
    return _sample_data()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def rrf_with_fake(fake_llm: FakeLLM) -> RRF:
    return RRF(
        qgen_llmc=LLM_CHOICE,
        name="test_rrf",
        max_samples_as_context=5,
        max_generated_questions=6,
        _llm=fake_llm,
    )


# ---------------------------------------------------------------------------
# set_tasks
# ---------------------------------------------------------------------------


class TestSetTasks:
    @pytest.mark.asyncio
    async def test_with_description(
        self, rrf_with_fake: RRF, fake_llm: FakeLLM
    ) -> None:
        template = await rrf_with_fake.set_tasks(
            task_description="Classify founders as successful or not"
        )
        assert "<number_of_questions>" in template
        assert rrf_with_fake.question_gen_instructions_template == template
        assert len(fake_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_with_custom_template(self, rrf_with_fake: RRF) -> None:
        custom = "Generate <number_of_questions> questions about founders."
        result = await rrf_with_fake.set_tasks(instructions_template=custom)
        assert result == custom

    @pytest.mark.asyncio
    async def test_missing_tag_raises(self, rrf_with_fake: RRF) -> None:
        with pytest.raises(ValueError, match="must contain the tag"):
            await rrf_with_fake.set_tasks(instructions_template="No tag here.")


# ---------------------------------------------------------------------------
# fit (end-to-end)
# ---------------------------------------------------------------------------


class TestFit:
    @pytest.mark.asyncio
    async def test_basic(
        self,
        rrf_with_fake: RRF,
        sample_data: tuple[pd.DataFrame, list[str]],
    ) -> None:
        X, y = sample_data
        await rrf_with_fake.set_tasks(task_description="Classify founders")
        rrf = await rrf_with_fake.fit(X, y)
        qdf = rrf.get_questions()
        assert len(qdf) > 0
        adf = rrf.get_answers()
        assert adf.shape[0] == len(X)

    @pytest.mark.asyncio
    async def test_sets_metrics(
        self,
        rrf_with_fake: RRF,
        sample_data: tuple[pd.DataFrame, list[str]],
    ) -> None:
        X, y = sample_data
        await rrf_with_fake.set_tasks(task_description="Classify founders")
        rrf = await rrf_with_fake.fit(X, y)
        qdf = rrf.get_questions()
        for col in ("precision", "recall", "f1_score", "accuracy"):
            assert qdf[col].dropna().shape[0] > 0

    @pytest.mark.asyncio
    async def test_without_set_tasks_raises(
        self,
        rrf_with_fake: RRF,
        sample_data: tuple[pd.DataFrame, list[str]],
    ) -> None:
        X, y = sample_data
        with pytest.raises(ValueError, match="template is not set"):
            await rrf_with_fake.fit(X, y)

    @pytest.mark.asyncio
    async def test_reset(
        self,
        rrf_with_fake: RRF,
        sample_data: tuple[pd.DataFrame, list[str]],
    ) -> None:
        X, y = sample_data
        await rrf_with_fake.set_tasks(task_description="Classify founders")
        await rrf_with_fake.fit(X, y)
        rrf = await rrf_with_fake.fit(X, y, reset=True)
        assert rrf.get_questions().shape[0] > 0


# ---------------------------------------------------------------------------
# fit with batch answering
# ---------------------------------------------------------------------------


class TestFitBatch:
    @pytest.mark.asyncio
    async def test_batch_mode(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_batch",
            max_samples_as_context=5,
            max_generated_questions=6,
            qanswer_batch_size=4,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        adf = rrf.get_answers()
        assert adf.shape[0] == len(X)
        # Verify batch calls happened (response_format=str, not set_tasks)
        batch_calls = [
            c
            for c in fake.calls
            if c["response_format"] is str
            and "generate yes/no" not in c["query"].lower()
        ]
        assert len(batch_calls) > 0


# ---------------------------------------------------------------------------
# question filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    @pytest.mark.asyncio
    async def test_pred_similarity(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM(default_answer="YES")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_filter",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        rrf.filter_questions_on_pred_similarity(threshold=0.9)
        qdf = rrf.get_questions()
        excluded = qdf[qdf["exclusion"].notna()]
        # All-YES answers → all questions identical → all but one excluded
        assert len(excluded) >= len(qdf) - 1

    @pytest.mark.asyncio
    async def test_semantics_hashed(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_sem",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        await rrf.filter_questions_on_semantics(
            threshold=0.5, emb_model="hashed_bag_of_words"
        )

    @pytest.mark.asyncio
    async def test_clear_semantic_exclusions(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_clear",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        await rrf.filter_questions_on_semantics(
            threshold=0.01, emb_model="hashed_bag_of_words"
        )
        await rrf.filter_questions_on_semantics(
            threshold=None, emb_model="hashed_bag_of_words"
        )
        qdf = rrf.get_questions()
        sem_excluded = qdf[qdf["exclusion"] == QuestionExclusion.SEMANTICS.value]
        assert len(sem_excluded) == 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    @pytest.mark.asyncio
    async def test_yields_results(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pred",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        preds = []
        async for pred in rrf.predict(X):
            preds.append(pred)
        assert len(preds) > 0
        _sample_idx, _qid, answer, _tc = preds[0]
        assert answer in ("YES", "NO")


# ---------------------------------------------------------------------------
# data validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_max_questions(self) -> None:
        with pytest.raises(ValueError):
            RRF(qgen_llmc=LLM_CHOICE, max_generated_questions=0)

    def test_invalid_class_ratio(self) -> None:
        with pytest.raises(ValueError):
            RRF(qgen_llmc=LLM_CHOICE, class_ratio=(-1, 1))

    def test_invalid_similarity_func(self) -> None:
        with pytest.raises(ValueError):
            RRF(qgen_llmc=LLM_CHOICE, answer_similarity_func="cosine")

    @pytest.mark.asyncio
    async def test_mismatched_xy_raises(self) -> None:
        fake = FakeLLM()
        rrf = RRF(qgen_llmc=LLM_CHOICE, name="test_val", _llm=fake)
        X = pd.DataFrame({"data": ["a", "b"]})
        y = ["YES"]
        await rrf.set_tasks(task_description="Test")
        with pytest.raises(DataError):
            await rrf.fit(X, y)

    @pytest.mark.asyncio
    async def test_invalid_labels_raises(self) -> None:
        fake = FakeLLM()
        rrf = RRF(qgen_llmc=LLM_CHOICE, name="test_val2", _llm=fake)
        X = pd.DataFrame({"data": ["a", "b"]})
        y = ["YES", "MAYBE"]
        await rrf.set_tasks(task_description="Test")
        with pytest.raises(DataError):
            await rrf.fit(X, y)


# ---------------------------------------------------------------------------
# question management
# ---------------------------------------------------------------------------


class TestQuestionManagement:
    @pytest.mark.asyncio
    async def test_add_question(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_add",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        before = len(rrf.get_questions())
        await rrf.add_question("Is the founder based in Silicon Valley?")
        after = len(rrf.get_questions())
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_update_question_exclusion(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        qdf = rrf.get_questions()
        qid = str(qdf.index[0])
        await rrf.update_question_exclusion(qid, QuestionExclusion.EXPERT)
        assert (
            rrf.get_questions().at[qid, "exclusion"] == QuestionExclusion.EXPERT.value
        )
        await rrf.update_question_exclusion(qid, None)
        assert rrf.get_questions().at[qid, "exclusion"] is None


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    @pytest.mark.asyncio
    async def test_round_trip(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_sl",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        save_dir = str(tmp_path)
        rrf.save(save_dir)

        loaded = RRF.load(save_dir)
        assert loaded.get_questions().shape == rrf.get_questions().shape
        assert loaded.get_answers().shape == rrf.get_answers().shape
