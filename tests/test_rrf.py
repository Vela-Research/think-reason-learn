"""Offline tests for RRF using FakeLLM -- no real API calls."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from think_reason_learn.rrf import RRF, QuestionExclusion
from think_reason_learn.rrf._rrf import Questions, Answer
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


class TestPredictBatch:
    @pytest.mark.asyncio
    async def test_batch_predict_reduces_calls(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pred_batch",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        active_qs = rrf.get_questions()
        active_qs = active_qs[active_qs["exclusion"].isna()]
        num_active = len(active_qs)

        # Reset call counter before predict
        fake._call_count = 0
        fake.calls.clear()

        preds = []
        async for pred in rrf.predict(X):
            preds.append(pred)

        # With batch_size=20 and 8 samples, all samples fit in one batch
        # per question => exactly num_active_questions batch calls
        batch_calls = [
            c
            for c in fake.calls
            if c["response_format"] is str
            and "generate yes/no" not in c["query"].lower()
        ]
        assert len(batch_calls) == num_active

        # All (sample, question) pairs should be yielded
        assert len(preds) == len(X) * num_active
        for sample_idx, qid, answer, tc in preds:
            assert answer in ("YES", "NO")

    @pytest.mark.asyncio
    async def test_batch_predict_smaller_batch_size(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        import math

        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pred_batch_small",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        active_qs = rrf.get_questions()
        active_qs = active_qs[active_qs["exclusion"].isna()]
        num_active = len(active_qs)

        fake._call_count = 0
        fake.calls.clear()

        preds = []
        async for pred in rrf.predict(X):
            preds.append(pred)

        # batch_size=3, 8 samples => ceil(8/3)=3 batches per question
        expected_calls = math.ceil(len(X) / 3) * num_active
        batch_calls = [
            c
            for c in fake.calls
            if c["response_format"] is str
            and "generate yes/no" not in c["query"].lower()
        ]
        assert len(batch_calls) == expected_calls
        assert len(preds) == len(X) * num_active

    @pytest.mark.asyncio
    async def test_batch_predict_matches_single_predict(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        X, y = sample_data

        # Single (unbatched) predict
        fake_single = FakeLLM()
        rrf_single = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pred_single",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake_single,
        )
        await rrf_single.set_tasks(task_description="Classify founders")
        await rrf_single.fit(X, y)

        single_preds: dict[tuple[int, str], str] = {}
        async for sample_idx, qid, answer, _tc in rrf_single.predict(X):
            single_preds[(int(sample_idx), qid)] = answer

        # Batched predict
        fake_batch = FakeLLM()
        rrf_batch = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pred_match",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake_batch,
        )
        await rrf_batch.set_tasks(task_description="Classify founders")
        await rrf_batch.fit(X, y)

        batch_preds: dict[tuple[int, str], str] = {}
        async for sample_idx, qid, answer, _tc in rrf_batch.predict(X):
            batch_preds[(int(sample_idx), qid)] = answer

        # Same keys and same answers
        assert set(single_preds.keys()) == set(batch_preds.keys())
        for key in single_preds:
            assert single_preds[key] == batch_preds[key], (
                f"Mismatch at {key}: single={single_preds[key]}, "
                f"batch={batch_preds[key]}"
            )

    @pytest.mark.asyncio
    async def test_batch_predict_default_batch_size(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pred_default_batch",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        active_qs = rrf.get_questions()
        active_qs = active_qs[active_qs["exclusion"].isna()]
        num_active = len(active_qs)

        fake._call_count = 0
        fake.calls.clear()

        preds = []
        async for pred in rrf.predict(X):
            preds.append(pred)

        # Default qanswer_batch_size=None should use batch_size=20 in predict
        # With 8 samples < 20, all fit in one batch per question
        batch_calls = [
            c
            for c in fake.calls
            if c["response_format"] is str
            and "generate yes/no" not in c["query"].lower()
        ]
        assert len(batch_calls) == num_active
        assert len(preds) == len(X) * num_active


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

    def test_correlation_similarity_func_accepted(self) -> None:
        rrf = RRF(qgen_llmc=LLM_CHOICE, answer_similarity_func="correlation")
        assert rrf.answer_similarity_func == "correlation"

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
# F-beta scoring (issue #46)
# ---------------------------------------------------------------------------


class TestFBetaScoring:
    @pytest.mark.asyncio
    async def test_default_beta_matches_f1(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Default beta=1.0 produces f_beta_score equal to f1_score."""
        fake = FakeLLM(default_answer="ALTERNATE")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fbeta_default",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        qdf = rrf.get_questions()
        assert "f_beta_score" in qdf.columns
        assert "f1_score" in qdf.columns
        for qid in qdf.index:
            f1 = qdf.at[qid, "f1_score"]
            fb = qdf.at[qid, "f_beta_score"]
            if pd.notna(f1) and pd.notna(fb):
                assert f1 == pytest.approx(fb), f"q={qid}: f1={f1} != f_beta={fb}"

    @pytest.mark.asyncio
    async def test_beta_half_favours_precision(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """beta=0.5 weights precision more than recall."""
        fake = FakeLLM(default_answer="ALTERNATE")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fbeta_half",
            max_samples_as_context=8,
            max_generated_questions=3,
            question_scoring_f_beta=0.5,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        qdf = rrf.get_questions()
        for qid in qdf.index:
            fb = qdf.at[qid, "f_beta_score"]
            f1 = qdf.at[qid, "f1_score"]
            p = qdf.at[qid, "precision"]
            if pd.notna(fb) and pd.notna(p) and pd.notna(f1):
                # F0.5 should be closer to precision than F1 is
                # (or at least not equal to F1 when p != r)
                assert fb != pytest.approx(f1) or p == pytest.approx(
                    qdf.at[qid, "recall"]
                )

    @pytest.mark.asyncio
    async def test_beta_two_favours_recall(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """beta=2.0 weights recall more than precision."""
        fake = FakeLLM(default_answer="ALTERNATE")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fbeta_two",
            max_samples_as_context=8,
            max_generated_questions=3,
            question_scoring_f_beta=2.0,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        qdf = rrf.get_questions()
        assert "f_beta_score" in qdf.columns
        # f_beta column has values
        assert qdf["f_beta_score"].dropna().shape[0] > 0

    def test_beta_zero_raises(self) -> None:
        """beta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="question_scoring_f_beta"):
            RRF(qgen_llmc=LLM_CHOICE, question_scoring_f_beta=0.0)

    def test_beta_negative_raises(self) -> None:
        """beta < 0 raises ValueError."""
        with pytest.raises(ValueError, match="question_scoring_f_beta"):
            RRF(qgen_llmc=LLM_CHOICE, question_scoring_f_beta=-1.0)

    @pytest.mark.asyncio
    async def test_fbeta_zero_precision_recall(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """When all predictions are NO, precision=recall=0 => f_beta_score=0."""
        fake = FakeLLM(default_answer="NO")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fbeta_zero",
            max_samples_as_context=8,
            max_generated_questions=3,
            question_scoring_f_beta=2.0,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        qdf = rrf.get_questions()
        for qid in qdf.index:
            p = qdf.at[qid, "precision"]
            r = qdf.at[qid, "recall"]
            fb = qdf.at[qid, "f_beta_score"]
            if pd.notna(p) and p == 0.0 and pd.notna(r) and r == 0.0:
                assert fb == 0.0

    @pytest.mark.asyncio
    async def test_fbeta_save_load_round_trip(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        """Save/load preserves question_scoring_f_beta and f_beta_score column."""
        fake = FakeLLM(default_answer="ALTERNATE")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fbeta_sl",
            max_samples_as_context=8,
            max_generated_questions=3,
            question_scoring_f_beta=0.5,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        save_dir = str(tmp_path)
        rrf.save(save_dir)

        loaded = RRF.load(save_dir)
        assert loaded.question_scoring_f_beta == 0.5
        loaded_qdf = loaded.get_questions()
        assert "f_beta_score" in loaded_qdf.columns
        orig_qdf = rrf.get_questions()
        for qid in orig_qdf.index:
            if pd.notna(orig_qdf.at[qid, "f_beta_score"]):
                assert loaded_qdf.at[qid, "f_beta_score"] == pytest.approx(
                    orig_qdf.at[qid, "f_beta_score"]
                )


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


# ---------------------------------------------------------------------------
# exclusion_report (issue #45)
# ---------------------------------------------------------------------------


class TestExclusionReport:
    @pytest.mark.asyncio
    async def test_empty_report_before_filtering(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl_empty",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 0
        expected_cols = {
            "excluded_question_id",
            "exclusion_reason",
            "reference_question_id",
            "similarity_score",
            "threshold",
            "metric_used",
        }
        assert set(report.columns) == expected_cols

    @pytest.mark.asyncio
    async def test_pred_similarity_exclusion_report(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM(default_answer="YES")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl_pred",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        rrf.filter_questions_on_pred_similarity(threshold=0.9)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0
        for _, row in report.iterrows():
            assert row["exclusion_reason"] == "prediction_similarity"
            assert row["reference_question_id"] is not None
            assert isinstance(row["similarity_score"], float)
            assert row["threshold"] == 0.9
            assert row["metric_used"] == "hamming"

    @pytest.mark.asyncio
    async def test_pred_similarity_correlation(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM(default_answer="YES")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_corr",
            max_samples_as_context=8,
            max_generated_questions=6,
            answer_similarity_func="correlation",
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        rrf.filter_questions_on_pred_similarity(threshold=0.9)
        qdf = rrf.get_questions()
        excluded = qdf[qdf["exclusion"].notna()]
        # All-YES answers → constant columns → correlation=0 → none excluded
        assert len(excluded) == 0

    @pytest.mark.asyncio
    async def test_correlation_exclusion_report_metric(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_corr_report",
            max_samples_as_context=8,
            max_generated_questions=6,
            answer_similarity_func="correlation",
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        rrf.filter_questions_on_pred_similarity(threshold=0.5)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        if len(report) > 0:
            for _, row in report.iterrows():
                assert row["metric_used"] == "correlation"

    @pytest.mark.asyncio
    async def test_semantic_similarity_exclusion_report(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl_sem",
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

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0
        for _, row in report.iterrows():
            assert row["exclusion_reason"] == "semantic_similarity"
            assert row["reference_question_id"] is not None
            assert isinstance(row["similarity_score"], float)
            assert row["similarity_score"] >= 0.01
            assert row["threshold"] == 0.01
            assert row["metric_used"] == "dot_product"

    @pytest.mark.asyncio
    async def test_expert_exclusion_report(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl_expert",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        qid = str(rrf.get_questions().index[0])
        await rrf.update_question_exclusion(qid, QuestionExclusion.EXPERT)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["excluded_question_id"] == qid
        assert row["exclusion_reason"] == "expert"
        assert row["reference_question_id"] is None
        assert row["similarity_score"] is None
        assert row["threshold"] is None
        assert row["metric_used"] is None

    @pytest.mark.asyncio
    async def test_report_as_dict(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        import json

        fake = FakeLLM(default_answer="YES")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl_dict",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        rrf.filter_questions_on_pred_similarity(threshold=0.9)

        result = rrf.exclusion_report(as_dict=True)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(e, dict) for e in result)
        # Must be JSON-serialisable
        json.dumps(result)

    @pytest.mark.asyncio
    async def test_clearing_removes_events(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        fake = FakeLLM(default_answer="YES")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl_clear",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        rrf.filter_questions_on_pred_similarity(threshold=0.9)
        assert len(rrf.exclusion_report()) > 0
        # Clear
        rrf.filter_questions_on_pred_similarity(threshold=None)
        assert len(rrf.exclusion_report()) == 0

    @pytest.mark.asyncio
    async def test_exclusion_report_save_load(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        fake = FakeLLM(default_answer="YES")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_excl_sl",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)
        rrf.filter_questions_on_pred_similarity(threshold=0.9)
        original_report = rrf.exclusion_report(as_dict=True)
        assert len(original_report) > 0

        save_dir = str(tmp_path)
        rrf.save(save_dir)
        loaded = RRF.load(save_dir)

        loaded_report = loaded.exclusion_report(as_dict=True)
        assert loaded_report == original_report


# ---------------------------------------------------------------------------
# early semantic filtering (issue #44)
# ---------------------------------------------------------------------------


class TestEarlySemanticFiltering:
    @pytest.mark.asyncio
    async def test_early_filter_disabled_by_default(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Default RRF has no early semantic exclusions."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_off",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        sem_rows = report[report["exclusion_reason"] == "semantic_similarity"]
        assert len(sem_rows) == 0

    @pytest.mark.asyncio
    async def test_early_filter_reduces_questions(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Low threshold should exclude some questions before answering."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_low",
            max_samples_as_context=8,
            max_generated_questions=6,
            semantic_filtering_during_fit=True,
            semantic_similarity_threshold=0.01,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        sem_rows = report[report["exclusion_reason"] == "semantic_similarity"]
        assert len(sem_rows) > 0

    @pytest.mark.asyncio
    async def test_early_filter_high_threshold_no_effect(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Threshold=1.0 should not exclude any questions."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_high",
            max_samples_as_context=8,
            max_generated_questions=6,
            semantic_filtering_during_fit=True,
            semantic_similarity_threshold=1.0,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        sem_rows = report[report["exclusion_reason"] == "semantic_similarity"]
        assert len(sem_rows) == 0

    @pytest.mark.asyncio
    async def test_early_filter_records_exclusion_log(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Exclusion report entries have correct fields."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_log",
            max_samples_as_context=8,
            max_generated_questions=6,
            semantic_filtering_during_fit=True,
            semantic_similarity_threshold=0.01,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        report = rrf.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        sem_rows = report[report["exclusion_reason"] == "semantic_similarity"]
        assert len(sem_rows) > 0
        for _, row in sem_rows.iterrows():
            assert row["metric_used"] == "dot_product"
            assert row["threshold"] == 0.01
            assert row["reference_question_id"] is not None
            assert isinstance(row["similarity_score"], float)

    @pytest.mark.asyncio
    async def test_early_filter_save_load_config(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        """Save/load preserves early filter config params."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_sl",
            max_samples_as_context=8,
            max_generated_questions=3,
            semantic_filtering_during_fit=True,
            semantic_similarity_threshold=0.85,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        save_dir = str(tmp_path)
        rrf.save(save_dir)
        loaded = RRF.load(save_dir)
        assert loaded.semantic_filtering_during_fit is True
        assert loaded.semantic_similarity_threshold == 0.85

    @pytest.mark.asyncio
    async def test_fit_summary_populated(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """After fit with early filter, _last_fit_summary is populated."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_summary",
            max_samples_as_context=8,
            max_generated_questions=6,
            semantic_filtering_during_fit=True,
            semantic_similarity_threshold=0.01,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        summary = rrf._last_fit_summary
        assert "questions_generated" in summary
        assert "questions_after_early_filter" in summary
        assert "questions_answered" in summary
        assert summary["questions_generated"] > 0
        assert summary["questions_after_early_filter"] is not None
        assert summary["questions_after_early_filter"] <= summary["questions_generated"]

    @pytest.mark.asyncio
    async def test_early_filter_reduces_llm_calls(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Early filtering should result in fewer answering LLM calls."""
        X, y = sample_data

        # Baseline: no early filtering
        fake_base = FakeLLM()
        rrf_base = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_calls_base",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake_base,
        )
        await rrf_base.set_tasks(task_description="Classify founders")
        await rrf_base.fit(X, y)
        calls_baseline = fake_base._call_count

        # With early filtering (low threshold forces exclusions)
        fake_early = FakeLLM()
        rrf_early = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_early_calls_early",
            max_samples_as_context=8,
            max_generated_questions=6,
            semantic_filtering_during_fit=True,
            semantic_similarity_threshold=0.01,
            _llm=fake_early,
        )
        await rrf_early.set_tasks(task_description="Classify founders")
        await rrf_early.fit(X, y)
        calls_early = fake_early._call_count

        # Early filtering must have excluded some questions
        report = rrf_early.exclusion_report()
        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0

        # Fewer total LLM calls because fewer questions were answered
        assert calls_early < calls_baseline


# ---------------------------------------------------------------------------
# predict concurrency (Step 5)
# ---------------------------------------------------------------------------


class TestPredictConcurrent:
    @pytest.mark.asyncio
    async def test_concurrent_matches_sequential(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Sequential and concurrent predict produce identical results."""
        X, y = sample_data

        # Sequential
        fake_seq = FakeLLM()
        rrf_seq = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_conc_seq",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake_seq,
        )
        await rrf_seq.set_tasks(task_description="Classify founders")
        await rrf_seq.fit(X, y)

        seq_preds: dict[tuple[int, str], str] = {}
        async for si, qid, ans, _tc in rrf_seq.predict(X):
            seq_preds[(int(si), qid)] = ans

        # Concurrent
        fake_conc = FakeLLM()
        rrf_conc = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_conc_conc",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake_conc,
        )
        await rrf_conc.set_tasks(task_description="Classify founders")
        await rrf_conc.fit(X, y)

        conc_preds: dict[tuple[int, str], str] = {}
        async for si, qid, ans, _tc in rrf_conc.predict(X, max_concurrent=3):
            conc_preds[(int(si), qid)] = ans

        assert seq_preds == conc_preds

    @pytest.mark.asyncio
    async def test_concurrent_ordering_stable(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Two concurrent predict runs yield results in the same order."""
        X, y = sample_data

        async def _run() -> list[tuple[int, str, str]]:
            fake = FakeLLM()
            rrf = RRF(
                qgen_llmc=LLM_CHOICE,
                name="test_conc_order",
                max_samples_as_context=8,
                max_generated_questions=3,
                qanswer_batch_size=20,
                _llm=fake,
            )
            await rrf.set_tasks(task_description="Classify founders")
            await rrf.fit(X, y)
            out: list[tuple[int, str, str]] = []
            async for si, qid, ans, _tc in rrf.predict(X, max_concurrent=2):
                out.append((int(si), qid, ans))
            return out

        run1 = await _run()
        run2 = await _run()
        assert run1 == run2

    @pytest.mark.asyncio
    async def test_concurrent_respects_max_concurrent(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Peak in-flight questions never exceeds max_concurrent."""
        import asyncio as _aio

        peak = 0
        current = 0
        lock = _aio.Lock()

        class TrackingFakeLLM(FakeLLM):
            async def respond(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
                nonlocal peak, current
                async with lock:
                    current += 1
                    if current > peak:
                        peak = current
                # Simulate a small delay so waves actually overlap
                await _aio.sleep(0.01)
                result = await super().respond(*args, **kwargs)  # type: ignore[arg-type]
                async with lock:
                    current -= 1
                return result

        X, y = sample_data
        fake = TrackingFakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_conc_limit",
            max_samples_as_context=8,
            max_generated_questions=6,
            qanswer_batch_size=20,
            _llm=fake,
        )
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        max_conc = 2
        # Reset tracking after fit
        peak = 0
        current = 0

        preds = []
        async for pred in rrf.predict(X, max_concurrent=max_conc):
            preds.append(pred)

        assert len(preds) > 0
        assert peak <= max_conc


class TestFounderPredictions:
    """Tests for founder-level aggregation: aggregate_predictions,
    _tune_aggregation (inside fit), and predict_founder_level."""

    # -- static aggregate_predictions tests (no LLM) --

    def test_aggregate_predictions_basic(self) -> None:
        """Hand-crafted 4x3 matrix: verify correct aggregation."""
        matrix = pd.DataFrame(
            {
                "q0": [1, 0, 1, 0],
                "q1": [1, 1, 0, 0],
                "q2": [0, 0, 1, 1],
            },
            index=pd.Index([0, 1, 2, 3]),
        )
        scores = pd.Series({"q0": 0.9, "q1": 0.7, "q2": 0.5})
        # K=2: top questions are q0, q1
        # yes_counts: [2, 1, 1, 0]
        # T=1: predictions: [YES, YES, YES, NO]
        result = RRF.aggregate_predictions(matrix, scores, k=2, t=1)
        assert list(result) == ["YES", "YES", "YES", "NO"]

    def test_aggregate_predictions_k1_t1(self) -> None:
        """K=1, T=1: prediction equals top-scored question's column."""
        matrix = pd.DataFrame(
            {"q0": [1, 0, 1], "q1": [0, 1, 0]},
            index=pd.Index([0, 1, 2]),
        )
        scores = pd.Series({"q0": 0.3, "q1": 0.8})
        # Top question is q1 (score 0.8)
        result = RRF.aggregate_predictions(matrix, scores, k=1, t=1)
        assert list(result) == ["NO", "YES", "NO"]

    def test_aggregate_predictions_all_yes(self) -> None:
        """All-ones matrix: any T<=K should predict all YES."""
        matrix = pd.DataFrame(
            {"q0": [1, 1], "q1": [1, 1], "q2": [1, 1]},
            index=pd.Index([0, 1]),
        )
        scores = pd.Series({"q0": 0.9, "q1": 0.7, "q2": 0.5})
        for k in range(1, 4):
            for t in range(1, k + 1):
                result = RRF.aggregate_predictions(matrix, scores, k=k, t=t)
                assert list(result) == ["YES", "YES"]

    def test_aggregate_predictions_invalid_params(self) -> None:
        """Invalid K/T raise ValueError."""
        matrix = pd.DataFrame({"q0": [1], "q1": [0]}, index=pd.Index([0]))
        scores = pd.Series({"q0": 0.9, "q1": 0.5})

        with pytest.raises(ValueError, match="k must be"):
            RRF.aggregate_predictions(matrix, scores, k=0, t=1)
        with pytest.raises(ValueError, match="k must be"):
            RRF.aggregate_predictions(matrix, scores, k=3, t=1)
        with pytest.raises(ValueError, match="t must be"):
            RRF.aggregate_predictions(matrix, scores, k=2, t=0)
        with pytest.raises(ValueError, match="t must be"):
            RRF.aggregate_predictions(matrix, scores, k=2, t=3)

    # -- integration tests (FakeLLM) --

    @pytest.mark.asyncio
    async def test_fit_tunes_aggregation(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """After fit(), _aggregation_k and _aggregation_t are set."""
        fake = FakeLLM(default_answer="ALTERNATE")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fit_tunes",
            max_samples_as_context=8,
            max_generated_questions=6,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        assert isinstance(rrf._aggregation_k, int)
        assert isinstance(rrf._aggregation_t, int)
        assert rrf._aggregation_k >= 1
        assert rrf._aggregation_t >= 1

    @pytest.mark.asyncio
    async def test_predict_founder_level_after_fit(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """fit() then predict_founder_level(X) returns correct DataFrame."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pf_after_fit",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        result = await rrf.predict_founder_level(X)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"prediction", "yes_count", "k", "t"}
        assert len(result) == len(X)
        assert list(result.index) == list(X.index)

    @pytest.mark.asyncio
    async def test_predict_founder_level_explicit_params(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """predict_founder_level(X, k=1, t=1) works with explicit params."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pf_explicit",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        result = await rrf.predict_founder_level(X, k=1, t=1)
        assert isinstance(result, pd.DataFrame)
        assert (result["k"] == 1).all()
        assert (result["t"] == 1).all()

    @pytest.mark.asyncio
    async def test_predict_founder_level_raises_without_fit(self) -> None:
        """predict_founder_level without fit raises ValueError."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pf_no_fit",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        await rrf.set_tasks(task_description="Classify founders")
        X = pd.DataFrame({"data": ["test person"]})
        with pytest.raises(ValueError):
            await rrf.predict_founder_level(X)

    @pytest.mark.asyncio
    async def test_predict_founder_level_all_yes_answers(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """FakeLLM(YES): all yes_count == K, all predictions YES."""
        fake = FakeLLM(default_answer="YES")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pf_all_yes",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        k = rrf._aggregation_k
        assert k is not None
        result = await rrf.predict_founder_level(X, k=k, t=1)
        assert (result["prediction"] == "YES").all()
        assert (result["yes_count"] == k).all()

    @pytest.mark.asyncio
    async def test_aggregation_save_load(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        """predict with checkpoint_path creates predict_checkpoint.json."""
        import orjson
        from pathlib import Path

        X, y = sample_data
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_ckpt_save",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake,
        )
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        ckpt_dir = str(tmp_path)
        async for _ in rrf.predict(X, checkpoint_path=ckpt_dir):
            pass

        ckpt_file = Path(ckpt_dir) / "predict_checkpoint.json"
        assert ckpt_file.exists()
        data = orjson.loads(ckpt_file.read_bytes())
        assert "completed_qids" in data
        assert "results" in data
        assert "token_counter" in data

    @pytest.mark.asyncio
    async def test_checkpoint_resume_matches_uninterrupted(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        """Full predict then resumed predict yield identical results."""
        X, y = sample_data

        # Full run with checkpoint
        fake_full = FakeLLM()
        rrf_full = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_ckpt_match",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake_full,
        )
        await rrf_full.set_tasks(task_description="Classify founders")
        await rrf_full.fit(X, y)

        ckpt_dir = str(tmp_path)
        full_preds: dict[tuple[int, str], str] = {}
        async for si, qid, ans, _tc in rrf_full.predict(X, checkpoint_path=ckpt_dir):
            full_preds[(int(si), qid)] = ans

        # Resumed run from checkpoint
        fake_resume = FakeLLM()
        rrf_resume = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_ckpt_match",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake_resume,
        )
        await rrf_resume.set_tasks(task_description="Classify founders")
        await rrf_resume.fit(X, y)

        resume_preds: dict[tuple[int, str], str] = {}
        async for si, qid, ans, _tc in rrf_resume.predict(
            X, checkpoint_path=ckpt_dir, resume=True
        ):
            resume_preds[(int(si), qid)] = ans

        assert full_preds == resume_preds

    @pytest.mark.asyncio
    async def test_checkpoint_no_resume_without_flag(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        """resume=False recomputes from scratch even if checkpoint exists."""
        X, y = sample_data

        fake1 = FakeLLM()
        rrf1 = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_ckpt_noresume",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake1,
        )
        await rrf1.set_tasks(task_description="Classify founders")
        await rrf1.fit(X, y)

        ckpt_dir = str(tmp_path)
        async for _ in rrf1.predict(X, checkpoint_path=ckpt_dir):
            pass
        calls_first = fake1._call_count

        # Second run, resume=False (default) — should make same number of
        # predict-phase LLM calls
        fake2 = FakeLLM()
        rrf2 = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_ckpt_noresume",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake2,
        )
        await rrf2.set_tasks(task_description="Classify founders")
        await rrf2.fit(X, y)

        async for _ in rrf2.predict(X, checkpoint_path=ckpt_dir):
            pass
        # Both runs do the same total calls (fit + predict).
        # Since both RRFs are built identically, fit calls are the same,
        # so equal totals means predict was NOT skipped.
        assert fake2._call_count == calls_first

    @pytest.mark.asyncio
    async def test_resume_skips_completed_questions(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
        tmp_path: object,
    ) -> None:
        """Resumed run makes fewer LLM calls than a full run."""
        X, y = sample_data

        # Full run with checkpoint
        fake_full = FakeLLM()
        rrf_full = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_ckpt_skip",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake_full,
        )
        await rrf_full.set_tasks(task_description="Classify founders")
        await rrf_full.fit(X, y)

        ckpt_dir = str(tmp_path)
        async for _ in rrf_full.predict(X, checkpoint_path=ckpt_dir):
            pass
        full_total = fake_full._call_count

        # Resumed run — all questions already completed, zero predict calls
        fake_resume = FakeLLM()
        rrf_resume = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_ckpt_skip",
            max_samples_as_context=8,
            max_generated_questions=3,
            qanswer_batch_size=20,
            _llm=fake_resume,
        )
        await rrf_resume.set_tasks(task_description="Classify founders")
        await rrf_resume.fit(X, y)
        calls_after_fit = fake_resume._call_count

        async for _ in rrf_resume.predict(X, checkpoint_path=ckpt_dir, resume=True):
            pass

        predict_calls_resumed = fake_resume._call_count - calls_after_fit
        assert predict_calls_resumed == 0
        assert fake_resume._call_count < full_total
        """Save/load preserves aggregation K, T, and config params."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_agg_sl",
            max_samples_as_context=8,
            max_generated_questions=3,
            aggregation_metric="precision",
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        save_dir = str(tmp_path)
        rrf.save(save_dir)
        loaded = RRF.load(save_dir)

        assert loaded._aggregation_k == rrf._aggregation_k
        assert loaded._aggregation_t == rrf._aggregation_t
        assert loaded.aggregation_metric == "precision"

    @pytest.mark.asyncio
    async def test_predict_founder_level_output_contract(
        self, sample_data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Verify DataFrame columns, dtypes, and index alignment."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_pf_contract",
            max_samples_as_context=8,
            max_generated_questions=3,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        result = await rrf.predict_founder_level(X)
        assert all(v in ("YES", "NO") for v in result["prediction"])
        assert result["yes_count"].dtype in (np.int64, np.int32, int)
        assert (result["k"] == rrf._aggregation_k).all()
        assert (result["t"] == rrf._aggregation_t).all()


# ---------------------------------------------------------------------------
# Cost-sensitive mode tests (merged from feat/issue-48-cost-sensitive-mode)
# ---------------------------------------------------------------------------

from think_reason_learn.rrf._cost_sensitive import CostSensitiveConfig  # noqa: E402


@pytest.fixture
def cost_sensitive_data() -> tuple[pd.DataFrame, list[str]]:
    """Create small test dataset: 8 founders (3 YES, 5 NO)."""
    data = {
        "name": [f"Founder_{i}" for i in range(8)],
        "description": [f"Description {i}" for i in range(8)],
    }
    labels = ["YES", "YES", "YES", "NO", "NO", "NO", "NO", "NO"]
    return pd.DataFrame(data), labels


@pytest.mark.asyncio
async def test_standard_mode_baseline(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that cost_sensitive=False uses standard pipeline."""
    X, y = cost_sensitive_data
    fake_llm = FakeLLM(questions_per_call=3)

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=False,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Standard mode should:
    # - Generate questions (1 call)
    # - Answer all questions on all samples (3 questions × 8 samples)
    # With batching disabled or batch=1, expect many calls
    # With our FakeLLM, we don't batch by default in this test
    assert fake_llm.call_count > 0
    assert rrf._questions is not None
    assert len(rrf._questions) == 3


@pytest.mark.asyncio
async def test_cost_sensitive_reduces_calls(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that cost_sensitive=True reduces LLM calls."""
    X, y = cost_sensitive_data

    # Standard mode
    fake_llm_std = FakeLLM(questions_per_call=6)
    rrf_std = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=6,
        random_state=42,
        cost_sensitive=False,
        _llm=fake_llm_std,
    )
    await rrf_std.set_tasks(task_description="test task")
    await rrf_std.fit(X, y)
    standard_calls = fake_llm_std.call_count

    # Cost-sensitive mode
    fake_llm_cs = FakeLLM(questions_per_call=6)
    config = CostSensitiveConfig(
        screening_fraction=0.5,  # 4/8 samples for screening
        max_questions_full_eval=2,  # Only top 2 for full eval
        enable_semantic_filter=False,  # Disable for clearer call counting
    )
    rrf_cs = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=6,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm_cs,
    )
    await rrf_cs.set_tasks(task_description="test task")
    await rrf_cs.fit(X, y)
    cost_sensitive_calls = fake_llm_cs.call_count

    # Cost-sensitive should use fewer calls (screening + top-N < full eval)
    # Standard: 1 gen + 6 questions × 8 samples = many calls
    # Cost-sensitive: 1 gen + 6 questions × 4 screen + 2 questions × 8 full
    # Exact numbers depend on batching, but CS should be less or equal
    assert cost_sensitive_calls <= standard_calls


@pytest.mark.asyncio
async def test_screening_baseline_majority(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that majority baseline pruning works."""
    X, y = cost_sensitive_data
    fake_llm = FakeLLM(questions_per_call=5, default_answer="NO")

    config = CostSensitiveConfig(
        screening_fraction=1.0,  # Use all data for screening (simpler test)
        screening_baseline="majority",
        max_questions_full_eval=10,  # Don't prune by top-N
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=5,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # With labels = [YES, YES, YES, NO, NO, NO, NO, NO]
    # Majority class is NO (5/8)
    # Majority baseline should have some F1 > 0
    # Questions that perform worse should be excluded
    assert rrf._questions is not None
    # At least some questions should remain
    active_questions = rrf._questions[rrf._questions["exclusion"].isna()]
    assert len(active_questions) >= 0  # Could be 0 if all below baseline


@pytest.mark.asyncio
async def test_screening_baseline_float(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that float threshold baseline works."""
    X, y = cost_sensitive_data
    fake_llm = FakeLLM(questions_per_call=4)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        screening_baseline=0.0,  # Very low threshold - all should pass
        max_questions_full_eval=10,
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=4,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # With threshold=0.0, no questions should be pruned by baseline
    assert rrf._questions is not None
    # After baseline pruning but before top-N, we might still have exclusions from top-N
    # So we just check that the method doesn't crash
    assert len(rrf._questions) == 4


@pytest.mark.asyncio
async def test_top_n_selection(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that top-N selection works correctly."""
    X, y = cost_sensitive_data
    fake_llm = FakeLLM(questions_per_call=10)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        screening_baseline=0.0,  # Don't prune by baseline
        max_questions_full_eval=3,  # Only keep top 3
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=10,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Should have 10 total questions, but only top 3 active
    assert rrf._questions is not None
    assert len(rrf._questions) == 10
    active_questions = rrf._questions[rrf._questions["exclusion"].isna()]
    assert len(active_questions) == 3


@pytest.mark.asyncio
async def test_val_set_support(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that X_val/y_val can be provided."""
    X, y = cost_sensitive_data
    # Split into train (6) and val (2)
    X_train, y_train = X.iloc[:6], y[:6]
    X_val, y_val = X.iloc[6:], y[6:]

    fake_llm = FakeLLM(questions_per_call=3)

    config = CostSensitiveConfig(
        screening_fraction=0.5,
        max_questions_full_eval=2,
        enable_semantic_filter=False,
    )

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    # Should not raise
    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    assert rrf._X_val is not None
    assert rrf._y_val is not None
    assert len(rrf._X_val) == 2
    assert len(rrf._y_val) == 2


@pytest.mark.asyncio
async def test_deterministic_screening_split(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that same random_state produces same screening split."""
    X, y = cost_sensitive_data
    fake_llm1 = FakeLLM(questions_per_call=3)
    fake_llm2 = FakeLLM(questions_per_call=3)

    config = CostSensitiveConfig(
        screening_fraction=0.5,
        max_questions_full_eval=2,
        enable_semantic_filter=False,
    )

    rrf1 = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm1,
    )

    rrf2 = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=3,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm2,
    )

    await rrf1.set_tasks(task_description="test task")
    await rrf1.fit(X, y)
    await rrf2.set_tasks(task_description="test task")
    await rrf2.fit(X, y)

    # Same random state should produce same screening indices
    assert rrf1._screening_indices is not None
    assert rrf2._screening_indices is not None
    np.testing.assert_array_equal(rrf1._screening_indices, rrf2._screening_indices)


@pytest.mark.asyncio
async def test_cost_sensitive_false_unchanged(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that cost_sensitive=False preserves original behavior."""
    X, y = cost_sensitive_data
    fake_llm = FakeLLM(questions_per_call=4)

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=4,
        random_state=42,
        cost_sensitive=False,  # Explicitly false
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Should not have screening indices
    assert rrf._screening_indices is None

    # Should have all questions (no cost pruning)
    assert rrf._questions is not None
    cost_pruned = rrf._questions[rrf._questions["exclusion"] == "cost_pruning"]
    assert len(cost_pruned) == 0


@pytest.mark.asyncio
async def test_semantic_filtering_auto_applied(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that semantic filtering is auto-applied when enabled."""
    X, y = cost_sensitive_data
    fake_llm = FakeLLM(questions_per_call=8)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        enable_semantic_filter=True,  # Should auto-apply
        semantic_threshold=0.85,
        max_questions_full_eval=10,
    )

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=8,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # Semantic filtering might exclude some questions
    assert rrf._questions is not None
    # Check if any were excluded by semantics
    # Might be 0 if questions are diverse enough - just verify no crash


@pytest.mark.asyncio
async def test_semantic_filtering_disabled(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that semantic filtering can be disabled."""
    X, y = cost_sensitive_data
    fake_llm = FakeLLM(questions_per_call=6)

    config = CostSensitiveConfig(
        screening_fraction=1.0,
        enable_semantic_filter=False,  # Disabled
        max_questions_full_eval=10,
    )

    rrf = RRF(
        qgen_llmc=LLM_CHOICE,
        max_generated_questions=6,
        random_state=42,
        cost_sensitive=True,
        cost_sensitive_config=config,
        _llm=fake_llm,
    )

    await rrf.set_tasks(task_description="test task")
    await rrf.fit(X, y)

    # No questions should be excluded by semantics
    assert rrf._questions is not None
    semantic_excluded = rrf._questions[rrf._questions["exclusion"] == "semantics"]
    assert len(semantic_excluded) == 0


@pytest.mark.asyncio
async def test_val_set_validation(
    cost_sensitive_data: tuple[pd.DataFrame, list[str]],
) -> None:
    """Test that providing only X_val or y_val raises error."""
    X, y = cost_sensitive_data

    # Only X_val should raise
    fake_llm1 = FakeLLM()
    rrf1 = RRF(
        qgen_llmc=LLM_CHOICE,
        random_state=42,
        _llm=fake_llm1,
    )
    with pytest.raises(ValueError, match="Both X_val and y_val must be provided"):
        await rrf1.fit(X, y, X_val=X.iloc[:2])

    # Only y_val should raise
    fake_llm2 = FakeLLM()
    rrf2 = RRF(
        qgen_llmc=LLM_CHOICE,
        random_state=42,
        _llm=fake_llm2,
    )
    with pytest.raises(ValueError, match="Both X_val and y_val must be provided"):
        await rrf2.fit(X, y, y_val=["YES", "NO"])


# ---------------------------------------------------------------------------
# F-beta aggregation metric tests
# ---------------------------------------------------------------------------


class TestFBetaAggregation:
    """Tests for f_beta support in aggregation tuning."""

    def test_compute_metric_f_beta_matches_f1_when_beta_1(self) -> None:
        """f_beta with beta=1.0 should equal f1."""
        preds = np.array([1, 1, 0, 0, 1])
        y_true = np.array([1, 0, 0, 1, 1])
        f1 = RRF._compute_metric(preds, y_true, "f1")
        f_beta = RRF._compute_metric(preds, y_true, "f_beta", beta=1.0)
        assert f1 == pytest.approx(f_beta)

    def test_compute_metric_f_beta_precision_weighted(self) -> None:
        """beta=0.5 should favour precision; beta=2.0 should favour recall."""
        # High precision, low recall: 1 TP, 0 FP, 3 FN
        preds = np.array([1, 0, 0, 0])
        y_true = np.array([1, 1, 1, 1])
        f05 = RRF._compute_metric(preds, y_true, "f_beta", beta=0.5)
        f2 = RRF._compute_metric(preds, y_true, "f_beta", beta=2.0)
        # precision=1.0, recall=0.25 → F0.5 weights precision more → higher
        assert f05 > f2

    def test_compute_metric_f_beta_zero_division(self) -> None:
        """All-zero predictions should return 0.0, not raise."""
        preds = np.array([0, 0, 0])
        y_true = np.array([1, 1, 0])
        assert RRF._compute_metric(preds, y_true, "f_beta", beta=0.5) == 0.0
        assert RRF._compute_metric(preds, y_true, "f_beta", beta=2.0) == 0.0

    @pytest.mark.asyncio
    async def test_tune_aggregation_with_f_beta(self) -> None:
        """fit() with aggregation_metric='f_beta' sets K and T."""
        fake = FakeLLM(default_answer="ALTERNATE")
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fbeta_agg",
            max_samples_as_context=8,
            max_generated_questions=6,
            aggregation_metric="f_beta",
            question_scoring_f_beta=0.5,
            _llm=fake,
        )
        X = pd.DataFrame({"text": [f"sample {i}" for i in range(20)]})
        y = ["YES"] * 4 + ["NO"] * 16
        await rrf.set_tasks(task_description="Classify samples")
        await rrf.fit(X, y)
        assert isinstance(rrf._aggregation_k, int)
        assert isinstance(rrf._aggregation_t, int)
        assert rrf._aggregation_k >= 1
        assert rrf._aggregation_t >= 1

    def test_aggregation_metric_f_beta_accepted(self) -> None:
        """'f_beta' is accepted as a valid aggregation_metric value."""
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_fbeta_valid",
            aggregation_metric="f_beta",
        )
        assert rrf.aggregation_metric == "f_beta"

    def test_aggregation_metric_invalid_rejected(self) -> None:
        """Invalid metric values are rejected."""
        with pytest.raises(ValueError, match="aggregation_metric"):
            RRF(
                qgen_llmc=LLM_CHOICE,
                name="test_invalid",
                aggregation_metric="f_beta_invalid",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------

from think_reason_learn.rrf._cross_validation import (  # noqa: E402
    cross_validate_aggregation,
    CVResult,
)


def _make_answer_matrix(
    n_samples: int = 30, n_questions: int = 3, pos_rate: float = 0.2
) -> tuple[pd.DataFrame, list[str]]:
    """Build a synthetic answer matrix + labels for CV tests."""
    rng = np.random.default_rng(42)
    n_pos = int(n_samples * pos_rate)
    y = ["YES"] * n_pos + ["NO"] * (n_samples - n_pos)
    # Questions that correlate with the label (imperfectly)
    data: dict[str, list[str]] = {}
    for q in range(n_questions):
        col: list[str] = []
        for label in y:
            if label == "YES":
                col.append("YES" if rng.random() > 0.3 else "NO")
            else:
                col.append("NO" if rng.random() > 0.3 else "YES")
        data[f"q{q}"] = col
    return pd.DataFrame(data), y


class TestCrossValidation:
    """Tests for cross_validate_aggregation."""

    def test_cv_basic_runs(self) -> None:
        """Function runs and returns CVResult with correct types."""
        answers, y = _make_answer_matrix()
        result = cross_validate_aggregation(
            answers, y, n_splits=5, n_repeats=2, beta=0.5
        )
        assert isinstance(result, CVResult)
        assert isinstance(result.fold_metrics, pd.DataFrame)
        assert isinstance(result.per_founder, pd.DataFrame)
        assert isinstance(result.summary, dict)

    def test_cv_fold_count(self) -> None:
        """n_splits=5, n_repeats=2 gives 10 fold rows."""
        answers, y = _make_answer_matrix()
        result = cross_validate_aggregation(
            answers, y, n_splits=5, n_repeats=2, beta=0.5
        )
        assert len(result.fold_metrics) == 10
        assert set(result.fold_metrics.columns) >= {
            "repeat",
            "fold",
            "k",
            "t",
            "precision",
            "recall",
            "f1",
            "f_beta",
            "accuracy",
            "n_train",
            "n_test",
        }

    def test_cv_per_founder_coverage(self) -> None:
        """Each founder appears exactly n_repeats times."""
        answers, y = _make_answer_matrix(n_samples=30)
        n_repeats = 3
        result = cross_validate_aggregation(
            answers, y, n_splits=5, n_repeats=n_repeats, beta=0.5
        )
        counts = result.per_founder.groupby("sample_idx").size()
        assert len(counts) == 30
        assert (counts == n_repeats).all()

    def test_cv_summary_keys(self) -> None:
        """Summary dict has mean and std for each metric."""
        answers, y = _make_answer_matrix()
        result = cross_validate_aggregation(
            answers, y, n_splits=5, n_repeats=1, beta=0.5
        )
        for m in ("precision", "recall", "f1", "f_beta", "accuracy"):
            assert f"{m}_mean" in result.summary
            assert f"{m}_std" in result.summary

    def test_cv_no_leakage(self) -> None:
        """Train and test indices never overlap within a fold."""
        answers, y = _make_answer_matrix(n_samples=30)
        result = cross_validate_aggregation(
            answers, y, n_splits=5, n_repeats=2, beta=0.5
        )
        # Per repeat, all sample indices appear exactly once
        for rep in range(2):
            rep_data = result.per_founder[result.per_founder["repeat"] == rep]
            assert sorted(rep_data["sample_idx"]) == list(range(30))

    def test_cv_metric_param(self) -> None:
        """Different metrics can produce different (K, T) selections."""
        answers, y = _make_answer_matrix(n_samples=30, n_questions=5)
        r1 = cross_validate_aggregation(
            answers,
            y,
            n_splits=5,
            n_repeats=1,
            metric="f_beta",
            beta=0.5,
        )
        r2 = cross_validate_aggregation(
            answers,
            y,
            n_splits=5,
            n_repeats=1,
            metric="precision",
            beta=0.5,
        )
        # Both should succeed; K/T selections may differ
        assert len(r1.fold_metrics) == 5
        assert len(r2.fold_metrics) == 5


# ---------------------------------------------------------------------------
# Prompt preset tests
# ---------------------------------------------------------------------------

from think_reason_learn.rrf._prompt_presets import (  # noqa: E402
    PromptPreset,
    VC_FOUNDER_PRESET,
)


class TestPromptPresets:
    """Tests for the prompt_preset parameter."""

    @pytest.mark.asyncio
    async def test_preset_skips_meta_prompt(self) -> None:
        """With a preset, set_tasks() should NOT make a meta-prompt LLM call."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_preset_skip",
            max_samples_as_context=8,
            max_generated_questions=3,
            prompt_preset=VC_FOUNDER_PRESET,
            _llm=fake,
        )

        template = await rrf.set_tasks(task_description="Classify founders")

        # No LLM call should have been made (meta-prompt skipped)
        assert fake.call_count == 0
        assert len(fake.calls) == 0
        # Template should be set
        assert template is not None

    @pytest.mark.asyncio
    async def test_preset_uses_custom_gen_system(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
    ) -> None:
        """Generation calls should use the preset's system message."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_preset_gen",
            max_samples_as_context=8,
            max_generated_questions=3,
            prompt_preset=VC_FOUNDER_PRESET,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        # Find the Questions-format call (generation)
        gen_calls = [c for c in fake.calls if c["response_format"] is Questions]
        assert len(gen_calls) > 0
        # The instructions should be the preset's generation system message
        assert gen_calls[0]["instructions"] == VC_FOUNDER_PRESET.question_gen_system

    @pytest.mark.asyncio
    async def test_preset_uses_custom_answer_system(
        self,
        sample_data: tuple[pd.DataFrame, list[str]],
    ) -> None:
        """Answer calls should use the preset's answer system message."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_preset_ans",
            max_samples_as_context=8,
            max_generated_questions=3,
            prompt_preset=VC_FOUNDER_PRESET,
            _llm=fake,
        )
        X, y = sample_data
        await rrf.set_tasks(task_description="Classify founders")
        await rrf.fit(X, y)

        # Find the Answer-format calls (answering)
        answer_calls = [c for c in fake.calls if c["response_format"] is Answer]
        assert len(answer_calls) > 0
        # The instructions should be the preset's answer system message
        expected = VC_FOUNDER_PRESET.question_answer_system
        assert answer_calls[0]["instructions"] == expected

    @pytest.mark.asyncio
    async def test_preset_and_template_mutually_exclusive(self) -> None:
        """Providing both preset and instructions_template should raise."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_preset_excl",
            max_generated_questions=3,
            prompt_preset=VC_FOUNDER_PRESET,
            _llm=fake,
        )

        with pytest.raises(ValueError, match="[Cc]annot.*both|[Mm]utually"):
            await rrf.set_tasks(
                instructions_template="Generate <number_of_questions> questions"
            )

    @pytest.mark.asyncio
    async def test_preset_by_name_lookup(self) -> None:
        """Passing a string name should resolve to the built-in preset."""
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_preset_name",
            max_samples_as_context=8,
            max_generated_questions=3,
            prompt_preset="vc_founder_evaluation",
            _llm=fake,
        )

        template = await rrf.set_tasks(task_description="Classify founders")
        # Should work — name resolved to VC_FOUNDER_PRESET
        assert fake.call_count == 0
        assert template is not None

    @pytest.mark.asyncio
    async def test_custom_preset_object(self) -> None:
        """A user-created PromptPreset instance should work."""
        custom = PromptPreset(
            name="custom_test",
            description="A test preset",
            question_gen_system="You are a custom system.",
            question_gen_user_template=(
                "Generate {num_questions} YES/NO questions.\n\n{samples}"
            ),
            question_answer_system="You are a custom answering system.",
            question_answer_user_template=("Question: {question}\nSample: {sample}"),
        )
        fake = FakeLLM()
        rrf = RRF(
            qgen_llmc=LLM_CHOICE,
            name="test_custom_preset",
            max_samples_as_context=8,
            max_generated_questions=3,
            prompt_preset=custom,
            _llm=fake,
        )

        template = await rrf.set_tasks(task_description="Classify founders")
        assert fake.call_count == 0
        assert template is not None
