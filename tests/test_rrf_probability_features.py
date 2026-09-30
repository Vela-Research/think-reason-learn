"""RRF combiners use Jev's probabilities as features (answer_features)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from think_reason_learn.core.llms import JevChoice, LLMChoice, OpenAIChoice
from think_reason_learn.rrf import RRF
from tests.fake_jev import FakeJevLLM

pytestmark = pytest.mark.usefixtures("typesafe_key")

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]
PEOPLE = [
    "A: 30yo woman, CS Stanford, 6yr Google, AI healthcare startup, $2M seed.",
    "B: 25yo man, marketing NYU, 3yr Apple marketing mgr, social media app.",
    "C: LA doctor, UCLA, remote monitoring platform, no tech/startup exp.",
    "D: 40yo man, law UChicago, corporate lawyer, legal-tech idea, no tech bg.",
    "E: 28yo woman, CE Berkeley, fintech YC startup, own fintech product.",
    "F: 32yo man, MBA Columbia, marketing Apple/Spotify, subscription box.",
]
LABELS = ["YES", "NO", "NO", "YES", "NO", "YES"]
# Every answer is YES at 0.5, so YES/NO features carry no signal; the
# probabilities separate the classes.
PROB = {"YES": 0.9, "NO": 0.6}


def _prob(state: str, instructions: str) -> float:
    letter = state.split(": ", 1)[1][0]
    return PROB[LABELS["ABCDEF".index(letter)]]


def _rrf(tmp_path: Path, **kwargs: Any) -> RRF:
    return RRF(
        qgen_llmc=CHAT,
        qanswer_llmc=kwargs.pop("qanswer_llmc", [JevChoice()]),
        name="prob_rrf",
        save_path=tmp_path,
        max_samples_as_context=5,
        max_generated_questions=6,
        _llm=kwargs.pop("_llm", FakeJevLLM(prob=_prob)),
        **kwargs,
    )


async def _fit(rrf: RRF) -> RRF:
    await rrf.set_tasks(task_description="Classify founders")
    return await rrf.fit(pd.DataFrame({"data": PEOPLE}), LABELS)


X = pd.DataFrame({"data": PEOPLE})


ELASTICNET: dict[str, Any] = {
    "aggregation_method": "elasticnet",
    "elasticnet_cs": (10.0, 100.0),
    "elasticnet_cv": 2,
}


@pytest.mark.asyncio
async def test_elasticnet_learns_from_probabilities(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path, **ELASTICNET))

    result = await rrf.predict_founder_level(X)

    assert any(w != 0 for w in (rrf._aggregation_weights or {}).values())
    assert list(result["prediction"]) == LABELS


@pytest.mark.asyncio
async def test_binary_setting_keeps_yes_no_features(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path, answer_features="binary", **ELASTICNET))

    result = await rrf.predict_founder_level(X)

    assert all(w == 0 for w in (rrf._aggregation_weights or {}).values())
    assert result["prediction"].nunique() == 1


@pytest.mark.asyncio
async def test_vote_keeps_counting_yes_answers(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path))

    result = await rrf.predict_founder_level(X)

    k = int(result["k"].iloc[0])
    assert set(result["prediction"]) == {"YES"}
    assert list(result["yes_count"]) == [k] * len(PEOPLE)


@pytest.mark.asyncio
async def test_vote_with_one_question_can_still_say_yes(tmp_path: Path) -> None:
    probs = {"A": 0.9, "B": 0.3, "C": 0.6, "D": 0.8, "E": 0.2, "F": 0.95}

    def prob(state: str, instructions: str) -> float:
        return probs[state.split(": ", 1)[1][0]]

    rrf = await _fit(_rrf(tmp_path, aggregation_max_k=1, _llm=FakeJevLLM(prob=prob)))

    result = await rrf.predict_founder_level(X)

    assert list(result["prediction"]) == ["YES", "NO", "YES", "YES", "NO", "YES"]


@pytest.mark.asyncio
async def test_response_matrix_holds_probabilities_for_elasticnet(
    tmp_path: Path,
) -> None:
    rrf = await _fit(_rrf(tmp_path, **ELASTICNET))

    matrix = await rrf._build_response_matrix(X)

    expected = np.array([PROB[label] for label in LABELS])
    for qid in matrix.columns:
        assert matrix[qid].to_numpy() == pytest.approx(expected)


@pytest.mark.asyncio
async def test_response_matrix_holds_yes_no_for_the_vote(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path))

    matrix = await rrf._build_response_matrix(X)

    assert set(np.unique(matrix.to_numpy())) == {1}


@pytest.mark.asyncio
async def test_duplicate_index_labels_keep_rows_apart(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path, **ELASTICNET))
    X_dup = pd.DataFrame({"data": PEOPLE}, index=pd.Index([0, 0, 1, 1, 2, 2]))

    result = await rrf.predict_founder_level(X_dup)

    assert list(result["prediction"]) == LABELS


@pytest.mark.asyncio
async def test_predict_still_yields_yes_no(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path))

    answers = {a async for _, _, a, _ in rrf.predict(X)}

    assert answers == {"YES"}


@pytest.mark.asyncio
async def test_chat_answers_stay_binary_features(tmp_path: Path) -> None:
    from tests.fake_llm import FakeLLM

    rrf = await _fit(_rrf(tmp_path, qanswer_llmc=CHAT, _llm=FakeLLM("ALTERNATE")))
    qids = list(rrf.get_answers().columns)

    features = rrf._feature_matrix(qids)

    assert set(np.unique(features.to_numpy())) <= {0.0, 1.0}


@pytest.mark.asyncio
async def test_setting_survives_save_and_load(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path, answer_features="binary"))
    rrf.save(tmp_path / "saved")

    assert RRF.load(tmp_path / "saved").answer_features == "binary"


@pytest.mark.asyncio
async def test_models_saved_before_the_setting_load_as_binary(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path))
    rrf.save(tmp_path / "saved")
    manifest_path = next((tmp_path / "saved").glob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("answer_features")
    manifest_path.write_text(json.dumps(manifest))

    assert RRF.load(tmp_path / "saved").answer_features == "binary"


def test_unknown_setting_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="answer_features"):
        _rrf(tmp_path, answer_features="logits")
