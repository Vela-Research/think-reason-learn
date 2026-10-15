"""RRF combiners use Jev's probabilities as features (answer_features)."""

from __future__ import annotations

import hashlib
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
    rrf = await _fit(_rrf(tmp_path, aggregation_method="vote"))

    result = await rrf.predict_founder_level(X)

    k = int(result["k"].iloc[0])
    assert set(result["prediction"]) == {"YES"}
    assert list(result["yes_count"]) == [k] * len(PEOPLE)


@pytest.mark.asyncio
async def test_vote_with_one_question_can_still_say_yes(tmp_path: Path) -> None:
    probs = {"A": 0.9, "B": 0.3, "C": 0.6, "D": 0.8, "E": 0.2, "F": 0.95}

    def prob(state: str, instructions: str) -> float:
        return probs[state.split(": ", 1)[1][0]]

    rrf = await _fit(
        _rrf(
            tmp_path,
            aggregation_method="vote",
            aggregation_max_k=1,
            _llm=FakeJevLLM(prob=prob),
        )
    )

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
    rrf = await _fit(_rrf(tmp_path, aggregation_method="vote"))

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


@pytest.mark.asyncio
async def test_default_rrf_learns_from_jev_probabilities(tmp_path: Path) -> None:
    # All defaults: Jev answers, elastic-net combiner, default Cs and CV folds.
    # The earlier Cs grid (0.05, 0.1, 0.5) zeroed every weight on this data.
    rrf = await _fit(_rrf(tmp_path))

    result = await rrf.predict_founder_level(X)

    assert rrf.aggregation_method == "elasticnet"
    assert any(w != 0 for w in (rrf._aggregation_weights or {}).values())
    assert list(result["prediction"]) == LABELS


# ---------------------------------------------------------------------------
# The default elastic-net combiner on edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rare_positives_still_get_yes_predictions(tmp_path: Path) -> None:
    # 10 positives in 1,000 with a weak, noisy signal: the model's P(YES) stays
    # below 0.03, under the 0.05 floor of the old fixed threshold grid.
    n, positives = 1000, 10

    def prob(state: str, instructions: str) -> float:
        digest = hashlib.sha256(f"{state}|{instructions}".encode()).hexdigest()
        noise = int(digest[:8], 16) / 0xFFFFFFFF - 0.5
        return 0.5 + (0.05 if "positive" in state else -0.05) + 0.4 * noise

    rows = [f"positive {i}" for i in range(positives)]
    rows += [f"negative {i}" for i in range(n - positives)]
    rrf = RRF(
        qgen_llmc=CHAT,
        name="rare",
        save_path=tmp_path,
        max_samples_as_context=100,
        max_generated_questions=4,
        _llm=FakeJevLLM(prob=prob, questions_per_call=1),
    )
    await rrf.set_tasks(task_description="Classify founders")
    await rrf.fit(
        pd.DataFrame({"data": rows}), ["YES"] * positives + ["NO"] * (n - positives)
    )

    features = rrf._feature_matrix(list(rrf._aggregation_feature_order or []))
    proba = rrf._elasticnet_proba(features)
    predicted = proba >= (rrf._aggregation_threshold or 0)

    assert proba.max() < 0.05
    assert predicted[:positives].any()
    assert predicted.sum() < n / 2


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [2, 4])
async def test_tiny_training_sets_fit(tmp_path: Path, rows: int) -> None:
    rrf = _rrf(tmp_path)
    await rrf.set_tasks(task_description="Classify founders")

    await rrf.fit(pd.DataFrame({"data": PEOPLE[:rows]}), LABELS[:rows])

    assert rrf._aggregation_threshold is not None


def test_elasticnet_penalty_argument_follows_sklearn_version() -> None:
    from think_reason_learn.rrf._rrf import _elasticnet_penalty

    assert _elasticnet_penalty("1.7.2") == {"penalty": "elasticnet"}
    assert _elasticnet_penalty("1.8.0") == {}
    assert _elasticnet_penalty("1.10.1") == {}


@pytest.mark.asyncio
async def test_k_and_t_are_refused_outside_the_vote(tmp_path: Path) -> None:
    rrf = await _fit(_rrf(tmp_path))

    with pytest.raises(ValueError, match='aggregation_method="vote"'):
        await rrf.predict_founder_level(X, k=2, t=1)


@pytest.mark.asyncio
async def test_unfitted_elasticnet_raises_before_asking_jev(tmp_path: Path) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, aggregation_method="vote", _llm=fake))
    rrf.aggregation_method = "elasticnet"
    fake.answer_requests.clear()

    with pytest.raises(ValueError, match="not fitted"):
        await rrf.predict_founder_level(X)

    assert fake.answer_requests == []


@pytest.mark.asyncio
async def test_prediction_asks_only_the_questions_the_model_weighs(
    tmp_path: Path,
) -> None:
    fake = FakeJevLLM(prob=_prob)
    rrf = await _fit(_rrf(tmp_path, _llm=fake))
    await rrf.add_question("Did the founder work at a large tech company?")
    fake.answer_requests.clear()

    await rrf.predict_founder_level(X)

    weighted = {q for q, w in (rrf._aggregation_weights or {}).items() if w != 0}
    assert weighted
    assert all(set(r.questions) == weighted for r in fake.answer_requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("features", ["probability", "binary"])
async def test_excluding_a_question_after_fit_keeps_the_model(
    tmp_path: Path, features: str, caplog: pytest.LogCaptureFixture
) -> None:
    from think_reason_learn.rrf import QuestionExclusion

    def prob(state: str, instructions: str) -> float:
        return (
            0.9 if LABELS["ABCDEF".index(state.split(": ", 1)[1][0])] == "YES" else 0.3
        )

    fake = FakeJevLLM(prob=prob)
    rrf = await _fit(_rrf(tmp_path, answer_features=features, _llm=fake))
    before = await rrf.predict_founder_level(X)
    weights = rrf._aggregation_weights or {}
    weighted = {q for q, w in weights.items() if w != 0}
    top = max(weights, key=lambda q: abs(weights[q]))
    await rrf.update_question_exclusion(top, QuestionExclusion.EXPERT)
    fake.answer_requests.clear()

    with caplog.at_level("WARNING"):
        after = await rrf.predict_founder_level(X)

    assert list(after["prediction"]) == list(before["prediction"])
    assert {q for r in fake.answer_requests for q in r.questions} == weighted
    assert "questions changed after fit" in caplog.text


@pytest.mark.asyncio
async def test_threshold_can_predict_all_no(tmp_path: Path) -> None:
    # With accuracy as the metric and 2 positives in 100, predicting NO for
    # everyone (0.98) beats any cut, because 9 negatives look like positives.
    n = 100

    def prob(state: str, instructions: str) -> float:
        i = int(state.split("sample ")[1])
        return 0.9 if i < 11 else 0.1

    rrf = RRF(
        qgen_llmc=CHAT,
        name="all_no",
        save_path=tmp_path,
        aggregation_metric="accuracy",
        max_generated_questions=1,
        _llm=FakeJevLLM(prob=prob, questions_per_call=1),
    )
    await rrf.set_tasks(task_description="Classify founders")
    await rrf.fit(
        pd.DataFrame({"data": [f"sample {i}" for i in range(n)]}),
        ["YES"] * 2 + ["NO"] * (n - 2),
    )

    features = rrf._feature_matrix(list(rrf._aggregation_feature_order or []))
    proba = rrf._elasticnet_proba(features)

    assert not (proba >= (rrf._aggregation_threshold or 0)).any()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["elasticnet", "vote"])
async def test_fit_tunes_the_combiner_once(
    tmp_path: Path, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rrf = _rrf(tmp_path, aggregation_method=method)
    original = rrf._tune_aggregation
    calls: list[str] = []

    def counting() -> None:
        calls.append(method)
        original()

    monkeypatch.setattr(rrf, "_tune_aggregation", counting)

    await _fit(rrf)

    assert calls == [method]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"aggregation_method": "vote"},
        {"aggregation_method": "elasticnet", "answer_features": "binary"},
    ],
)
async def test_yes_no_paths_keep_rows_apart_with_repeated_index_labels(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    def prob(state: str, instructions: str) -> float:
        label = LABELS["ABCDEF".index(state.split(": ", 1)[1][0])]
        return 0.9 if label == "YES" else 0.3

    rrf = await _fit(_rrf(tmp_path, _llm=FakeJevLLM(prob=prob), **config))
    X_dup = pd.DataFrame({"data": PEOPLE}, index=pd.Index([0, 0, 1, 1, 2, 2]))

    plain = await rrf.predict_founder_level(X)
    repeated = await rrf.predict_founder_level(X_dup)

    assert list(repeated.index) == list(X_dup.index)
    score = "yes_count" if config["aggregation_method"] == "vote" else "probability"
    for column in ("prediction", score):
        assert list(repeated[column]) == list(plain[column])


@pytest.mark.asyncio
async def test_chat_answerer_predicts_with_a_string_index(tmp_path: Path) -> None:
    from tests.fake_llm import FakeLLM

    rrf = await _fit(
        _rrf(
            tmp_path,
            qanswer_llmc=CHAT,
            aggregation_method="vote",
            _llm=FakeLLM("ALTERNATE"),
        )
    )
    X_named = pd.DataFrame({"data": PEOPLE}, index=pd.Index(list("abcdef")))

    result = await rrf.predict_founder_level(X_named)

    assert list(result.index) == list("abcdef")
