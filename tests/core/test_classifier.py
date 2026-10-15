"""The estimator base and the shared contract, on a small model built from it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from think_reason_learn.core._classifier import Labels, Proba, TRLClassifier
from think_reason_learn.core._data import encode_labels, example_batches, to_texts
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import LLMSession, Models, YesNoQuestion
from think_reason_learn.testing import FakeLLM, check_classifier, fake_llms


class Question(BaseModel):
    text: str


class OneQuestion(TRLClassifier):
    """A model with one written question: P(class | answer) from training."""

    def __init__(
        self,
        task: str | None = None,
        generator: Models | None = None,
        answerer: Models | None = None,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.generator = generator
        self.answerer = answerer
        self.verbose = verbose

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        written = await session.generate(
            self._models("generator", "generate"),
            f"Task: {self.task}",
            role="generator",
            schema=Question,
        )
        self.question_ = written.text
        yes = await self._yes(texts, session)
        table = np.ones((2, len(self.classes_)))  # Laplace smoothing
        for answer, label in zip(yes, y):
            table[int(answer), label] += 1
        self.table_ = table / table.sum(axis=1, keepdims=True)

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        return self.table_[(await self._yes(texts, session)).astype(int)]

    async def _yes(self, texts: list[str], session: LLMSession) -> np.ndarray:
        with self._progress(len(texts), "answers") as progress:
            answers = await session.answer(
                self._models("answerer", "answer"),
                texts,
                {"q": YesNoQuestion(self.question_)},
                role="answerer",
                progress=progress,
            )
        return np.array([a["q"] is not None and a["q"].label == "YES" for a in answers])


X = pd.DataFrame(
    {
        "bio": ["an engineer", "a lawyer", "an engineer too", "a chef"],
        "age": [30, 40, None, 50],
    }
)
y = ["hit", "miss", "hit", "miss"]
WRITER = FakeLLM(generate=lambda request: Question(text="Is the founder an engineer?"))
JUDGE = FakeLLM(
    answer=lambda sample, question: 0.9 if "engineer" in sample else 0.1,
    kind="decision",
)


def make() -> OneQuestion:
    return OneQuestion(
        task="Will the founder succeed?", generator="fake:writer", answerer="fake:judge"
    )


def test_contract(tmp_path: Path) -> None:
    with fake_llms(writer=WRITER, judge=JUDGE):
        check_classifier(make, X, y, tmp_path)


def test_predictions_and_usage() -> None:
    with fake_llms(writer=WRITER, judge=JUDGE):
        model = make().fit(X, y)
        assert list(model.predict(["an engineer", "a pilot"])) == ["hit", "miss"]
    assert model.question_ == "Is the founder an engineer?"
    assert set(model.usage_.by("role")["role"]) == {"generator", "answerer"}
    assert list(model.feature_names_in_) == ["bio", "age"]


@pytest.mark.asyncio
async def test_async_twins() -> None:
    with fake_llms(writer=WRITER, judge=JUDGE):
        model = await make().afit(X, y)
        assert list(await model.apredict(["a chef"])) == ["miss"]
        assert (await model.apredict_proba(["a chef"])).shape == (1, 2)


def test_missing_role_and_wrong_columns() -> None:
    with fake_llms(writer=WRITER, judge=JUDGE):
        with pytest.raises(ValueError, match="needs generator="):
            OneQuestion(task="t").fit(X, y)
        model = make().fit(X, y)
        with pytest.raises(DataError, match="columns"):
            model.predict(X.rename(columns={"bio": "text"}))


def test_decision_model_cannot_write() -> None:
    with fake_llms(writer=WRITER, judge=JUDGE):
        with pytest.raises(ValueError, match="decision model"):
            OneQuestion(task="t", generator="fake:judge", answerer="fake:judge").fit(
                X, y
            )


def test_saved_model_has_no_key(tmp_path: Path) -> None:
    from think_reason_learn.llm import LLM

    with fake_llms(writer=WRITER, judge=JUDGE):
        model = OneQuestion(
            task="t",
            generator=LLM("fake:writer", api_key="secret"),
            answerer="fake:judge",
        )
        folder = model.fit(X, y).save(tmp_path / "m")
    assert "secret" not in (folder / "model.json").read_text()
    loaded = OneQuestion.load(folder)
    assert loaded.generator.api_key is None  # type: ignore[union-attr]  # an LLM here


def test_verbose_shows_progress(capsys: pytest.CaptureFixture[str]) -> None:
    with fake_llms(writer=WRITER, judge=JUDGE):
        OneQuestion(
            task="t", generator="fake:writer", answerer="fake:judge", verbose=1
        ).fit(X, y)
    assert "answers" in capsys.readouterr().err


class TestData:
    def test_rows_become_column_lines(self) -> None:
        texts, names = to_texts(X)
        assert texts[2] == "bio: an engineer too"  # missing values left out
        assert texts[0] == "bio: an engineer\nage: 30.0"
        assert names == ["bio", "age"]
        assert to_texts(["a", "b"]) == (["a", "b"], None)

    def test_bad_input(self) -> None:
        with pytest.raises(DataError):
            to_texts([])
        with pytest.raises(DataError, match="two classes"):
            encode_labels(["a", "a"])

    def test_labels_are_sorted_codes(self) -> None:
        classes, codes = encode_labels(["yes", "no", "yes"])
        assert list(classes) == ["no", "yes"] and list(codes) == [1, 0, 1]

    def test_example_batches_keep_the_ratio_and_reach_rare_classes(self) -> None:
        codes = np.array([0] * 98 + [1] * 2, dtype=np.intp)
        classes = np.array(["common", "rare"], dtype=object)
        batches = example_batches(
            codes,
            classes,
            n_batches=5,
            batch_size=10,
            class_ratio="balanced",
            rng=np.random.default_rng(0),
        )
        assert all(len(b) == 10 and (codes[b] == 1).sum() == 5 for b in batches)
        weighted = example_batches(
            codes,
            classes,
            n_batches=1,
            batch_size=10,
            class_ratio={"common": 4, "rare": 1},
            rng=np.random.default_rng(0),
        )
        assert (codes[weighted[0]] == 1).sum() == 2

    def test_missing_class_is_filled_from_others(self) -> None:
        codes = np.zeros(10, dtype=np.intp)
        classes = np.array(["a", "b"], dtype=object)
        [batch] = example_batches(
            codes,
            classes,
            n_batches=1,
            batch_size=4,
            class_ratio="balanced",
            rng=np.random.default_rng(0),
        )
        assert len(batch) == 4
        with pytest.raises(ValueError, match="unknown classes"):
            example_batches(
                codes,
                classes,
                n_batches=1,
                batch_size=4,
                class_ratio={"c": 1},
                rng=np.random.default_rng(0),
            )
