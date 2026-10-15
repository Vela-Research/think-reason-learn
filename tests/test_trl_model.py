"""Every LLM-powered model is a TRLModel, with the same task, fit, save and load API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path

import pandas as pd
import pytest

from think_reason_learn.core import TRLModel
from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining
from think_reason_learn.rrf import RRF
from think_reason_learn.verifiable_rl import VerifiableRL

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]
TEMPLATE = "Ask <number_of_questions> questions about founders."


def _rrf(name: str | None = None) -> RRF:
    return RRF(qgen_llmc=CHAT, qanswer_llmc=CHAT, name=name)


def _pi(name: str | None = None) -> PolicyInduction:
    return PolicyInduction(gen_llmc=CHAT, predict_llmc=CHAT, name=name)


def _rrm(name: str | None = None) -> ReasonedRuleMining:
    return ReasonedRuleMining(reason_llmc=CHAT, name=name)


MODELS: list[tuple[Callable[..., TRLModel], str]] = [
    (_rrf, "rrfs"),
    (_pi, "policy_induction"),
    (_rrm, "reasoned_rule_mining"),
]
TEMPLATED: list[Callable[[], RRF]] = [_rrf]


@pytest.mark.parametrize(("make", "default_dir"), MODELS)
def test_every_llm_model_is_a_trl_model(
    make: Callable[..., TRLModel], default_dir: str
) -> None:
    assert isinstance(make(), TRLModel)


def test_verifiable_rl_is_not_a_trl_model() -> None:
    assert not issubclass(VerifiableRL, TRLModel)


def test_a_model_missing_an_abstract_method_cannot_be_instantiated() -> None:
    class NoPredict(TRLModel):
        _default_save_dir = "no_predict"

        async def set_task(self, task_description: str) -> None:
            pass

        async def fit(self, X: pd.DataFrame, y: Sequence[str]) -> NoPredict:
            return self

        def save(self, dir_path: str | PathLike[str] | None = None) -> None:
            pass

        @classmethod
        def load(cls, dir_path: str | PathLike[str]) -> NoPredict:
            return cls()

    with pytest.raises(TypeError, match="predict"):
        NoPredict()  # pyright: ignore[reportAbstractUsage]  (what is under test)


@pytest.mark.parametrize(("make", "default_dir"), MODELS)
def test_default_save_path_is_named_after_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make: Callable[..., TRLModel],
    default_dir: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    model = make(name="m")

    assert model.save_path == (tmp_path / default_dir / "m").resolve()
    assert repr(model) == f"{type(model).__name__}(name=m)"


@pytest.mark.parametrize(("make", "default_dir"), MODELS)
def test_load_takes_dir_path(
    tmp_path: Path, make: Callable[..., TRLModel], default_dir: str
) -> None:
    model = make(name="m")
    model.save(tmp_path / "saved")

    loaded = type(model).load(dir_path=tmp_path / "saved")

    assert type(loaded) is type(model)
    assert loaded.name == "m"


@pytest.mark.parametrize("make", TEMPLATED)
@pytest.mark.asyncio
async def test_set_task_takes_the_task_description_first(
    make: Callable[[], RRF],
) -> None:
    model = make()

    assert await model.set_task(None, TEMPLATE) == TEMPLATE
    assert model.question_gen_instructions_template == TEMPLATE


@pytest.mark.parametrize("make", TEMPLATED)
@pytest.mark.asyncio
async def test_set_task_keeps_the_description_with_a_custom_template(
    make: Callable[[], RRF],
) -> None:
    """Jev's questions and GPTree's critic read the description; keep it."""
    model = make()

    await model.set_task("Predict founder success.", TEMPLATE)
    assert model.task_description == "Predict founder success."


@pytest.mark.parametrize("make", TEMPLATED)
@pytest.mark.asyncio
async def test_set_tasks_warns_and_forwards(
    make: Callable[[], RRF], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make()
    calls: list[tuple[str | None, str | None]] = []

    async def set_task(
        task_description: str | None = None, instructions_template: str | None = None
    ) -> str:
        calls.append((task_description, instructions_template))
        return "template"

    monkeypatch.setattr(model, "set_task", set_task)

    with pytest.warns(DeprecationWarning, match="set_task"):
        assert await model.set_tasks("tmpl", "desc") == "template"
    with pytest.warns(DeprecationWarning):
        await model.set_tasks(task_description="desc", instructions_template="tmpl")
    with pytest.warns(DeprecationWarning):
        await model.set_tasks(task_description="desc")
    with pytest.warns(DeprecationWarning):
        await model.set_tasks("tmpl")

    assert calls == [
        ("desc", "tmpl"),
        ("desc", "tmpl"),
        ("desc", None),
        (None, "tmpl"),
    ]


@pytest.mark.parametrize("make", TEMPLATED)
@pytest.mark.asyncio
async def test_set_tasks_still_sets_the_template(
    make: Callable[[], RRF],
) -> None:
    model = make()

    with pytest.warns(DeprecationWarning):
        assert await model.set_tasks(TEMPLATE) == TEMPLATE

    assert model.question_gen_instructions_template == TEMPLATE
