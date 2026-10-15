"""Every LLM-powered model is a TRLModel, with the same task, fit, save and load API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path

import pandas as pd
import pytest

from think_reason_learn.core import TRLModel
from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining
from think_reason_learn.verifiable_rl import VerifiableRL

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]
TEMPLATE = "Ask <number_of_questions> questions about founders."


def _rrm(name: str | None = None) -> ReasonedRuleMining:
    return ReasonedRuleMining(reason_llmc=CHAT, name=name)


MODELS: list[tuple[Callable[..., TRLModel], str]] = [
    (_rrm, "reasoned_rule_mining"),
]


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
