"""Saving and loading: one JSON file that rebuilds the model without running code."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from think_reason_learn import LLM, Price
from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.core._persistence import FORMAT, decode, encode
from think_reason_learn.exceptions import DataError
from think_reason_learn.rrf import RRF
from think_reason_learn.testing import FakeLLM


def test_every_model_comes_back_whole(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    tmp_path: Path,
) -> None:
    (_, make), (X, y) = any_model, founders
    model = make().fit(X, y)
    folder = model.save(tmp_path / "m")
    loaded = type(model).load(folder)
    assert [p.name for p in folder.iterdir()] == ["model.json"]  # no pickle
    assert loaded.get_params() == model.get_params()
    fitted = sorted(k for k in vars(model) if k.endswith("_"))
    assert sorted(k for k in vars(loaded) if k.endswith("_")) == fitted
    assert loaded.usage_.calls == model.usage_.calls
    assert (loaded.predict_proba(X) == model.predict_proba(X)).all()


def test_values_come_back_as_they_were() -> None:
    frame = pd.DataFrame({"question": ["a", "b"], "score": [0.5, np.nan]})
    value = {
        "tuple": (1, "two", None),
        "array": np.array([1.5, 2.0]),
        "labels": np.array(["no", "yes"]),
        "keys": {1: "one", 2: "two"},
        "frame": frame,
        "llm": LLM("fake:x", temperature=0.0, price=Price(1.0), api_key="secret"),
        "flag": np.array([True])[0],
    }
    text = json.dumps(encode(value))
    back = decode(json.loads(text))
    assert "secret" not in text
    assert json.dumps(encode(back)) == text  # same types: tuples, dtypes, keys
    assert isinstance(back, dict) and isinstance(back["llm"], LLM)


def test_a_file_cannot_make_load_build_other_objects(
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    tmp_path: Path,
) -> None:
    folder = _saved_rrf(founders, tmp_path)
    document = json.loads((folder / "model.json").read_text())
    document["params"]["task"] = {
        "__trl__": "data",
        "class": "subprocess:Popen",
        "fields": {"args": ["touch", str(tmp_path / "pwned")]},
    }
    (folder / "model.json").write_text(json.dumps(document))
    with pytest.raises(DataError, match="not one of this library's"):
        RRF.load(folder)
    assert not (tmp_path / "pwned").exists()


def test_a_0_2_folder_says_to_refit(tmp_path: Path) -> None:
    (tmp_path / "rrf.json").write_text("{}")
    with pytest.raises(DataError, match="saved by think-reason-learn 0.2"):
        RRF.load(tmp_path)


def test_another_format_is_refused_and_another_version_warned(
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    folder = _saved_rrf(founders, tmp_path)
    file = folder / "model.json"
    document = json.loads(file.read_text())
    assert document["format"] == FORMAT and document["library"]
    file.write_text(json.dumps({**document, "library": "0.0.1"}))
    RRF.load(folder)
    assert "saved by think-reason-learn 0.0.1" in caplog.text
    file.write_text(json.dumps({**document, "format": FORMAT + 1}))
    with pytest.raises(DataError, match=f"save format {FORMAT}"):
        RRF.load(folder)


def _saved_rrf(founders: tuple[list[str], list[str]], tmp_path: Path) -> Path:
    model = RRF(task="t", generator="fake:writer", answerer="fake:judge", cv=2)
    return model.fit(*founders).save(tmp_path / "rrf")
