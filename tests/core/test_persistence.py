"""Saving and loading: one JSON file that rebuilds the model without running code."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from think_reason_learn import LLM, Price
from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.core._persistence import FORMAT, decode, encode
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import ChoiceQuestion
from think_reason_learn.llm._cache import JSON
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
    assert encode(loaded.get_params()) == encode(model.get_params())  # keys aside
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


@pytest.mark.parametrize(
    ("where", "planted"),
    [
        ("fitted", {"answerer": "openai:gpt-5"}),  # a setting, not something learned
        (
            "fitted",
            {"x_": {"__trl__": "data", "class": "think_reason_learnx:A", "fields": {}}},
        ),
        ("fitted", {"x_": {"__trl__": "data", "class": "x:A", "fields": {}}}),
        ("params", {"no_such_setting": 1}),
        ("fitted", {"x_": {"__trl__": "array", "dtype": "V9999", "items": [0]}}),
    ],
)
def test_a_damaged_file_is_refused(
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    tmp_path: Path,
    where: str,
    planted: dict[str, object],
) -> None:
    folder = _saved_rrf(founders, tmp_path)
    document = json.loads((folder / "model.json").read_text())
    document[where].update(planted)
    (folder / "model.json").write_text(json.dumps(document))
    with pytest.raises(DataError):
        RRF.load(folder)


def test_usage_is_saved_compactly(
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    tmp_path: Path,
) -> None:
    folder = _saved_rrf(founders, tmp_path)
    usage = json.loads((folder / "model.json").read_text())["fitted"]["usage_"]
    size = len(json.dumps(usage["rows"])) / len(usage["rows"])
    assert size < 150  # bytes per call, so a large run's file stays small


def test_text_arrays_take_their_width_from_the_items() -> None:
    labels = decode({"__trl__": "array", "dtype": "<U10000000", "items": ["a", "b"]})
    assert isinstance(labels, np.ndarray) and labels.nbytes == 8


def test_a_file_never_makes_load_import_a_module() -> None:
    import sys

    named = "think_reason_learn.no_such_module"
    with pytest.raises(DataError, match="not one of this library's"):
        decode({"__trl__": "data", "class": f"{named}:A", "fields": {}})
    assert named not in sys.modules


def test_a_password_in_a_base_url_is_not_saved() -> None:
    llm = LLM("my-model", base_url="https://me:s3cret@llm.example.com/v1")
    text = json.dumps(encode(llm))
    assert "s3cret" not in text and "me@llm.example.com" in text


def test_choice_questions_come_back_with_their_descriptions() -> None:
    # GPTree saves its questions, so one saved before descriptions existed loads too.
    plain = ChoiceQuestion("Stage?", ("seed", "series_a"))
    described = ChoiceQuestion(plain.text, plain.labels, {"seed": "no priced round"})
    assert decode(json.loads(json.dumps(encode([plain, described])))) == [
        plain,
        described,
    ]
    saved = encode(plain)
    assert isinstance(saved, dict) and isinstance(saved["fields"], dict)
    del saved["fields"]["descriptions"]
    assert decode(saved) == plain


def strict(text: str) -> JSON:
    """``json.loads`` as a strict reader (such as JavaScript's) does it."""

    def refuse(constant: str) -> object:
        raise ValueError(f"not strict JSON: {constant}")

    parsed: JSON = json.loads(text, parse_constant=refuse)
    return parsed


def test_non_finite_floats_are_strict_json() -> None:
    value: dict[str, object] = {
        "floats": [np.nan, np.inf, -np.inf, 1.5],
        "array": np.array([np.nan, 2.0]),
        "frame": pd.DataFrame({"score": [np.nan, 0.5]}),
        "scalar": np.float64("nan"),
    }
    text = json.dumps(encode(value), allow_nan=False)
    back = cast(dict[str, object], decode(strict(text)))
    assert json.dumps(encode(back), allow_nan=False) == text
    floats = cast(list[float], back["floats"])
    array = cast(npt.NDArray[np.float64], back["array"])
    frame = cast(pd.DataFrame, back["frame"])
    assert np.isnan(floats[0]) and floats[1:] == [np.inf, -np.inf, 1.5]
    assert np.isnan(array[0]) and array.dtype == np.float64
    assert frame["score"].isna().tolist() == [True, False]


def test_files_with_bare_nan_still_load() -> None:
    old = '{"x": NaN, "a": {"__trl__": "array", "dtype": "<f8", "items": [NaN, 1.0]}}'
    back = cast(dict[str, object], decode(json.loads(old)))
    array = cast(npt.NDArray[np.float64], back["a"])
    assert np.isnan(cast(float, back["x"])) and np.isnan(array[0])
    with pytest.raises(DataError, match="not a float"):
        decode({"__trl__": "float", "value": "1e999"})


def test_a_saved_model_is_strict_json(
    generic_fakes: tuple[FakeLLM, FakeLLM], tmp_path: Path
) -> None:
    X = [f"Review {i} is {'good' if i % 2 else 'bad'}." for i in range(40)]
    y = ["good" if i % 2 else "bad" for i in range(40)]
    model = RRF(
        task="t", generator="fake:writer", answerer="fake:judge", n_prompts=3, cv=2
    )
    model.fit(X, y)
    model.threshold_ = float("inf")  # a value strict JSON cannot hold as is
    saved = (model.save(tmp_path / "m") / "model.json").read_text()
    document = cast(dict[str, dict[str, object]], strict(saved))
    assert document["summary"]["threshold"] is None
    assert RRF.load(tmp_path / "m").threshold_ == float("inf")
