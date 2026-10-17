"""Saving and loading models as JSON, readable and safe to load.

``save`` writes ``model.json``: the save format, the library version, the
settings, everything the model learned, and a readable summary. ``load``
rebuilds the model from it without running code from the file: values are plain
JSON, and the only objects it builds are this library's data classes, numpy
arrays and pandas frames. Network weights go in a file of their own.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import ClassVar, Final, Self

import numpy as np
import pandas as pd

from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import LLM, Usage, UsageRecord
from think_reason_learn.llm._cache import JSON

logger = logging.getLogger(__name__)

FORMAT: Final = 1
"""The save format; ``load`` reads only this one."""

_TAG: Final = "__trl__"
"""Marks a JSON object that stands for a tuple, array, frame or data class."""

_OLD_FILES: Final = (
    "rrf.json",
    "gptree.json",
    "policy_induction.json",
    "reasoned_rule_mining.json",
    "meta.json",
)
"""What a folder saved by 0.2 holds instead of ``model.json``."""


class Persistable:
    """``save`` and ``load`` for estimators, as one ``model.json``.

    API keys are never saved. Attributes named in ``_saved_apart`` are left out
    of ``model.json``, for ``_save_apart`` and ``_load_apart`` to handle.
    """

    get_params: Callable[..., dict[str, object]]
    """The settings, from scikit-learn's ``BaseEstimator``."""

    _saved_apart: ClassVar[tuple[str, ...]] = ()

    def save(self, path: str | PathLike[str]) -> Path:
        """Save the model to a folder, as ``model.json``.

        Returns:
            The folder.
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        fitted = {
            name: value
            for name, value in vars(self).items()
            if _is_fitted(name) and name not in self._saved_apart
        }
        document: dict[str, JSON] = {
            "format": FORMAT,
            "library": _library_version(),
            "class": type(self).__name__,
            "params": encode(self.get_params(deep=False)),
            "fitted": encode(fitted),
            "summary": json.loads(json.dumps(self._readable(), default=_jsonable)),
        }
        (folder / "model.json").write_text(
            json.dumps(document, indent=2), encoding="utf-8"
        )
        self._save_apart(folder)
        return folder

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        """Load a model saved with ``save``, by this version or another of format 1.

        Loading runs no code from the file. The file does name the models the
        model calls and their servers: check ``get_params()`` before predicting
        with a file from someone else, since your samples go to them.

        Raises:
            DataError: The folder was saved by 0.2, in another format, or the
                file is damaged.
            TypeError: It holds another kind of model.
        """
        folder = Path(path)
        document = _document(folder, cls.__name__)
        try:
            model = construct(cls, _named(decode(document["params"])))
            fitted = _named(decode(document["fitted"]))
        except (KeyError, TypeError, ValueError) as error:
            raise DataError(f"model.json is damaged: {error!r}") from error
        for name, value in fitted.items():
            if not _is_fitted(name):
                raise DataError(f"model.json is damaged: {name} is not learned.")
            setattr(model, name, value)
        model._load_apart(folder)
        return model

    def _summary(self) -> Mapping[str, object]:
        """What the model learned, for people reading ``model.json``."""
        return {}

    def _save_apart(self, folder: Path) -> None:
        """Save the attributes in ``_saved_apart``; there are none by default."""

    def _load_apart(self, folder: Path) -> None:
        """Load what ``_save_apart`` saved; nothing by default."""

    def _readable(self) -> Mapping[str, object]:
        if (classes := getattr(self, "classes_", None)) is None:
            return {}
        return {"classes": classes, **self._summary()}


def encode(value: object) -> JSON:
    """``value`` as JSON; tuples, arrays, frames and data classes come back as such.

    Raises:
        TypeError: ``value`` holds something that cannot be saved.
    """
    match value:
        case None | bool() | int() | float() | str():
            return value
        case np.generic():
            return encode(value.item())  # pyright: ignore[reportUnknownArgumentType]  # a Python scalar
        case list():
            return _encode_items(value)  # pyright: ignore[reportUnknownArgumentType]  # any list
        case tuple():
            return {_TAG: "tuple", "items": _encode_items(value)}  # pyright: ignore[reportUnknownArgumentType]  # any tuple
        case dict():
            return _encode_dict(value)  # pyright: ignore[reportUnknownArgumentType]  # any dict
        case np.ndarray():
            values: object = value.tolist()
            return {_TAG: "array", "dtype": value.dtype.str, "items": encode(values)}
        case pd.DataFrame():
            split = value.to_dict(orient="split")
            return {_TAG: "frame", **{k: encode(v) for k, v in split.items()}}
        case Usage():
            return _encode_usage(value)
        case _ if _is_own_data(value):
            return _encode_data(value)
        case _:
            raise TypeError(f"A {type(value).__name__} cannot be saved.")


def decode(value: JSON) -> object:
    """A value written by ``encode``."""
    match value:
        case list():
            return [decode(item) for item in value]
        case dict() if _TAG in value:
            return _decode_tagged(value)
        case dict():
            return {key: decode(item) for key, item in value.items()}
        case _:
            return value


def _is_fitted(name: str) -> bool:
    """A fitted attribute, by scikit-learn's convention: ``classes_``, not ``_x``."""
    return name.endswith("_") and not name.startswith("_")


def _library_version() -> str:
    from think_reason_learn import __version__  # the package imports this module

    return __version__


def _document(folder: Path, name: str) -> dict[str, JSON]:
    """``model.json`` of ``folder``, checked to be a ``name`` this version reads."""
    file = folder / "model.json"
    if not file.exists():
        if any((folder / old).exists() for old in _OLD_FILES):
            raise DataError(
                f"{folder} was saved by think-reason-learn 0.2, which this version "
                "cannot load: fit the model again. The migration guide shows how "
                "to keep 0.2's questions or policy."
            )
        raise FileNotFoundError(f"No model.json in {folder}.")
    document: JSON = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise DataError(
            f"{file} is not in save format {FORMAT}, which this version reads: load "
            "it with the version that saved it, or fit the model again."
        )
    if document.get("class") != name:
        raise TypeError(f"{folder} holds a {document.get('class')}, not a {name}.")
    if (saved := document.get("library")) != (current := _library_version()):
        logger.warning(
            f"{folder} was saved by think-reason-learn {saved}; this is {current}. "
            "Check that it predicts as before."
        )
    return document


def _named(value: object) -> dict[str, object]:
    """``value`` as keyword arguments, or a ``DataError``."""
    if not isinstance(value, dict):
        raise DataError("model.json is damaged: settings are not a mapping.")
    return {str(key): item for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # checked to be a dict


def _encode_items(values: Iterable[object]) -> list[JSON]:
    return [encode(item) for item in values]


def _encode_dict(value: dict[object, object]) -> JSON:
    if _TAG not in value and all(isinstance(key, str) for key in value):
        return {str(key): encode(item) for key, item in value.items()}
    pairs: list[JSON] = [[encode(key), encode(item)] for key, item in value.items()]
    return {_TAG: "pairs", "items": pairs}


def _is_own_data(value: object) -> bool:
    """A data class instance from this library (its only objects ``load`` builds)."""
    module = type(value).__module__
    return (
        dataclasses.is_dataclass(value)
        and not isinstance(value, type)
        and (module == "think_reason_learn" or module.startswith("think_reason_learn."))
    )


def _encode_data(value: object) -> JSON:
    cls = type(value)
    fields = {
        field.name: encode(getattr(value, field.name))
        for field in dataclasses.fields(value)  # pyright: ignore[reportArgumentType]  # checked to be a data class
        if field.name != "api_key"  # never saved, whatever holds it
    }
    return {
        _TAG: "data",
        "class": f"{cls.__module__}:{cls.__qualname__}",
        "fields": fields,
    }


def _decode_tagged(value: dict[str, JSON]) -> object:
    match value[_TAG]:
        case "tuple":
            return tuple(_items(value))
        case "pairs":
            return {_key(pair): _value(pair) for pair in _items(value)}
        case "array":
            return np.array(_items(value), dtype=_dtype(value["dtype"]))
        case "frame":
            return pd.DataFrame(  # pyright: ignore[reportCallIssue]  # rows and labels from the file
                decode(value["data"]),  # pyright: ignore[reportArgumentType]  # rows from the file
                index=_listed(value["index"]),
                columns=_listed(value["columns"]),
            )
        case "usage":
            names = [str(name) for name in _listed(decode(value["fields"]))]
            rows = _listed(decode(value["rows"]))
            return Usage(
                construct(UsageRecord, dict(zip(names, _listed(row), strict=True)))
                for row in rows
            )
        case "data":
            return _decode_data(str(value["class"]), value["fields"])
        case tag:
            raise DataError(f"model.json is damaged: unknown value type {tag!r}.")


def _decode_data(path: str, fields: JSON) -> object:
    """An instance of this library's data class at ``path`` (``module:Class``)."""
    module, _, name = path.partition(":")
    own = module == "think_reason_learn" or module.startswith("think_reason_learn.")
    if not own or "." in name:
        raise DataError(f"model.json names {path!r}, not one of this library's.")
    cls: object = getattr(importlib.import_module(module), name, None)
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise DataError(f"model.json names {path!r}, not one of this library's.")
    return construct(cls, _named(decode(fields)))


def _encode_usage(usage: Usage) -> JSON:
    """Usage as one row per call, in columns: a large run has many calls."""
    names = [field.name for field in dataclasses.fields(UsageRecord)]
    rows: list[JSON] = [
        [encode(getattr(record, name)) for name in names] for record in usage
    ]
    return {_TAG: "usage", "fields": list[JSON](names), "rows": rows}


def _dtype(name: JSON) -> np.dtype[np.generic]:
    """An array's dtype: numbers, booleans, text or Python objects only."""
    dtype = np.dtype(str(name))
    if dtype.kind not in "biufUO":
        raise DataError(f"model.json is damaged: arrays of {dtype} are not saved.")
    return dtype


def _items(value: dict[str, JSON]) -> list[object]:
    return _listed(decode(value["items"]))


def _listed(value: object) -> list[object]:
    if not isinstance(value, list):
        raise DataError("model.json is damaged: a list was expected.")
    return list(value)  # pyright: ignore[reportUnknownArgumentType]  # checked to be a list


def _key(pair: object) -> object:
    return _listed(pair)[0]


def _value(pair: object) -> object:
    return _listed(pair)[1]


def construct[T](cls: type[T], params: Mapping[str, object]) -> T:
    """``cls`` built from keyword arguments read from a file."""
    return cls(**params)


def _jsonable(value: object) -> object:
    """A value ``json`` cannot write, as one it can, for the readable summary.

    LLM settings lose their key, numpy values become lists or numbers, and
    anything else becomes text.
    """
    if isinstance(value, LLM):
        fields = {f: getattr(value, f) for f in value.__dataclass_fields__}
        return {k: v for k, v in fields.items() if k != "api_key" and v is not None}
    if isinstance(value, np.ndarray | np.generic):
        listed: object = np.asarray(value, dtype=object).tolist()
        return listed
    return str(value)
