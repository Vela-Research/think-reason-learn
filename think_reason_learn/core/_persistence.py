"""Saving and loading models: a pickled model plus a readable summary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Self

import joblib
import numpy as np

from think_reason_learn.llm import LLM


class Persistable:
    """``save`` and ``load`` for estimators.

    ``save`` writes ``model.joblib`` and a readable ``model.json`` with the
    settings and what the model learned (``_summary``). API keys are never
    saved: ``LLM`` pickles without its key.
    """

    def save(self, path: str | PathLike[str]) -> Path:
        """Save the model to a folder: ``model.joblib`` and ``model.json``.

        Returns:
            The folder.
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, folder / "model.joblib")
        summary: dict[str, object] = {"class": type(self).__name__}
        if (get_params := getattr(self, "get_params", None)) is not None:
            summary["params"] = _jsonable(get_params())
        if (classes := getattr(self, "classes_", None)) is not None:
            summary |= {"classes": _jsonable(list(classes)), **self._summary()}
        (folder / "model.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return folder

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        """Load a model saved with ``save``. Only load files you trust."""
        model = joblib.load(Path(path) / "model.joblib")
        if not isinstance(model, cls):
            raise TypeError(
                f"{path} holds a {type(model).__name__}, not a {cls.__name__}."
            )
        return model

    def _summary(self) -> Mapping[str, object]:
        """What the model learned, for ``model.json``. Subclasses add to it."""
        return {}


def _jsonable(value: object) -> object:
    """``value`` as JSON-ready data; LLM settings without their key."""
    if isinstance(value, LLM):
        fields = {
            f: getattr(value, f) for f in value.__dataclass_fields__ if f != "api_key"
        }
        return {k: _jsonable(v) for k, v in fields.items() if v is not None}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
