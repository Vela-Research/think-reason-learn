"""Saving and loading models: a pickled model plus a readable summary."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Self

import numpy as np

from think_reason_learn.core._sklearn import dump, load
from think_reason_learn.llm import LLM


class Persistable:
    """``save`` and ``load`` for estimators.

    ``save`` writes ``model.joblib`` and a readable ``model.json`` with the
    settings and what the model learned (``_summary``). API keys are never
    saved: ``LLM`` pickles without its key.
    """

    get_params: Callable[..., dict[str, object]]
    """The settings, from scikit-learn's ``BaseEstimator``."""

    def save(self, path: str | PathLike[str]) -> Path:
        """Save the model to a folder: ``model.joblib`` and ``model.json``.

        Returns:
            The folder.
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        dump(self, folder / "model.joblib")
        summary: dict[str, object] = {
            "class": type(self).__name__,
            "params": self.get_params(),
        }
        if (classes := getattr(self, "classes_", None)) is not None:
            summary |= {"classes": classes, **self._summary()}
        (folder / "model.json").write_text(
            json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8"
        )
        return folder

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        """Load a model saved with ``save``.

        .. warning::
            Only load files you trust. ``model.joblib`` is a pickle: loading it
            runs any code it holds.
        """
        model = load(Path(path) / "model.joblib")
        if not isinstance(model, cls):
            raise TypeError(
                f"{path} holds a {type(model).__name__}, not a {cls.__name__}."
            )
        return model

    def _summary(self) -> Mapping[str, object]:
        """What the model learned, for ``model.json``. Subclasses add to it."""
        return {}


def _jsonable(value: object) -> object:
    """A value ``json`` cannot write, as one it can.

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
