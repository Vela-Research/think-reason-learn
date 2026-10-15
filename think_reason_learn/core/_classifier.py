"""The base of every LLM-powered classifier: a scikit-learn estimator.

Subclasses list their settings in ``__init__`` (stored untouched, so ``clone``,
``get_params`` and grid search work) and implement ``_afit`` and
``_apredict_proba``. The base gives them sync and async ``fit``/``predict``,
input checks, labels, usage, progress, and ``save``/``load``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import ClassVar, Self

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from think_reason_learn.core._data import LabelInput, TextInput, encode_labels, to_texts
from think_reason_learn.core._prompts import write_instructions
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import (
    LLM,
    Endpoint,
    LLMSession,
    Model,
    Models,
    Usage,
    resolve_models,
    run_async,
    run_sync,
)
from think_reason_learn.llm._spec import Task

logger = logging.getLogger(__name__)

type Proba = npt.NDArray[np.float64]
type Labels = npt.NDArray[np.intp]


class TRLClassifier(ClassifierMixin, BaseEstimator, ABC):
    """Base class of the LLM-powered classifiers.

    ``fit``, ``predict`` and ``predict_proba`` work in scripts and notebooks;
    ``afit``, ``apredict`` and ``apredict_proba`` are their async twins. Every
    model call of a fitted model is recorded in ``usage_``.
    """

    _instructions_purpose: ClassVar[str | None] = None
    """What a model's ``instructions`` guide; ``None`` if it takes none."""

    task: str | None
    instructions: str | None
    classes_: npt.NDArray[np.object_]
    n_features_in_: int
    feature_names_in_: npt.NDArray[np.object_]
    usage_: Usage

    @abstractmethod
    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        """Learn from ``texts`` and label codes ``y`` (indices into ``classes_``)."""

    @abstractmethod
    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        """Class probabilities, one row per text, columns in ``classes_`` order."""

    def fit(self, X: TextInput, y: LabelInput) -> Self:
        """Fit the model.

        Args:
            X: Samples: a DataFrame, or one text per sample.
            y: One label per sample, any two or more classes.

        Returns:
            The fitted model.
        """
        return run_sync(self._fit(X, y, LLMSession()))

    async def afit(self, X: TextInput, y: LabelInput) -> Self:
        """Fit the model, from async code. See ``fit``."""
        return await run_async(self._fit(X, y, LLMSession()))

    def predict_proba(self, X: TextInput) -> Proba:
        """Class probabilities, one row per sample, columns in ``classes_`` order."""
        return run_sync(self._proba(X, self._session()))

    async def apredict_proba(self, X: TextInput) -> Proba:
        """Class probabilities, from async code. See ``predict_proba``."""
        return await run_async(self._proba(X, self._session()))

    def predict(self, X: TextInput) -> npt.NDArray[np.object_]:
        """The predicted label of each sample."""
        proba = self.predict_proba(X)
        return self.classes_[self._decide(proba)]

    async def apredict(self, X: TextInput) -> npt.NDArray[np.object_]:
        """The predicted label of each sample, from async code."""
        proba = await self.apredict_proba(X)
        return self.classes_[self._decide(proba)]

    def write_instructions(self) -> str:
        """Write the instructions now, to read or edit before ``fit``.

        One call to the generator, which turns ``task`` into instructions. The
        result is stored in ``instructions``, so ``fit`` uses it; set
        ``instructions`` yourself to use your own.
        """
        writer = self._models("generator", "generate")
        self.instructions = run_sync(self._written_instructions(LLMSession(), writer))
        return self.instructions

    def save(self, path: str | PathLike[str]) -> Path:
        """Save the model to a folder: ``model.joblib`` and a readable ``model.json``.

        API keys are never saved.

        Returns:
            The folder.
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, folder / "model.joblib")
        summary = {"class": type(self).__name__, "params": _jsonable(self.get_params())}
        if hasattr(self, "classes_"):
            summary |= {"classes": _jsonable(list(self.classes_)), **self._summary()}
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

    def __sklearn_tags__(self) -> Tags:
        """Text input, and answers that can vary between calls."""
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.non_deterministic = True
        return tags

    def _summary(self) -> Mapping[str, object]:
        """What the model learned, for ``model.json``. Subclasses add to it."""
        return {}

    def _decide(self, proba: Proba) -> Labels:
        """Label codes from probabilities: the most likely class."""
        return np.argmax(proba, axis=1).astype(np.intp)

    async def _instructions(
        self, session: LLMSession, writer: tuple[Endpoint, ...]
    ) -> str:
        """``instructions`` if set, else written from ``task`` by the generator."""
        if self.instructions:
            return self.instructions
        return await self._written_instructions(session, writer)

    async def _written_instructions(
        self, session: LLMSession, writer: tuple[Endpoint, ...]
    ) -> str:
        if self._instructions_purpose is None:
            raise NotImplementedError(f"{type(self).__name__} takes no instructions.")
        return await write_instructions(
            session, writer, self._task(), self._instructions_purpose
        )

    def _task(self) -> str:
        """The task description, which every model needs."""
        if not self.task or not self.task.strip():
            raise ValueError(
                f"{type(self).__name__} needs task=..., what to predict in plain words."
            )
        return self.task

    def _models(self, param: str, task: Task) -> tuple[Endpoint, ...]:
        """The models of a role setting, checked for its task."""
        models: Models | list[Model] | None = getattr(self, param)
        if models is None:
            raise ValueError(
                f"{type(self).__name__} needs {param}=..., e.g. 'openai:gpt-5-mini'."
            )
        return resolve_models(models, param=param, task=task)

    @contextmanager
    def _progress(
        self, total: int, description: str
    ) -> Iterator[Callable[[int], None] | None]:
        """A progress callback when ``verbose`` is set, else ``None``."""
        if not getattr(self, "verbose", 0):
            yield None
            return
        from tqdm.auto import tqdm

        with tqdm(total=total, desc=description, unit="answer") as bar:

            def advance(count: int) -> None:
                bar.update(count)

            yield advance

    def _session(self) -> LLMSession:
        check_is_fitted(self)
        return LLMSession(usage=self.usage_)

    async def _fit(self, X: TextInput, y: LabelInput, session: LLMSession) -> Self:
        texts, names = to_texts(X)
        classes, codes = encode_labels(y)
        if len(codes) != len(texts):
            raise DataError(
                f"X has {len(texts)} samples and y has {len(codes)} labels."
            )
        self.classes_, self.usage_ = classes, session.usage
        self.n_features_in_ = len(names) if names is not None else 1
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        await self._afit(texts, codes, session)
        return self

    async def _proba(self, X: TextInput, session: LLMSession) -> Proba:
        texts, names = to_texts(X)
        expected = getattr(self, "feature_names_in_", None)
        if expected is not None and names is not None and list(expected) != names:
            raise DataError(
                f"X has columns {names}; the model was fitted on {list(expected)}."
            )
        proba = await self._apredict_proba(texts, session)
        if proba.shape != (len(texts), len(self.classes_)):
            raise RuntimeError(
                f"{type(self).__name__} returned probabilities of shape {proba.shape}."
            )
        return proba


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
