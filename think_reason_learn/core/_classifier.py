"""The base of every LLM-powered classifier: a scikit-learn estimator.

Subclasses list their settings in ``__init__`` (stored untouched, so ``clone``,
``get_params`` and grid search work) and implement ``_afit`` and
``_apredict_proba``. The base gives them sync and async ``fit``/``predict``,
input checks, labels, usage, progress, and ``save``/``load``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from think_reason_learn.core._persistence import Persistable
from think_reason_learn.core._data import LabelInput, TextInput, encode_labels, to_texts
from think_reason_learn.core._prompts import write_instructions
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import (
    Endpoint,
    LLMSession,
    Model,
    Models,
    Usage,
    UsageRecord,
    estimate_generations,
    resolve_models,
    run_async,
    run_sync,
)
from think_reason_learn.llm._estimate import text_tokens
from think_reason_learn.llm._spec import Task

logger = logging.getLogger(__name__)

type Proba = npt.NDArray[np.float64]
type Labels = npt.NDArray[np.intp]
type Work[T] = Callable[[list[str], LLMSession], Awaitable[T]]
"""Work on a model's texts within one session, at prediction time."""


class TRLClassifier(Persistable, ClassifierMixin, BaseEstimator, ABC):
    """Base class of the LLM-powered classifiers.

    ``fit``, ``predict`` and ``predict_proba`` work in scripts and notebooks;
    ``afit``, ``apredict`` and ``apredict_proba`` are their async twins. Every
    model call of a fitted model is recorded in ``usage_``.
    """

    _instructions_purpose: ClassVar[str | None] = None
    """What a model's ``instructions`` guide; ``None`` if it takes none."""

    task: str | None
    instructions: str | None
    verbose: int = 0
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
        return run_sync(self._fit(X, y))

    async def afit(self, X: TextInput, y: LabelInput) -> Self:
        """Fit the model, from async code. See ``fit``."""
        return await run_async(self._fit(X, y))

    def predict_proba(self, X: TextInput) -> Proba:
        """Class probabilities, one row per sample, columns in ``classes_`` order."""
        return self._run(X, "predicting", self._checked_proba)

    async def apredict_proba(self, X: TextInput) -> Proba:
        """Class probabilities, from async code. See ``predict_proba``."""
        check_is_fitted(self)
        return await run_async(self._arun(X, "predicting", self._checked_proba))

    def predict(self, X: TextInput) -> npt.NDArray[np.object_]:
        """The predicted label of each sample."""
        proba = self.predict_proba(X)
        return self.classes_[self._decide(proba)]

    async def apredict(self, X: TextInput) -> npt.NDArray[np.object_]:
        """The predicted label of each sample, from async code."""
        proba = await self.apredict_proba(X)
        return self.classes_[self._decide(proba)]

    def estimate_cost(self, X: TextInput, y: LabelInput | None = None) -> Usage:
        """What ``fit(X, y)`` would spend, or with no ``y`` ``predict(X)``.

        Nothing is sent. Calls are counted from the settings (and, for
        ``predict``, the fitted model) and tokens from the text lengths, then
        priced like real calls. Steps that depend on the answers, such as how
        deep a tree grows or how many refinements are judged, are counted in
        full; replies already in the cache are not subtracted.

        Returns:
            The planned calls, as a ``Usage``; read ``cost_usd`` or
            ``by("role", "model")``.
        """
        texts, _ = to_texts(X)
        if y is None:
            check_is_fitted(self)
            planned = self._predict_plan(texts)
        else:
            classes, codes = _encoded(texts, y)
            self._task()
            self._check_classes(classes)
            planned = self._fit_plan(texts, codes, classes)
        return Usage(r for r in planned if r.calls)

    def write_instructions(self) -> str:
        """Write the instructions now, to read or edit before ``fit``.

        One call to the generator, which turns ``task`` into instructions. The
        result is stored in ``instructions``, so ``fit`` uses it; set
        ``instructions`` yourself to use your own.
        """
        writer = self._models("generator", "generate")
        self.instructions = run_sync(self._written_instructions(LLMSession(), writer))
        return self.instructions

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted once a fit has finished; a failed fit leaves only ``usage_``."""
        return hasattr(self, "classes_")

    def __sklearn_tags__(self) -> Tags:
        """Text input, and answers that can vary between calls."""
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.non_deterministic = True
        return tags

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: npt.NDArray[np.object_]
    ) -> list[UsageRecord]:
        """The calls ``fit`` would make, for ``estimate_cost``."""
        raise NotImplementedError(f"{type(self).__name__} cannot estimate its cost.")

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        """The calls ``predict`` would make, for ``estimate_cost``."""
        raise NotImplementedError(f"{type(self).__name__} cannot estimate its cost.")

    def _instructions_plan(self) -> list[UsageRecord]:
        """The call that writes the instructions, unless they are given."""
        if self.instructions or self._instructions_purpose is None:
            return []
        writer = self._models("generator", "generate")
        return [estimate_generations(writer, text_tokens(self._task()))]

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
    def _session(self, doing: str, usage: Usage | None = None) -> Iterator[LLMSession]:
        """A session for one run, counting answers on a progress bar if ``verbose``."""
        if not self.verbose:
            yield LLMSession(usage=usage)
            return
        from tqdm.auto import tqdm

        with tqdm(desc=f"{type(self).__name__} {doing}", unit=" answers") as bar:

            def advance(count: int) -> None:
                bar.update(count)

            yield LLMSession(usage=usage, progress=advance)

    def _run[T](self, X: TextInput, doing: str, work: Work[T]) -> T:
        """Run ``work`` on the texts of ``X``, recorded in ``usage_``."""
        check_is_fitted(self)
        return run_sync(self._arun(X, doing, work))

    async def _arun[T](self, X: TextInput, doing: str, work: Work[T]) -> T:
        texts = self._texts(X)
        with self._session(doing, self.usage_) as session:
            return await work(texts, session)

    def _texts(self, X: TextInput) -> list[str]:
        """The texts of ``X``, whose columns must be those fitted on."""
        texts, names = to_texts(X)
        expected = getattr(self, "feature_names_in_", None)
        if expected is not None and names is not None and list(expected) != names:
            raise DataError(
                f"X has columns {names}; the model was fitted on {list(expected)}."
            )
        return texts

    async def _fit(self, X: TextInput, y: LabelInput) -> Self:
        texts, names = to_texts(X)
        classes, codes = _encoded(texts, y)
        self._task()
        self._check_classes(classes)
        self._forget_fitted()
        with self._session("fitting") as session:
            self.classes_, self.usage_ = classes, session.usage
            self.n_features_in_ = len(names) if names is not None else 1
            if names is not None:
                self.feature_names_in_ = np.asarray(names, dtype=object)
            try:
                await self._afit(texts, codes, session)
            except BaseException:
                self._forget_fitted(keep="usage_")  # what the failed fit spent
                raise
        return self

    def _check_classes(self, classes: npt.NDArray[np.object_]) -> None:
        """Raise before any call if the model cannot learn ``classes``."""

    def _forget_fitted(self, *, keep: str | None = None) -> None:
        """Drop what an earlier or failed fit learned, so nothing stale remains."""
        for name in [
            n
            for n in vars(self)
            if n.endswith("_") and not n.startswith("_") and n != keep
        ]:
            delattr(self, name)

    async def _checked_proba(self, texts: list[str], session: LLMSession) -> Proba:
        proba = await self._apredict_proba(texts, session)
        if proba.shape != (len(texts), len(self.classes_)):
            raise RuntimeError(
                f"{type(self).__name__} returned probabilities of shape {proba.shape}."
            )
        return proba


class BinaryTRLClassifier(TRLClassifier):
    """Base of the two-class models: a positive class and a decision threshold.

    The positive class is ``pos_label``, by default the second sorted label
    (``"YES"`` of ``"NO"``/``"YES"``). A sample is positive when its positive
    probability reaches ``threshold_``.
    """

    pos_label: str | int | None
    threshold_: float

    def __sklearn_tags__(self) -> Tags:
        """Two classes only."""
        tags = super().__sklearn_tags__()
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_class = False
        return tags

    def _check_classes(self, classes: npt.NDArray[np.object_]) -> None:
        if len(classes) != 2:
            raise DataError(
                f"{type(self).__name__} predicts two classes; y has {len(classes)}."
            )
        positive_index(classes, self.pos_label)

    def _positive(self, y: Labels) -> npt.NDArray[np.bool_]:
        """Which labels are the positive class."""
        return y == self._pos_index()

    def _pos_index(self) -> int:
        return positive_index(self.classes_, self.pos_label)

    def _positive_label(self) -> str:
        """The positive class, as text for prompts."""
        return str(self.classes_[self._pos_index()])

    def _binary_proba(self, positive: npt.NDArray[np.float64]) -> Proba:
        """Class probabilities from the positive class's."""
        proba = np.empty((len(positive), 2))
        proba[:, self._pos_index()] = positive
        proba[:, 1 - self._pos_index()] = 1 - positive
        return proba

    def _decide(self, proba: Proba) -> Labels:
        return binary_decision(proba, self._pos_index(), self.threshold_)


def positive_index(
    classes: npt.NDArray[np.object_], pos_label: str | int | None
) -> int:
    """Index of the positive class: ``pos_label``, or the second class."""
    if pos_label is None:
        return 1
    labels = [str(c) for c in classes]
    if str(pos_label) not in labels:
        raise ValueError(f"pos_label={pos_label!r} is not one of {labels}.")
    return labels.index(str(pos_label))


def binary_decision(proba: Proba, pos: int, threshold: float) -> Labels:
    """Label codes: positive where its probability reaches ``threshold``."""
    positive = proba[:, pos] >= threshold - 1e-9
    return np.where(positive, pos, 1 - pos).astype(np.intp)


def _encoded(texts: list[str], y: LabelInput) -> tuple[npt.NDArray[np.object_], Labels]:
    """The classes and label codes of ``y``, one per text."""
    classes, codes = encode_labels(y)
    if len(codes) != len(texts):
        raise DataError(f"X has {len(texts)} samples and y has {len(codes)} labels.")
    return classes, codes
