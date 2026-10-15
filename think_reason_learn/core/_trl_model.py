from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from os import PathLike
from pathlib import Path
from typing import ClassVar, Self
from uuid import uuid4

import pandas as pd

from think_reason_learn.core.llms import TokenCounter

logger = logging.getLogger(__name__)


class TRLModel(ABC):
    """Base class of the LLM-powered models.

    Random Rule Forest, GPTree, Policy Induction and Reasoned Rule Mining
    subclass it, so each one is given its task with ``set_task``, trained with
    ``fit``, predicts with ``predict``, and is saved to and loaded from a
    directory with ``save`` and ``load``. ``set_task``, ``fit`` and ``predict``
    are async.

    VerifiableRL is not a ``TRLModel``: it makes no LLM calls, and its ``fit``
    and ``predict`` are synchronous.

    A subclass sets ``_default_save_dir``: ``save_path`` defaults to
    ``./<_default_save_dir>/<name>``.
    """

    _default_save_dir: ClassVar[str]

    name: str
    save_path: Path
    _token_counter: TokenCounter
    _task_description: str | None

    @abstractmethod
    async def set_task(self, task_description: str) -> str | None:
        """Describe the classification task to the model.

        Args:
            task_description: Description of the classification task.

        Returns:
            The generation instructions template, for models that use one.
        """

    @abstractmethod
    async def fit(self, X: pd.DataFrame, y: Sequence[str]) -> Self:
        """Fit the model to labelled samples.

        Args:
            X: Training samples, one per row.
            y: Training labels, one per row of ``X``.

        Returns:
            The fitted model.
        """

    @abstractmethod
    def predict(self, samples: pd.DataFrame) -> AsyncIterator[tuple[object, ...]]:
        """Predict samples, yielding records as they are ready.

        Args:
            samples: Samples to predict, one per row.

        Yields:
            Tuples starting with the sample's index. The rest of the tuple
            depends on the model.
        """

    @abstractmethod
    def save(self, dir_path: str | PathLike[str] | None = None) -> None:
        """Save the model to a directory.

        Args:
            dir_path: Directory to save to. Defaults to ``save_path``.
        """

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str | PathLike[str]) -> Self:
        """Load a model saved with ``save``.

        Args:
            dir_path: Directory the model was saved to.

        Returns:
            The loaded model.
        """

    @property
    def token_usage(self) -> TokenCounter:
        """Tokens used by the model's LLM calls."""
        return self._token_counter

    @property
    def task_description(self) -> str | None:
        """Description of the classification task."""
        return self._task_description

    def __repr__(self) -> str:
        """Return the class and the model's name."""
        return f"{type(self).__name__}(name={self.name})"

    def _get_name(self, name: str | None) -> str:
        if name is None:
            name = str(uuid4()).replace("-", "_")
            logger.debug(f"No name provided. Assigned name: {name}")
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError("Name must be only alphanumeric and underscores")
        return name

    def _set_save_path(self, save_path: str | PathLike[str] | None) -> Path:
        if save_path is None:
            return (Path.cwd() / self._default_save_dir / self.name).resolve()
        path = Path(save_path).resolve()
        if path.is_file():
            raise ValueError("Please provide a directory, not a file.")
        return path


__all__ = ["TRLModel"]
