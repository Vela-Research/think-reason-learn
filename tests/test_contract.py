"""Every model keeps the library's contract: scikit-learn, pickling, save and load."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from think_reason_learn.core._classifier import BinaryTRLClassifier, TRLClassifier
from think_reason_learn.exceptions import DataError
from think_reason_learn.testing import FakeLLM, check_classifier


def test_every_model_keeps_the_contract(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    tmp_path: Path,
) -> None:
    (_, make), (X, y) = any_model, founders
    check_classifier(make, X, y, tmp_path)


def test_two_class_models_refuse_more_classes(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
) -> None:
    (_, make), (X, _) = any_model, founders
    if not isinstance(model := make(), BinaryTRLClassifier):
        pytest.skip("takes any number of classes")
    with pytest.raises(DataError, match="two classes"):
        model.fit(X, [str(i % 3) for i in range(len(X))])
    writer, judge = generic_fakes
    assert writer.n_calls == judge.n_calls == 0
