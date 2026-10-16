"""Every model keeps the library's contract: scikit-learn, pickling, save and load."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.testing import FakeLLM, check_classifier


def test_every_model_keeps_the_contract(
    any_model: tuple[str, Callable[[], TRLClassifier]],
    generic_fakes: tuple[FakeLLM, FakeLLM],
    founders: tuple[list[str], list[str]],
    tmp_path: Path,
) -> None:
    (_, make), (X, y) = any_model, founders
    check_classifier(make, X, y, tmp_path)
