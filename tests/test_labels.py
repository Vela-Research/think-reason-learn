"""YES/NO labels are accepted in any case and stored upper-case."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from think_reason_learn.core.exceptions import DataError
from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.policy_induction import PolicyInduction

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]
X = pd.DataFrame({"text": ["a", "b", "c", "d"]})


def _pi(tmp_path: Path) -> PolicyInduction:
    return PolicyInduction(
        gen_llmc=CHAT, predict_llmc=CHAT, save_path=tmp_path, confirm_requests=False
    )


@pytest.mark.parametrize("make", [_pi])
def test_mixed_case_labels_are_accepted(tmp_path: Path, make: object) -> None:
    model = make(tmp_path)  # type: ignore[operator]
    model._set_data(X, ["yes", "No", "YES", "nO"], **_copy_kwarg(model))
    assert list(model._y) == ["YES", "NO", "YES", "NO"]


@pytest.mark.parametrize("make", [_pi])
def test_other_labels_are_rejected(tmp_path: Path, make: object) -> None:
    model = make(tmp_path)  # type: ignore[operator]
    with pytest.raises(DataError, match="YES"):
        model._set_data(X, ["yes", "no", "maybe", "no"], **_copy_kwarg(model))


def _copy_kwarg(model: object) -> dict[str, bool]:
    return {}
