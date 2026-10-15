"""PolicyInduction's weight fit across scikit-learn versions (offline)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.policy_induction import PolicyInduction

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]


def _fitted(tmp_path: Path, penalty: str) -> PolicyInduction:
    """Fit weights on one informative policy and five noise policies."""
    rng = np.random.default_rng(0)
    n = 120
    y = np.where(np.arange(n) % 2 == 0, "YES", "NO")
    informative = np.where(rng.random(n) < 0.9, y, np.where(y == "YES", "NO", "YES"))
    columns: list[Any] = [informative] + [
        np.where(rng.random(n) < 0.5, "YES", "NO") for _ in range(5)
    ]

    pi = PolicyInduction(
        gen_llmc=CHAT,
        predict_llmc=CHAT,
        config={"penalty": penalty, "Cs": (0.1,), "cv_folds": 3},
        save_path=tmp_path,
        name="weights",
        confirm_requests=False,
    )
    pi._set_data(pd.DataFrame({"text": [f"sample {i}" for i in range(n)]}), list(y))
    index = pi._X.index  # type: ignore[union-attr]
    pi._policy_memory = pd.DataFrame(
        {
            "policy": [f"policy {j}" for j in range(len(columns))],
            "predictions": [pd.Series(col, index=index) for col in columns],
        }
    )
    pi._fit_weights()
    return pi


def test_penalty_kwargs_follow_the_sklearn_version() -> None:
    from think_reason_learn.policy_induction._policy_induction import (
        _penalty_kwargs,
    )

    assert _penalty_kwargs("l1", "1.7.2") == {"penalty": "l1"}
    assert _penalty_kwargs("l2", "1.7.2") == {"penalty": "l2"}
    assert _penalty_kwargs("l1", "1.8.0") == {"l1_ratio": 1.0}
    assert _penalty_kwargs("l2", "1.9.1") == {"l1_ratio": 0.0}
    assert _penalty_kwargs("l1", "1.10.0rc1") == {"l1_ratio": 1.0}


@pytest.mark.parametrize("version", ["1.7.2", "1.9.1"])
@pytest.mark.parametrize("penalty", ["L1", "elasticnet", None])
def test_an_unknown_penalty_is_refused(penalty: Any, version: str) -> None:
    from think_reason_learn.policy_induction._policy_induction import (
        _penalty_kwargs,
    )

    with pytest.raises(ValueError, match="penalty must be 'l1' or 'l2'"):
        _penalty_kwargs(penalty, version)


def test_l1_zeroes_noise_policies_and_l2_does_not(tmp_path: Path) -> None:
    l1 = _fitted(tmp_path, "l1").lr.coef_[0]
    l2 = _fitted(tmp_path, "l2").lr.coef_[0]

    assert l1[0] != 0
    assert (l1[1:] == 0).any()
    assert (l2 != 0).all()


def test_weight_fit_raises_no_sklearn_deprecation(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("error", category=UserWarning, module="sklearn")
        _fitted(tmp_path, "l1")
        _fitted(tmp_path, "l2")
