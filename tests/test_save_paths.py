"""save() writes to save_path, and a loaded model saves back where it came from."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]


def _pi(**kwargs: Any) -> PolicyInduction:
    return PolicyInduction(
        gen_llmc=CHAT, predict_llmc=CHAT, name="m", confirm_requests=False, **kwargs
    )


def _rrm(**kwargs: Any) -> ReasonedRuleMining:
    return ReasonedRuleMining(reason_llmc=CHAT, name="m", **kwargs)


MODELS: list[tuple[Callable[..., Any], Any, str, str]] = [
    (_pi, PolicyInduction, "policy_induction.json", "policy_induction"),
    (_rrm, ReasonedRuleMining, "reasoned_rule_mining.json", "reasoned_rule_mining"),
]


def _manifests(root: Path, manifest: str) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob(manifest))


@pytest.mark.parametrize(("make", "cls", "manifest", "default_dir"), MODELS)
def test_save_writes_to_save_path(
    tmp_path: Path, make: Any, cls: Any, manifest: str, default_dir: str
) -> None:
    make(save_path=tmp_path / "model").save()
    assert _manifests(tmp_path, manifest) == [Path("model") / manifest]


@pytest.mark.parametrize(("make", "cls", "manifest", "default_dir"), MODELS)
def test_default_save_path_is_not_nested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make: Any,
    cls: Any,
    manifest: str,
    default_dir: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    make().save()
    assert _manifests(tmp_path, manifest) == [Path(default_dir) / "m" / manifest]


@pytest.mark.parametrize(("make", "cls", "manifest", "default_dir"), MODELS)
def test_load_then_save_writes_back_in_place(
    tmp_path: Path, make: Any, cls: Any, manifest: str, default_dir: str
) -> None:
    """Saved under a directory not named after the model, then saved twice more."""
    make().save(tmp_path / "elsewhere")
    loaded = cls.load(tmp_path / "elsewhere")
    assert loaded.save_path == (tmp_path / "elsewhere").resolve()
    loaded.save()
    cls.load(tmp_path / "elsewhere").save()
    assert _manifests(tmp_path, manifest) == [Path("elsewhere") / manifest]


@pytest.mark.parametrize(
    ("make", "training_files"),
    [
        (_pi, ["data.parquet", "policy_predictions.parquet"]),
    ],
)
def test_production_save_removes_training_files_left_in_place(
    tmp_path: Path, make: Any, training_files: list[str]
) -> None:
    """Saving a loaded model in place for production must not keep its data."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in training_files:
        (model_dir / name).write_bytes(b"stale")

    make(save_path=model_dir).save(for_production=True)

    assert not [name for name in training_files if (model_dir / name).exists()]
