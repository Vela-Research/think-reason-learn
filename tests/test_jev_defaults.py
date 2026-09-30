"""Jev is the default answerer; a missing TYPESAFE_API_KEY fails at construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from think_reason_learn.core._config import settings
from think_reason_learn.core.exceptions import MissingAPIKeyError
from think_reason_learn.core.llms import JevChoice, LLMChoice, OpenAIChoice
from think_reason_learn.gptree import GPTree
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.rrf import RRF

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]


def _rrf(tmp_path: Path, **kwargs: Any) -> RRF:
    return RRF(qgen_llmc=CHAT, save_path=tmp_path, **kwargs)


def _pi(tmp_path: Path, **kwargs: Any) -> PolicyInduction:
    return PolicyInduction(
        gen_llmc=CHAT, save_path=tmp_path, confirm_requests=False, **kwargs
    )


def _tree(tmp_path: Path, **kwargs: Any) -> GPTree:
    return GPTree(
        qgen_llmc=CHAT,
        critic_llmc=CHAT,
        qgen_instr_llmc=CHAT,
        save_path=tmp_path,
        **kwargs,
    )


METHODS: list[tuple[str, Callable[..., Any], str, str]] = [
    ("RRF", _rrf, "qanswer_llmc", "qanswer_llmc"),
    ("PolicyInduction", _pi, "predict_llmc", "predict_llmc"),
    ("GPTree", _tree, "qanswer_llmc", "qanswer_llmc"),
]


@pytest.fixture
def no_typesafe_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "")


@pytest.mark.parametrize("name,make,param,attr", METHODS)
def test_jev_answers_by_default(
    tmp_path: Path,
    typesafe_key: None,
    name: str,
    make: Callable[..., Any],
    param: str,
    attr: str,
) -> None:
    method = make(tmp_path)

    assert getattr(method, attr) == [JevChoice()]


@pytest.mark.parametrize("name,make,param,attr", METHODS)
def test_missing_key_raises_naming_the_key_and_the_way_out(
    tmp_path: Path,
    no_typesafe_key: None,
    name: str,
    make: Callable[..., Any],
    param: str,
    attr: str,
) -> None:
    with pytest.raises(MissingAPIKeyError) as info:
        make(tmp_path)

    message = str(info.value)
    assert "TYPESAFE_API_KEY" in message
    assert f"{name}(..., {param}=[OpenAIChoice(" in message
    assert isinstance(info.value, ValueError)


@pytest.mark.parametrize("name,make,param,attr", METHODS)
def test_jev_with_a_chat_fallback_needs_the_key(
    tmp_path: Path,
    no_typesafe_key: None,
    name: str,
    make: Callable[..., Any],
    param: str,
    attr: str,
) -> None:
    with pytest.raises(MissingAPIKeyError):
        make(tmp_path, **{param: [JevChoice(), *CHAT]})


@pytest.mark.parametrize("name,make,param,attr", METHODS)
def test_jev_after_a_chat_model_is_rejected(
    tmp_path: Path,
    typesafe_key: None,
    name: str,
    make: Callable[..., Any],
    param: str,
    attr: str,
) -> None:
    with pytest.raises(ValueError, match=f"JevChoice first in {param}"):
        make(tmp_path, **{param: [*CHAT, JevChoice()]})


@pytest.mark.parametrize("name,make,param,attr", METHODS)
def test_chat_answerer_needs_no_typesafe_key(
    tmp_path: Path,
    no_typesafe_key: None,
    name: str,
    make: Callable[..., Any],
    param: str,
    attr: str,
) -> None:
    method = make(tmp_path, **{param: CHAT})

    assert getattr(method, attr) == CHAT
