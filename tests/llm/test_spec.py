"""Model strings, LLM settings and role checks."""

from __future__ import annotations

import pytest

from think_reason_learn.exceptions import MissingAPIKeyError
from think_reason_learn.llm import LLM, resolve, resolve_models


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in ("OPENAI_API_KEY", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(env, f"{env.lower()}-value")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_openai_model() -> None:
    endpoint = resolve("openai:gpt-5-mini")
    assert (endpoint.provider, endpoint.model_id, endpoint.protocol) == (
        "openai",
        "gpt-5-mini",
        "openai",
    )
    assert endpoint.kind == "reasoning"
    assert endpoint.base_url == "https://api.openai.com/v1"
    assert endpoint.api_key == "openai_api_key-value"


def test_jev_is_a_decision_model_on_system_one() -> None:
    endpoint = resolve("jev")
    assert endpoint.name == "typesafe:jev-latest"
    assert (endpoint.kind, endpoint.protocol) == ("decision", "systemone")


def test_model_ids_keep_slashes_and_colons() -> None:
    assert (
        resolve("openrouter:meta-llama/llama-3.3-70b-instruct").model_id
        == "meta-llama/llama-3.3-70b-instruct"
    )
    local = resolve("ollama:qwen3:8b")
    assert (local.model_id, local.local, local.api_key) == ("qwen3:8b", True, None)


def test_ollama_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    assert resolve("ollama:qwen3:8b").base_url == "http://gpu-box:11434/v1"


def test_any_openai_compatible_server() -> None:
    endpoint = resolve(LLM("Qwen3-32B", base_url="http://gpu-box:8000/v1"))
    assert (endpoint.protocol, endpoint.model_id, endpoint.local) == (
        "openai",
        "Qwen3-32B",
        True,
    )
    with pytest.raises(ValueError, match="has no provider"):
        resolve("Qwen3-32B")


def test_unknown_provider() -> None:
    with pytest.raises(ValueError, match='Unknown model "nope:model"'):
        resolve("nope:model")


def test_missing_key_names_the_variable() -> None:
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        resolve("anthropic:claude-haiku-4-5")
    assert resolve(LLM("anthropic:claude-haiku-4-5", api_key="k")).api_key == "k"


def test_decision_models_route_to_their_protocol() -> None:
    assert resolve("openai:gpt-6-luna").protocol == "openai-decisions"
    routed = resolve(LLM("openrouter:vendor/new-decider", kind="decision"))
    assert (routed.protocol, routed.base_url) == (
        "systemone",
        "https://openrouter.ai/api/v1/systemone",
    )
    azure = resolve(
        LLM(
            "azure:Microsoft-Decision-1",
            base_url="https://r.example/systemone",
            api_key="k",
        )
    )
    assert (azure.protocol, azure.api_key_header) == ("systemone", True)


def test_decision_model_cannot_write() -> None:
    with pytest.raises(
        ValueError, match='generator="typesafe:jev-latest" is a decision model'
    ):
        resolve_models("jev", param="generator", task="generate")
    answerers = resolve_models(
        ("jev", "openai:gpt-5-mini"), param="answerer", task="answer"
    )
    assert [e.kind for e in answerers] == ["decision", "reasoning"]


def test_role_needs_a_model() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        resolve_models((), param="generator", task="generate")


def test_llm_checks_settings() -> None:
    with pytest.raises(ValueError, match="temperature"):
        LLM("openai:gpt-5-mini", temperature=-1.0)
    with pytest.raises(ValueError, match="questions_per_call"):
        LLM("openai:gpt-5-mini", questions_per_call=0)


def test_api_key_is_hidden_from_repr() -> None:
    assert "secret" not in repr(LLM("openai:gpt-5-mini", api_key="secret"))
