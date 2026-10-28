"""Model strings, LLM settings and role checks."""

from __future__ import annotations

import json
from dataclasses import replace

import httpx2
import pytest

from think_reason_learn.exceptions import MissingAPIKeyError
from think_reason_learn.llm import (
    LLM,
    Price,
    _catalog,
    get_config,
    resolve_models,
)
from think_reason_learn.llm._spec import keys_optional, resolve


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
    luna = LLM("openai:gpt-6-luna", api_key="k")
    assert resolve(luna).kind == "reasoning"  # the chat model, unless asked
    decisions = resolve(LLM("openai:gpt-6-luna", api_key="k", kind="decision"))
    assert (decisions.protocol, decisions.model_id) == (
        "openai-decisions",
        "gpt-6-luna",
    )
    azure = resolve(
        LLM(
            "azure:my-decision-deployment",
            base_url="https://r.example/systemone",
            api_key="k",
            kind="decision",
        )
    )
    assert (azure.protocol, azure.api_key_header) == ("systemone", True)


def test_openrouter_decision_models_come_from_its_catalog() -> None:
    listed = resolve(LLM("openrouter:openai/gpt-6-luna-decisions", api_key="k"))
    assert (listed.kind, listed.protocol, listed.base_url) == (
        "decision",
        "systemone",
        "https://openrouter.ai/api/alpha/decisions",
    )
    assert listed.settings.price == Price(input=0.1, output=0.0)
    chat = resolve(LLM("openrouter:openai/gpt-6-luna", api_key="k"))
    assert chat.kind == "reasoning" and chat.settings.price is None
    unlisted = resolve(LLM("openrouter:vendor/brand-new", api_key="k", kind="decision"))
    assert unlisted.protocol == "systemone"


def test_the_catalog_is_read_once_a_day_and_kept_on_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[int] = []
    served = _catalog._download  # the offline catalog from conftest

    def download() -> object:
        downloads.append(1)
        return served()

    monkeypatch.setattr(_catalog, "_download", download)
    for _ in range(3):
        resolve(LLM("openrouter:typesafe/jev-1.14", api_key="k"))
    _catalog._loaded = None  # a new process reads the disk copy
    resolve(LLM("openrouter:typesafe/jev-1.14", api_key="k"))
    assert len(downloads) == 1


def test_an_unreachable_catalog_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline() -> object:
        raise httpx2.ConnectError("offline")

    monkeypatch.setattr(_catalog, "_download", offline)
    assert resolve(LLM("openrouter:typesafe/jev-1.14", api_key="k")).kind == "reasoning"
    kept = resolve(LLM("openrouter:typesafe/jev-1.14", api_key="k", kind="decision"))
    assert kept.protocol == "systemone"


def test_a_price_that_is_not_an_amount_is_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = {
        "id": "openrouter/auto-decisions",
        "architecture": {"output_modalities": ["decisions"]},
        "pricing": {"prompt": "-1", "completion": "-1"},
    }
    monkeypatch.setattr(_catalog, "_download", lambda: {"data": [entry]})
    endpoint = resolve(LLM("openrouter:openrouter/auto-decisions", api_key="k"))
    assert endpoint.kind == "decision" and endpoint.settings.price is None


def test_a_tampered_catalog_file_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve(LLM("openrouter:typesafe/jev-1.14", api_key="k"))  # keeps it on disk
    path = get_config().cache_path.parent / "openrouter-decisions.json"
    stored = json.loads(path.read_text())
    stored["models"]["typesafe/jev-1.14"] = [-1e6, 0]
    path.write_text(json.dumps(stored))
    _catalog._loaded = None
    endpoint = resolve(LLM("openrouter:typesafe/jev-1.14", api_key="k"))
    assert endpoint.settings.price == Price(0.042, 0.0)  # downloaded again


def test_only_a_key_passed_goes_to_a_custom_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-env")
    proxy = "https://proxy.example/v1"
    assert resolve(LLM("openai:gpt-5-mini")).api_key == "sk-env"
    assert resolve(LLM("openai:gpt-5-mini", base_url=proxy)).api_key is None
    assert resolve(LLM("openai:gpt-5-mini", base_url=proxy, api_key="k")).api_key == "k"
    # A base_url written in code or carried in a saved model gets no environment
    # key, Azure included: model.json holds base_url, so a file from someone else
    # would otherwise send your Azure key to their host.
    azure = LLM("azure:my-deployment", base_url="https://me.openai.azure.com/v1")
    assert resolve(azure).api_key is None
    assert resolve(replace(azure, api_key="k")).api_key == "k"
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://me.openai.azure.com/v1")
    assert resolve(LLM("azure:my-deployment")).api_key == "az-env"


def test_a_key_sent_over_plain_http_is_warned_of(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolve(LLM("my-model", base_url="http://models.example.com/v1", api_key="k"))
    resolve(LLM("my-model", base_url="http://localhost:8000/v1", api_key="k"))
    resolve(LLM("my-model", base_url="https://models.example.com/v1", api_key="k"))
    assert caplog.text.count("plain http") == 1 and "models.example.com" in caplog.text


def test_planning_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with keys_optional():
        assert resolve("openai:gpt-5-mini").api_key is None
    with pytest.raises(MissingAPIKeyError):
        resolve("openai:gpt-5-mini")


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


def test_copies_keep_the_key_and_pickles_drop_it() -> None:
    import copy
    import pickle

    model = LLM("openai:gpt-5-mini", api_key="secret")
    assert copy.deepcopy(model).api_key == copy.copy(model).api_key == "secret"
    assert pickle.loads(pickle.dumps(model)).api_key is None


def test_custom_servers_keep_colons_and_only_nearby_ones_are_local() -> None:
    nearby = resolve(LLM("qwen2.5:7b", base_url="http://localhost:11434/v1"))
    assert (nearby.provider, nearby.model_id, nearby.local) == (
        "custom",
        "qwen2.5:7b",
        True,
    )
    assert resolve(LLM("m", base_url="http://10.0.0.5:8000/v1")).local
    assert not resolve(LLM("m", base_url="https://gateway.example.com/v1")).local


def test_only_openrouter_models_read_its_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def offline() -> object:
        raise AssertionError("the catalog was read")  # no internet needed

    monkeypatch.setattr(_catalog, "_download", offline)
    for model in (
        LLM("ollama:qwen3:8b"),
        LLM("qwen2.5:7b", base_url="http://localhost:11434/v1"),
        LLM("openai:gpt-5-mini"),
        "jev",
    ):
        resolve(model)
