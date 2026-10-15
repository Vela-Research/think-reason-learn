"""Model specs: ``"provider:model"`` strings, ``LLM`` settings, the provider table."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass, field
from typing import Final, Literal

from think_reason_learn.exceptions import MissingAPIKeyError

from ._usage import Price

type ModelName = str
"""A model as ``"provider:model"``, e.g. ``"openai:gpt-5-mini"``, or ``"jev"``."""

type ModelKind = Literal["reasoning", "decision"]
"""Reasoning models write text and answer questions; decision models only answer."""

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

type Protocol = Literal["openai", "anthropic", "systemone", "openai-decisions", "fake"]

type Task = Literal["generate", "answer"]
"""What a role asks of its models: writing text, or answering about samples."""


@dataclass(frozen=True, slots=True)
class LLM:
    """A model and the settings that change how it answers.

    Args:
        model: ``"provider:model"``, e.g. ``"anthropic:claude-sonnet-5-5"``. With
            ``base_url``, a bare model id calls any OpenAI-compatible server.
        temperature: Sampling temperature. ``None`` uses the model's default. A
            model that rejects a temperature is called without one.
        reasoning_effort: Reasoning effort for models that support it.
        max_output_tokens: Cap on output tokens per call.
        service_tier: Provider service tier, e.g. ``"flex"`` for half price on
            OpenAI and Gemini at slower speed.
        questions_per_call: Questions a reasoning model answers per call about
            one sample. Decision models take every question in one call.
        base_url: Endpoint to call instead of the provider's.
        api_key: Key to use instead of the provider's environment variable.
        kind: ``"decision"`` for a decision model the library does not know yet.
        price: Price to use for cost reports and budgets.
    """

    model: ModelName
    temperature: float | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_output_tokens: int | None = None
    service_tier: str | None = None
    questions_per_call: int = 1
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    kind: ModelKind | None = None
    price: Price | None = None

    def __post_init__(self) -> None:
        """Check the settings."""
        if not self.model or not self.model.strip():
            raise ValueError("LLM(model=...) needs a model, e.g. 'openai:gpt-5-mini'.")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}.")
        if self.questions_per_call < 1:
            raise ValueError(
                f"questions_per_call must be >= 1, got {self.questions_per_call}."
            )

    def __getstate__(self) -> tuple[object, ...]:
        """Pickle without the API key: saved models never hold secrets."""
        return tuple(
            None if name == "api_key" else getattr(self, name)
            for name in self.__dataclass_fields__
        )


type Model = ModelName | LLM
"""A model string, or an ``LLM`` with settings."""

type Models = Model | tuple[Model, ...]
"""One model, or a tuple of models tried in order (fallbacks)."""


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A resolved model: where and how to call it."""

    name: str
    provider: str
    model_id: str
    protocol: Protocol
    kind: ModelKind
    base_url: str | None
    api_key: str | None = field(repr=False)
    settings: LLM
    local: bool = False
    json_object_only: bool = False
    api_key_header: bool = False

    @property
    def client_key(self) -> str:
        """Identity of the endpoint and account, for API clients."""
        account = hashlib.sha256((self.api_key or "").encode()).hexdigest()[:12]
        return f"{self.protocol}|{self.base_url}|{account}"

    @property
    def key(self) -> str:
        """Identity of the model on this endpoint and account, for limits."""
        return f"{self.client_key}|{self.model_id}"


def resolve(model: Model) -> Endpoint:
    """Resolve a model string or ``LLM`` to an ``Endpoint``.

    Raises:
        ValueError: The model string or its provider is not valid.
        MissingAPIKeyError: The provider needs a key that is not set.
    """
    spec = model if isinstance(model, LLM) else LLM(model)
    name = _ALIASES.get(spec.model, spec.model)
    provider_name, has_provider, model_id = name.partition(":")
    if not has_provider:
        return _custom_endpoint(spec, name)
    provider = _PROVIDERS.get(provider_name)
    if provider is None or not model_id:
        raise ValueError(
            f'Unknown model "{name}". Write it as "provider:model" with a provider '
            f"in {sorted(_PROVIDERS)}, or pass base_url= for an OpenAI-compatible "
            "server."
        )
    kind = spec.kind or ("decision" if _is_decision(name) else "reasoning")
    protocol, base_url = _route(provider, provider_name, kind)
    return Endpoint(
        name=name,
        provider=provider_name,
        model_id=model_id,
        protocol=protocol,
        kind=kind,
        base_url=_base_url(spec, provider, base_url, name),
        api_key=_api_key(spec, provider, name),
        settings=spec,
        local=provider.local,
        json_object_only=provider.json_object_only,
        api_key_header=provider.api_key_header,
    )


def resolve_models(
    models: Models | list[Model], *, param: str, task: Task
) -> tuple[Endpoint, ...]:
    """Resolve a role's models in fallback order and check they can do its task.

    Args:
        models: One model, or models tried in order.
        param: The parameter that set them, for error messages.
        task: ``"generate"`` for roles that write text, ``"answer"`` otherwise.

    Raises:
        ValueError: No model was given, or a decision model was given a task
            that needs writing.
        MissingAPIKeyError: A model's API key is not set.
    """
    listed = tuple(models) if isinstance(models, (tuple, list)) else (models,)
    if not listed:
        raise ValueError(f"{param} needs at least one model.")
    endpoints = tuple(resolve(model) for model in listed)
    for endpoint in endpoints:
        if task == "generate" and endpoint.kind == "decision":
            raise ValueError(
                f'{param}="{endpoint.name}" is a decision model. It answers '
                "questions about samples but cannot write them. Use a reasoning "
                f'model such as "openai:gpt-5-mini" for {param}.'
            )
    return endpoints


@dataclass(frozen=True, slots=True)
class _Provider:
    protocol: Protocol
    base_url: str | None
    key_env: tuple[str, ...] = ()
    base_url_env: str | None = None
    local: bool = False
    json_object_only: bool = False
    decision: Protocol | None = None
    decision_base_url: str | None = None
    api_key_header: bool = False
    needs_base_url: bool = False


_PROVIDERS: Final[dict[str, _Provider]] = {
    "openai": _Provider(
        "openai",
        "https://api.openai.com/v1",
        ("OPENAI_API_KEY",),
        decision="openai-decisions",
    ),
    "anthropic": _Provider("anthropic", None, ("ANTHROPIC_API_KEY",)),
    "google": _Provider(
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "xai": _Provider("openai", "https://api.x.ai/v1", ("XAI_API_KEY",)),
    "openrouter": _Provider(
        "openai",
        "https://openrouter.ai/api/v1",
        ("OPENROUTER_API_KEY",),
        decision="systemone",
        decision_base_url="https://openrouter.ai/api/v1/systemone",
    ),
    "groq": _Provider("openai", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "together": _Provider(
        "openai", "https://api.together.xyz/v1", ("TOGETHER_API_KEY",)
    ),
    "deepseek": _Provider(
        "openai",
        "https://api.deepseek.com",
        ("DEEPSEEK_API_KEY",),
        json_object_only=True,
    ),
    "mistral": _Provider("openai", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    "ollama": _Provider(
        "openai",
        "http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
        local=True,
    ),
    "azure": _Provider(
        "openai",
        None,
        ("AZURE_OPENAI_API_KEY",),
        base_url_env="AZURE_OPENAI_BASE_URL",
        decision="systemone",
        api_key_header=True,
        needs_base_url=True,
    ),
    "typesafe": _Provider(
        "systemone",
        "https://api.typesafe.ai/v1/systemone",
        ("TYPESAFE_API_KEY",),
        decision="systemone",
    ),
    "fake": _Provider("fake", None, local=True, decision="fake"),
}

_ALIASES: Final[dict[str, str]] = {"jev": "typesafe:jev-latest"}

_DECISION_MODELS: Final[set[str]] = {
    "typesafe:*",
    "openai:gpt-6-luna*",
    "azure:Microsoft-Decision-*",
    "openrouter:typesafe/*",
}
"""Patterns of known decision models. Others need ``LLM(..., kind="decision")``."""


def _is_decision(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in _DECISION_MODELS)


def _route(
    provider: _Provider, provider_name: str, kind: ModelKind
) -> tuple[Protocol, str | None]:
    if kind == "reasoning":
        if provider.protocol == "systemone":
            raise ValueError(f"{provider_name} serves decision models only.")
        return provider.protocol, provider.base_url
    if provider.decision is None:
        raise ValueError(f"{provider_name} serves no decision models.")
    return provider.decision, provider.decision_base_url or provider.base_url


def _custom_endpoint(spec: LLM, name: str) -> Endpoint:
    if spec.base_url is None:
        raise ValueError(
            f'Model "{name}" has no provider. Write it as "provider:model", e.g. '
            '"openai:gpt-5-mini", or pass base_url= for an OpenAI-compatible server.'
        )
    kind = spec.kind or "reasoning"
    return Endpoint(
        name=name,
        provider="custom",
        model_id=name,
        protocol="systemone" if kind == "decision" else "openai",
        kind=kind,
        base_url=spec.base_url,
        api_key=spec.api_key,
        settings=spec,
        local=True,
    )


def _base_url(
    spec: LLM, provider: _Provider, default: str | None, name: str
) -> str | None:
    from_env = os.environ.get(provider.base_url_env) if provider.base_url_env else None
    base_url = spec.base_url or from_env or default
    if base_url is None and provider.needs_base_url:
        raise ValueError(
            f"{name} needs base_url=, or {provider.base_url_env} set in the "
            "environment, with your resource's endpoint."
        )
    return base_url


def _api_key(spec: LLM, provider: _Provider, name: str) -> str | None:
    if spec.api_key:
        return spec.api_key
    found = next(
        (os.environ[env] for env in provider.key_env if os.environ.get(env)), None
    )
    if found is None and provider.key_env:
        raise MissingAPIKeyError(
            f"{name} needs {provider.key_env[0]}, and it is not set. Set it in your "
            f'environment, or pass LLM("{name}", api_key=...).'
        )
    return found
