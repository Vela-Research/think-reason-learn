"""Model specs: ``"provider:model"`` strings, ``LLM`` settings, the provider table."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Final, Literal
from urllib.parse import urlparse

from think_reason_learn.exceptions import MissingAPIKeyError

from ._catalog import Catalog, openrouter_decisions
from ._config import MODEL_ALIASES
from ._usage import Price

logger = logging.getLogger(__name__)

type ModelName = str
"""A model as ``"provider:model"``, e.g. ``"openai:gpt-5-mini"``, or ``"jev"``."""

type ModelKind = Literal["reasoning", "decision"]
"""Reasoning models write text and answer questions; decision models only answer."""

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

type WireProtocol = Literal[
    "openai", "anthropic", "systemone", "openai-decisions", "fake"
]

type Task = Literal["generate", "answer"]
"""What a role asks of its models: writing text, or answering about samples."""


@dataclass(frozen=True, slots=True)
class LLM:
    """A model and the settings that change how it answers.

    Args:
        model: ``"provider:model"``, e.g. ``"anthropic:claude-sonnet-5-5"``. With
            ``base_url``, a bare model id calls any OpenAI-compatible server.
        temperature: Sampling temperature. ``None``: 0 when a reasoning model
            answers questions, so answers repeat, and the model's default when
            it writes or casts votes to average. A model that rejects a
            temperature is called without one, with a warning.
        reasoning_effort: Reasoning effort for models that support it. Claude's
            lowest is ``"low"``, which ``"none"`` and ``"minimal"`` ask for.
        max_output_tokens: Cap on output tokens per call.
        service_tier: Provider service tier, e.g. ``"flex"`` for half price on
            OpenAI and Gemini at slower speed.
        questions_per_call: Questions a reasoning model answers per call about
            one sample. Decision models take every question in one call.
        base_url: Endpoint to call instead of the provider's. It gets only an
            ``api_key`` passed here, never the provider's key from the
            environment. That holds for Azure too: a ``base_url`` can arrive in
            a saved model from someone else, so set ``AZURE_OPENAI_BASE_URL``
            in your environment, or pass ``api_key`` with it.
        api_key: Key to use instead of the provider's environment variable.
        kind: ``"decision"`` to call the model's decision API, e.g. OpenAI's
            ``gpt-6-luna`` (also a chat model) or an Azure deployment of
            Microsoft-Decision-1. By default Typesafe's models and those
            OpenRouter lists as decision models are decision models; the rest
            are reasoning models.
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

    def __copy__(self) -> LLM:
        """Itself: it cannot change, and a copy keeps its key (``clone``)."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> LLM:
        """Itself, so ``sklearn.clone`` and grid search keep the API key."""
        return self

    def __getstate__(self) -> tuple[object, ...]:
        """Pickle without the API key: saved models never hold secrets.

        Workers that receive a model by pickle (``n_jobs`` in scikit-learn) read
        keys from their environment.
        """
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
    protocol: WireProtocol
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
    name = MODEL_ALIASES.get(spec.model, spec.model)
    provider_name, has_provider, model_id = name.partition(":")
    provider = _PROVIDERS.get(provider_name)
    if not has_provider or (provider is None and spec.base_url is not None):
        return _custom_endpoint(spec, name)
    if provider is None or not model_id:
        raise ValueError(
            f'Unknown model "{name}". Write it as "provider:model" with a provider '
            f"in {sorted(_PROVIDERS)}, or pass base_url= for an OpenAI-compatible "
            "server."
        )
    kind, price = _kind(spec, provider, model_id)
    protocol, base_url = _route(provider, provider_name, kind)
    return _checked_transport(
        Endpoint(
            name=name,
            provider=provider_name,
            model_id=model_id,
            protocol=protocol,
            kind=kind,
            base_url=_base_url(spec, provider, base_url, name),
            api_key=_api_key(spec, provider, name),
            settings=spec
            if spec.price or price is None
            else replace(spec, price=price),
            local=provider.local,
            json_object_only=provider.json_object_only,
            api_key_header=provider.api_key_header,
        )
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
    protocol: WireProtocol
    base_url: str | None
    key_env: tuple[str, ...] = ()
    base_url_env: str | None = None
    local: bool = False
    json_object_only: bool = False
    decision: WireProtocol | None = None
    decision_base_url: str | None = None
    api_key_header: bool = False
    needs_base_url: bool = False
    catalog: Callable[[], Catalog] | None = None
    """The provider's decision models, read from where it lists them."""


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
        ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"),
    ),
    "xai": _Provider("openai", "https://api.x.ai/v1", ("XAI_API_KEY",)),
    "openrouter": _Provider(
        "openai",
        "https://openrouter.ai/api/v1",
        ("OPENROUTER_API_KEY",),
        decision="systemone",
        decision_base_url="https://openrouter.ai/api/alpha/decisions",
        catalog=openrouter_decisions,
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
}


_keys_needed: Final[ContextVar[bool]] = ContextVar("trl_keys_needed", default=True)
_warned_plain: Final[set[str]] = set()
"""Plain-http URLs already warned about."""


@contextmanager
def keys_optional() -> Generator[None]:
    """Resolve models without their API keys inside the block.

    For planning calls that are never sent, as ``estimate_cost`` does.
    """
    token = _keys_needed.set(False)
    try:
        yield
    finally:
        _keys_needed.reset(token)


@contextmanager
def extra_provider(
    name: str,
    protocol: WireProtocol,
    base_url: str,
    *,
    catalog: Callable[[], Catalog] | None = None,
    local: bool = True,
) -> Generator[None]:
    """Serve ``"<name>:<model>"`` inside the block, from ``base_url``.

    For tests: ``testing.fake_llms`` adds its fake provider this way. A local
    provider's models are free and get fewer calls at once.
    """
    before = _PROVIDERS.get(name)
    _PROVIDERS[name] = _Provider(
        protocol, base_url, local=local, decision=protocol, catalog=catalog
    )
    try:
        yield
    finally:
        if before is None:
            del _PROVIDERS[name]
        else:
            _PROVIDERS[name] = before


def _kind(
    spec: LLM, provider: _Provider, model_id: str
) -> tuple[ModelKind, Price | None]:
    """Whether a model is a decision model, and the price its provider lists.

    It is one if ``kind="decision"`` says so, if its provider serves only
    decision models, or if the provider's catalog lists it. Never by its name.
    """
    if spec.kind == "reasoning":
        return "reasoning", None
    listed: Catalog = provider.catalog() if provider.catalog is not None else {}
    if spec.kind == "decision" or provider.protocol == "systemone":
        return "decision", listed.get(model_id)
    if model_id in listed:
        return "decision", listed[model_id]
    return "reasoning", None


def _route(
    provider: _Provider, provider_name: str, kind: ModelKind
) -> tuple[WireProtocol, str | None]:
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
    return _checked_transport(
        Endpoint(
            name=name,
            provider="custom",
            model_id=name,
            protocol="systemone" if kind == "decision" else "openai",
            kind=kind,
            base_url=spec.base_url,
            api_key=spec.api_key,
            settings=spec,
            local=_is_local(spec.base_url),
        )
    )


def _checked_transport(endpoint: Endpoint) -> Endpoint:
    """``endpoint``; a key it would send unencrypted off this network is warned of."""
    url = endpoint.base_url or ""
    plain = url.startswith("http://") and not _is_local(url)
    if plain and endpoint.api_key and url not in _warned_plain:
        _warned_plain.add(url)
        logger.warning(
            f"{endpoint.name} sends its API key over plain http to {url}; "
            "use https, or anyone on the way can read the key."
        )
    return endpoint


def _is_local(url: str) -> bool:
    """Whether ``url`` is on this machine or network (fewer calls at once there)."""
    host = urlparse(url).hostname or ""
    try:
        return ipaddress.ip_address(host).is_private  # loopback and LAN included
    except ValueError:
        return "." not in host or host.endswith(".local")  # localhost, gpu-box


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
    if spec.base_url is not None:
        # The provider's key never goes to a server named in code or in a file.
        # A loaded model.json carries base_url, so a file from someone else
        # would otherwise send your key to their host. An endpoint set through
        # the provider's own environment variable still gets the key, since a
        # file cannot set that.
        return None
    found = next(
        (os.environ[env] for env in provider.key_env if os.environ.get(env)), None
    )
    if found is None and provider.key_env and _keys_needed.get():
        raise MissingAPIKeyError(
            f"{name} needs {provider.key_env[0]}, and it is not set. Set it in your "
            f'environment, or pass LLM("{name}", api_key=...).'
        )
    return found
