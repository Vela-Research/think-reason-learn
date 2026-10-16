"""Process-wide settings and per-model speed limits, in the style of scikit-learn."""

from __future__ import annotations

import dataclasses
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, Unpack

if TYPE_CHECKING:
    from ._spec import Endpoint, ModelName

MODEL_ALIASES: Final[dict[str, str]] = {"jev": "typesafe:jev-latest"}
"""Short model names, and the ``"provider:model"`` each stands for."""


@dataclass(frozen=True, slots=True)
class Config:
    """Settings for every model call.

    Args:
        cache: Reuse answers and generations already paid for, stored on disk.
        cache_dir: Folder of the cache. Defaults to
            ``$XDG_CACHE_HOME/think_reason_learn`` (``~/.cache`` if unset).
        max_cost_usd: Cost cap for each ``fit`` or ``predict`` run, in US dollars.
            ``None`` for no cap.
        max_retries: Retries of a call after a rate limit, server error or
            timeout.
        timeout: Seconds before a call times out (then it is retried).
    """

    cache: bool = True
    cache_dir: Path | None = None
    max_cost_usd: float | None = None
    max_retries: int = 5
    timeout: float = 120.0

    @property
    def cache_path(self) -> Path:
        """The cache's database file."""
        base = (
            self.cache_dir
            or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            / "think_reason_learn"
        )
        return base / "llm-cache-v1.sqlite"


@dataclass(frozen=True, slots=True)
class Limits:
    """Speed limits of one model on one account.

    Args:
        max_concurrency: Calls in flight at once.
        rpm: Requests per minute.
        tpm: Tokens per minute.
    """

    max_concurrency: int | None = None
    rpm: float | None = None
    tpm: float | None = None


def get_config() -> Config:
    """The settings in effect here."""
    return _context.get() or _default


class ConfigChanges(TypedDict, total=False):
    """Settings to change; those left out keep their value. See ``Config``."""

    cache: bool
    cache_dir: str | os.PathLike[str] | None
    max_cost_usd: float | None
    max_retries: int
    timeout: float


def set_config(**changes: Unpack[ConfigChanges]) -> None:
    """Change the settings for the whole process. See ``Config``.

    Only the settings passed change; ``max_cost_usd=None`` lifts a cap.
    """
    global _default
    with _lock:
        _default = _changed(_default, changes)


@contextmanager
def config_context(**changes: Unpack[ConfigChanges]) -> Iterator[Config]:
    """Change the settings inside a ``with`` block only. See ``Config``.

    Example:
        >>> with config_context(max_cost_usd=5):
        ...     model.fit(X, y)
    """
    changed = _changed(get_config(), changes)
    token = _context.set(changed)
    try:
        yield changed
    finally:
        _context.reset(token)


def set_limits(
    model: ModelName,
    *,
    max_concurrency: int | None = None,
    rpm: float | None = None,
    tpm: float | None = None,
) -> None:
    """Set the speed limits of a model, or of every model of a provider.

    Limits are per account: every estimator in the process that calls the
    model with the same key shares them. Without limits, calls start at 8 in
    flight and adapt to the provider's rate-limit replies.

    Args:
        model: ``"provider:model"``, or ``"provider"`` for all its models.
        max_concurrency: Calls in flight at once.
        rpm: Requests per minute, e.g. your account tier's limit.
        tpm: Tokens per minute.
    """
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}.")
    if any(rate is not None and rate <= 0 for rate in (rpm, tpm)):
        raise ValueError("rpm and tpm must be above 0.")
    with _lock:
        _limits[MODEL_ALIASES.get(model, model)] = Limits(max_concurrency, rpm, tpm)


def limits_for(endpoint: Endpoint) -> Limits:
    """The limits set for ``endpoint``'s model, or for its provider."""
    with _lock:
        return _limits.get(endpoint.name) or _limits.get(endpoint.provider) or Limits()


_lock: Final = threading.Lock()
_default: Config = Config()
_context: Final[ContextVar[Config | None]] = ContextVar("trl_config", default=None)
_limits: Final[dict[str, Limits]] = {}


def _changed(config: Config, changes: ConfigChanges) -> Config:
    if (folder := changes.get("cache_dir")) is not None:
        changes = {**changes, "cache_dir": Path(folder)}
    return dataclasses.replace(config, **changes)
