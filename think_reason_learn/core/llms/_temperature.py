"""Send a set temperature, and drop it for models that reject one."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])

_REJECTING_MODELS: Final[set[str]] = set()
"""Models that rejected a temperature in this process, e.g. reasoning models."""


def drop_rejected_temperature(omit: object) -> Callable[[F], F]:
    """Retry an adapter call without its temperature when the model rejects it.

    Some models (e.g. OpenAI's reasoning models) reject any temperature. The
    first rejection is retried once with ``omit`` and remembered, so later calls
    to that model leave the temperature out. Other failures keep the adapter's
    contract: raised with ``raise_=True``, ``None`` otherwise.

    Args:
        omit: The adapter's value for "no temperature".
    """

    def decorate(method: F) -> F:
        if inspect.iscoroutinefunction(method):
            return _wrap_async(method, omit)  # type: ignore[return-value]  # same signature as method
        return _wrap_sync(method, omit)  # type: ignore[return-value]  # same signature as method

    return decorate


def _wrap_sync(method: Callable[..., object], omit: object) -> Callable[..., object]:
    @functools.wraps(method)
    def respond_sync(
        self: object,
        *,
        model: str,
        temperature: object = omit,
        raise_: bool = False,
        **kwargs: object,
    ) -> object:
        temperature = omit if model in _REJECTING_MODELS else temperature
        if not _is_set(temperature):
            return method(
                self, model=model, temperature=temperature, raise_=raise_, **kwargs
            )
        try:
            return method(
                self, model=model, temperature=temperature, raise_=True, **kwargs
            )
        except Exception as exc:  # the adapter's contract: any failure is None
            if not _rejects_temperature(exc):
                return _fail(exc, raise_=raise_)
        _note_rejection(model, temperature)
        return method(self, model=model, temperature=omit, raise_=raise_, **kwargs)

    return respond_sync


def _wrap_async(
    method: Callable[..., Awaitable[object]], omit: object
) -> Callable[..., Awaitable[object]]:
    @functools.wraps(method)
    async def respond(
        self: object,
        *,
        model: str,
        temperature: object = omit,
        raise_: bool = False,
        **kwargs: object,
    ) -> object:
        temperature = omit if model in _REJECTING_MODELS else temperature
        if not _is_set(temperature):
            return await method(
                self, model=model, temperature=temperature, raise_=raise_, **kwargs
            )
        try:
            return await method(
                self, model=model, temperature=temperature, raise_=True, **kwargs
            )
        except Exception as exc:  # the adapter's contract: any failure is None
            if not _rejects_temperature(exc):
                return _fail(exc, raise_=raise_)
        _note_rejection(model, temperature)
        return await method(
            self, model=model, temperature=omit, raise_=raise_, **kwargs
        )

    return respond


def _is_set(temperature: object) -> bool:
    return isinstance(temperature, (int, float))


def _rejects_temperature(exc: Exception) -> bool:
    return "temperature" in str(exc).lower()


def _fail(exc: Exception, *, raise_: bool) -> None:
    """The adapter already logged ``exc``: re-raise it or report no response."""
    if raise_:
        raise exc


def _note_rejection(model: str, temperature: object) -> None:
    _REJECTING_MODELS.add(model)
    logger.warning(
        f"{model} does not accept temperature={temperature}; "
        "using the model's default temperature from now on."
    )
