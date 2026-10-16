"""Decision models read from a provider's public catalog, not hard-coded.

OpenRouter lists its decision models, with prices, at ``_OPENROUTER_URL`` (no key
needed). The list is kept on disk for a day, so new and renamed models work
without a release of this library.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Final

import httpx2
from pydantic import BaseModel, Field, JsonValue, ValidationError

from ._config import get_config
from ._usage import Price

logger = logging.getLogger(__name__)

_OPENROUTER_URL: Final = (
    "https://openrouter.ai/api/v1/models?output_modalities=decisions"
)
_MAX_AGE: Final = 24 * 3600.0
_RETRY_AFTER: Final = 300.0
_TIMEOUT: Final = 5.0

type Catalog = Mapping[str, Price | None]
"""Decision model ids and their prices, where listed."""

_loaded: tuple[float, Catalog] | None = None
"""When the catalog was read, and what it holds."""


def openrouter_decisions() -> Catalog:
    """OpenRouter's decision models, from memory, the disk, or the network.

    Read at most once a day, and a failed read again after five minutes; a model
    resolved meanwhile costs no request.
    ponytail: the daily refresh is a blocking GET of a few kilobytes, with a
    5-second timeout; move it off the event loop if it ever shows in a profile.
    """
    global _loaded
    now = time.time()
    if _loaded is not None and now - _loaded[0] < _MAX_AGE:
        return _loaded[1]
    path = get_config().cache_path.parent / "openrouter-decisions.json"
    stored = _read(path)
    if stored is not None and 0 <= now - stored[0] < _MAX_AGE:
        _loaded = stored
        return stored[1]
    try:
        models = _parse(_download())
    except (httpx2.HTTPError, ValueError) as error:  # ValidationError included
        fallback = "using the list from before" if stored else "none are known"
        logger.warning(
            f"Could not read OpenRouter's decision models ({error}); {fallback}. "
            'Mark one with LLM("openrouter:...", kind="decision").'
        )
        models: Catalog = stored[1] if stored is not None else {}
        _loaded = (now - _MAX_AGE + _RETRY_AFTER, models)  # try again soon
        return models
    _write(path, now, models)
    _loaded = (now, models)
    return models


def _download() -> JsonValue:
    with httpx2.Client(timeout=_TIMEOUT) as client:
        response = client.get(_OPENROUTER_URL)
        response.raise_for_status()
        payload: JsonValue = response.json()
        return payload


def _parse(payload: JsonValue) -> dict[str, Price | None]:
    """Decision models from a catalog response; prices per token become per million.

    Raises:
        ValueError: The response has no model list.
    """
    listed = (_listed(entry) for entry in _Catalog.model_validate(payload).data)
    return {
        model.id: _price(model.pricing)
        for model in listed
        if model is not None and "decisions" in model.architecture.output_modalities
    }


def _listed(entry: JsonValue) -> _Listed | None:
    """One model of the catalog; an entry this library cannot read is skipped."""
    try:
        return _Listed.model_validate(entry)
    except ValidationError:
        return None


def _price(pricing: JsonValue) -> Price | None:
    """The price per million tokens; ``None`` unless both are amounts (not ``-1``)."""
    million = Decimal(1_000_000)
    try:
        per_token = _Pricing.model_validate(pricing)
        return Price(
            input=float(per_token.prompt * million),
            output=float(per_token.completion * million),
        )
    except ValueError:  # ValidationError included
        return None


def _read(path: Path) -> tuple[float, Catalog] | None:
    try:
        stored = _Stored.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        if path.exists():
            logger.warning(f"Ignoring an unreadable {path.name}: {error}")
        return None
    models = {
        name: Price(*price) if price is not None else None
        for name, price in stored.models.items()
    }
    return stored.fetched_at, models


def _write(path: Path, now: float, models: Catalog) -> None:
    stored = _Stored(
        fetched_at=now,
        models={
            name: (price.input, price.output) if price is not None else None
            for name, price in models.items()
        },
    )
    partial = path.with_name(f"{path.name}.{os.getpid()}")
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        partial.write_text(stored.model_dump_json(), "utf-8")
        partial.replace(path)  # atomic, so a reader never sees half a file
    except OSError as error:
        logger.warning(f"Could not keep OpenRouter's decision models on disk: {error}")


type _Amount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Catalog(BaseModel):
    data: list[JsonValue]


class _Architecture(BaseModel):
    output_modalities: list[str] = Field(default_factory=list[str])


class _Pricing(BaseModel):
    prompt: Decimal
    completion: Decimal


class _Listed(BaseModel):
    id: str
    architecture: _Architecture = Field(default_factory=_Architecture)
    pricing: JsonValue = None


class _Stored(BaseModel):
    """The catalog as kept on disk."""

    fetched_at: float
    models: dict[str, tuple[_Amount, _Amount] | None]
