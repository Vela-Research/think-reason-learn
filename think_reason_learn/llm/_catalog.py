"""Decision models read from a provider's public catalog, not hard-coded.

OpenRouter lists its decision models, with prices, at ``_OPENROUTER_URL`` (no key
needed). The list is kept on disk for a day, so new and renamed models work
without a release of this library.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

import httpx2

from ._config import get_config
from ._usage import Price

logger = logging.getLogger(__name__)

_OPENROUTER_URL: Final = (
    "https://openrouter.ai/api/v1/models?output_modalities=decisions"
)
_MAX_AGE: Final = 24 * 3600.0
_TIMEOUT: Final = 5.0

type Catalog = Mapping[str, Price | None]
"""Decision model ids and their prices, where listed."""


def openrouter_decisions() -> Catalog:
    """OpenRouter's decision models, from memory, the disk, or the network.

    Read at most once a day; a model resolved meanwhile costs no request.
    ponytail: the daily refresh is a blocking GET of a few kilobytes, with a
    5-second timeout; move it off the event loop if it ever shows in a profile.
    """
    global _loaded
    now = time.time()
    if _loaded is not None and now - _loaded[0] < _MAX_AGE:
        return _loaded[1]
    path = get_config().cache_path.parent / "openrouter-decisions.json"
    stored = _read(path)
    if stored is not None and now - stored[0] < _MAX_AGE:
        _loaded = stored
        return stored[1]
    try:
        models = _parse(_download())
    except (httpx2.HTTPError, ValueError) as error:
        fallback = "using the list from before" if stored else "none are known"
        logger.warning(
            f"Could not read OpenRouter's decision models ({error}); {fallback}. "
            'Mark one with LLM("openrouter:...", kind="decision").'
        )
        models = stored[1] if stored is not None else {}
    else:
        _write(path, now, models)
    _loaded = (now, models)
    return models


_loaded: tuple[float, Catalog] | None = None
"""When the catalog was read, and what it holds."""


def _download() -> object:
    with httpx2.Client(timeout=_TIMEOUT) as client:
        response = client.get(_OPENROUTER_URL)
        response.raise_for_status()
        return response.json()


def _parse(payload: object) -> dict[str, Price | None]:
    """Decision models from a catalog response; prices per token become per million."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("the catalog has no model list")
    return {
        model["id"]: _price(model.get("pricing"))
        for model in data
        if isinstance(model, dict)
        and isinstance(model.get("id"), str)
        and "decisions" in _outputs(model)
    }


def _outputs(model: dict[str, object]) -> list[object]:
    architecture = model.get("architecture")
    outputs = (
        architecture.get("output_modalities")
        if isinstance(architecture, dict)
        else None
    )
    return outputs if isinstance(outputs, list) else []


def _price(pricing: object) -> Price | None:
    if not isinstance(pricing, dict):
        return None
    try:
        per_token = [
            Decimal(str(pricing.get(k, "0"))) for k in ("prompt", "completion")
        ]
    except InvalidOperation:
        return None
    million = Decimal(1_000_000)
    return Price(
        input=float(per_token[0] * million), output=float(per_token[1] * million)
    )


def _read(path: Path) -> tuple[float, Catalog] | None:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        models = {
            name: Price(*price) if price is not None else None
            for name, price in stored["models"].items()
        }
        return float(stored["fetched_at"]), models
    except (OSError, ValueError, KeyError, TypeError) as error:
        if path.exists():
            logger.warning(f"Ignoring an unreadable {path.name}: {error}")
        return None


def _write(path: Path, now: float, models: Catalog) -> None:
    rows = {
        name: [price.input, price.output] if price is not None else None
        for name, price in models.items()
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": now, "models": rows}), "utf-8")
    except OSError as error:
        logger.warning(f"Could not keep OpenRouter's decision models on disk: {error}")
