"""Token usage, prices, cost and budgets, recorded the same way for every model."""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import logging
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import pandas as pd

from think_reason_learn.exceptions import BudgetExceededError

if TYPE_CHECKING:
    from ._spec import Endpoint

logger = logging.getLogger(__name__)

type CostSource = Literal[
    "provider", "price", "price_table", "cache", "none", "unknown"
]
"""Where a call's cost came from: the provider's bill, a price, the cache (free),
nothing to pay (a failed call), or nowhere (unknown)."""


@dataclass(frozen=True, slots=True)
class Price:
    """A model's price in US dollars per million tokens.

    Args:
        input: Per million input tokens.
        output: Per million output tokens, reasoning included.
        cached_input: Per million input tokens read from the provider's cache.
            Defaults to ``input``.
        cache_write: Per million input tokens written to the provider's cache.
            Defaults to ``input``.
    """

    input: float
    output: float = 0.0
    cached_input: float | None = None
    cache_write: float | None = None

    def cost(self, tokens: TokenCounts) -> float:
        """The cost in US dollars of ``tokens``."""
        uncached = tokens.input - tokens.cache_read - tokens.cache_write
        cached = self.input if self.cached_input is None else self.cached_input
        written = self.input if self.cache_write is None else self.cache_write
        return (
            uncached * self.input
            + tokens.cache_read * cached
            + tokens.cache_write * written
            + tokens.output * self.output
        ) / 1e6


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Tokens of one call.

    ``input`` includes cache reads and writes; ``output`` includes reasoning.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One call to a model, or a batch of answers served from the cache.

    Field names follow OpenTelemetry's ``gen_ai.usage`` conventions.
    """

    time: float
    role: str
    model: str
    served_model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    cost_source: CostSource
    latency_s: float
    calls: int = 1
    cached: bool = False
    error: str | None = None


class Usage:
    """Every call a model made: tokens, cost and where the cost came from.

    Read ``cost_usd`` for the total, ``by("role", "model")`` for a summary, or
    ``to_frame()`` and ``to_json()`` for each call.
    """

    def __init__(self, records: Iterable[UsageRecord] = ()) -> None:
        """Start from ``records``, if any."""
        self._records: list[UsageRecord] = list(records)

    def add(self, record: UsageRecord) -> None:
        """Record one call."""
        self._records.append(record)

    @property
    def cost_usd(self) -> float:
        """Total cost in US dollars of the calls with a known cost."""
        return sum(r.cost_usd for r in self._records if r.cost_usd is not None)

    @property
    def unpriced(self) -> frozenset[str]:
        """Models with calls whose cost is unknown."""
        return frozenset(r.model for r in self._records if r.cost_source == "unknown")

    @property
    def calls(self) -> int:
        """Calls sent to providers, not counting answers from the cache."""
        return sum(r.calls for r in self._records if not r.cached)

    def to_frame(self) -> pd.DataFrame:
        """One row per record."""
        columns = [f.name for f in dataclasses.fields(UsageRecord)]
        return pd.DataFrame(
            [dataclasses.asdict(r) for r in self._records], columns=pd.Index(columns)
        )

    def by(self, *keys: str) -> pd.DataFrame:
        """Calls, tokens and cost summed by ``keys``, e.g. ``by("role", "model")``."""
        summed = [
            "calls",
            "input_tokens",
            "cache_read_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cost_usd",
        ]
        frame = self.to_frame()
        return (
            frame.groupby(list(keys or ("model",)), dropna=False)[summed]
            .sum()
            .reset_index()
        )

    def to_json(self, path: str | PathLike[str]) -> None:
        """Write every record to ``path`` as JSON."""
        records = [dataclasses.asdict(r) for r in self._records]
        Path(path).write_text(json.dumps(records, indent=1), encoding="utf-8")

    def __len__(self) -> int:
        """Number of records."""
        return len(self._records)

    def __iter__(self) -> Iterator[UsageRecord]:
        """The records, oldest first."""
        return iter(self._records)

    def __repr__(self) -> str:
        """Calls and cost in one line."""
        unknown = f", unknown cost for {sorted(self.unpriced)}" if self.unpriced else ""
        return f"Usage(calls={self.calls}, cost_usd={self.cost_usd:.4f}{unknown})"


def call_cost(
    endpoint: Endpoint, tokens: TokenCounts, provider_cost: float | None
) -> tuple[float | None, CostSource]:
    """A call's cost: the provider's, else the model's price, else a price table."""
    if provider_cost is not None:
        return provider_cost, "provider"
    if (price := endpoint.settings.price or _built_in_price(endpoint.name)) is not None:
        return price.cost(tokens), "price"
    if (cost := _price_table_cost(endpoint, tokens)) is not None:
        return cost, "price_table"
    return None, "unknown"


class Budget:
    """A cost cap shared by every call in one run.

    Args:
        max_cost_usd: The most a run may spend, in US dollars. ``None`` for no cap.
    """

    def __init__(self, max_cost_usd: float | None) -> None:
        """Start a run with nothing spent."""
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0
        self._reserved_usd = 0.0

    @property
    def remaining_usd(self) -> float:
        """Dollars left under the cap, net of calls in flight."""
        cap = math.inf if self.max_cost_usd is None else self.max_cost_usd
        return cap - self.spent_usd - self._reserved_usd

    def check_estimate(self, estimate_usd: float, what: str) -> None:
        """Refuse to start work estimated to pass the cap."""
        if estimate_usd > self.remaining_usd:
            raise BudgetExceededError(
                f"{what} is estimated at {format_usd(estimate_usd)}, more than the "
                f"{format_usd(max(self.remaining_usd, 0))} left under this run's cap "
                f"of {format_usd(self.max_cost_usd or 0)}. Nothing was sent. Raise "
                "it with think_reason_learn.config_context(max_cost_usd=...)."
            )

    def reserve(self, estimate_usd: float | None) -> float:
        """Hold the estimate of one call; returns what was held."""
        held = estimate_usd or 0.0
        if self.max_cost_usd is not None and held > self.remaining_usd:
            raise BudgetExceededError(
                self._stopped(f"the next call ({format_usd(held)})")
            )
        self._reserved_usd += held
        return held

    def settle(self, held: float, cost_usd: float | None) -> None:
        """Replace a held estimate with the call's cost."""
        self._reserved_usd -= held
        self.spent_usd += cost_usd or 0.0

    def check(self) -> None:
        """Stop the run once spending has passed the cap."""
        if self.max_cost_usd is not None and self.spent_usd > self.max_cost_usd:
            raise BudgetExceededError(self._stopped("spending"))

    def _stopped(self, what: str) -> str:
        return (
            f"The run stopped: {what} would pass its cap of "
            f"{format_usd(self.max_cost_usd or 0)} ({format_usd(self.spent_usd)} "
            "spent). Answers so far are cached; rerun with a higher cap to continue "
            "without paying for them again."
        )


def format_usd(usd: float) -> str:
    """Dollars to cents, or to the first two significant digits below a cent."""
    if usd <= 0:
        return "$0.00"
    if usd >= 0.01:
        return f"${usd:.2f}"
    return f"${usd:.{1 - math.floor(math.log10(usd))}f}"


_BUILT_IN_PRICES: Final[dict[str, Price]] = {"typesafe:*": Price(input=0.042)}
"""Prices of decision models the price table may lack (Typesafe list price)."""


def _built_in_price(name: str) -> Price | None:
    return next(
        (
            p
            for pattern, p in _BUILT_IN_PRICES.items()
            if fnmatch.fnmatchcase(name, pattern)
        ),
        None,
    )


def _price_table_cost(endpoint: Endpoint, tokens: TokenCounts) -> float | None:
    try:
        import genai_prices
    except ImportError:
        return None
    usage = genai_prices.Usage(
        input_tokens=tokens.input,
        output_tokens=tokens.output,
        cache_read_tokens=tokens.cache_read,
        cache_write_tokens=tokens.cache_write,
    )
    try:
        price = genai_prices.calc_price(
            usage, endpoint.model_id, provider_id=endpoint.provider
        )
    except LookupError:
        return None
    total: Decimal = price.total_price
    return float(total)
