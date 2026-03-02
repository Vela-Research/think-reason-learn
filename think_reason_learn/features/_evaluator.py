"""Safe evaluation of LLM-generated lambda expressions.

Lambda expression strings are compiled via :func:`eval` with a **restricted**
``__builtins__`` dict so that only safe built-in names are available.  This
blocks ``__import__``, ``exec``, ``open``, ``getattr``, and every other
dangerous built-in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pandas as pd

from ._types import HelperFunction, Rule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Restricted builtins whitelist
# ---------------------------------------------------------------------------

SAFE_BUILTINS: dict[str, Any] = {
    # Aggregation / iteration
    "len": len,
    "any": any,
    "all": all,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    # Type constructors / checks
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "set": set,
    "dict": dict,
    "tuple": tuple,
    "isinstance": isinstance,
    # Constants
    "True": True,
    "False": False,
    "None": None,
}
"""Built-ins available inside evaluated lambda expressions.

Explicitly **excluded**: ``__import__``, ``eval``, ``exec``, ``compile``,
``open``, ``getattr``, ``setattr``, ``delattr``, ``globals``, ``locals``,
``vars``, ``dir``, ``type``, ``super``, ``property``, ``breakpoint``, etc.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _CompiledRule:
    rule: Rule
    func: Callable[..., Any] | None
    error: str | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FeatureEvaluator:
    """Compile and evaluate LLM-generated lambda rules on structured records.

    Parameters
    ----------
    rules:
        Rules whose ``expression`` fields are Python lambda strings.
    helpers:
        Optional helper functions to make available inside the lambdas (e.g.
        ``parse_qs``, ``parse_duration``).  Each helper's ``.func`` is placed
        into the eval context under its ``.name``.
    """

    def __init__(
        self,
        rules: Sequence[Rule],
        helpers: Sequence[HelperFunction] | None = None,
    ) -> None:
        self._rules = list(rules)
        self._helper_map: dict[str, Callable[..., Any]] = {
            h.name: h.func for h in (helpers or [])
        }
        self._compiled: list[_CompiledRule] = self._compile_all()

    # -- compilation --------------------------------------------------------

    def _compile_all(self) -> list[_CompiledRule]:
        eval_globals: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            **self._helper_map,
        }
        compiled: list[_CompiledRule] = []
        for rule in self._rules:
            try:
                func = eval(rule.expression, eval_globals)  # noqa: S307
                compiled.append(_CompiledRule(rule=rule, func=func, error=None))
            except Exception as exc:
                logger.warning("Compile error for rule %s: %s", rule.name, exc)
                compiled.append(_CompiledRule(rule=rule, func=None, error=str(exc)))
        return compiled

    # -- single-record evaluation -------------------------------------------

    def evaluate(self, record: dict[str, Any]) -> dict[str, int]:
        """Evaluate all rules on a single record.

        Returns a dict mapping ``rule_name → 0 | 1``.  Rules that fail to
        compile or that raise at runtime silently return ``0``.
        """
        results: dict[str, int] = {}
        for entry in self._compiled:
            name = entry.rule.name
            if entry.func is None:
                results[name] = 0
                continue
            try:
                results[name] = 1 if entry.func(record) else 0
            except Exception:
                results[name] = 0
        return results

    # -- batch evaluation ---------------------------------------------------

    def evaluate_df(self, records: Sequence[dict[str, Any]]) -> pd.DataFrame:
        """Evaluate all rules on multiple records.

        Returns a :class:`~pandas.DataFrame` with one column per rule (values
        ``0`` or ``1``) and one row per record.
        """
        rows = [self.evaluate(r) for r in records]
        return pd.DataFrame(rows)

    # -- introspection ------------------------------------------------------

    @property
    def rules(self) -> list[Rule]:
        """The rules this evaluator was initialised with."""
        return list(self._rules)

    @property
    def compilation_errors(self) -> dict[str, str]:
        """Rules that failed to compile, keyed by rule name."""
        return {e.rule.name: e.error for e in self._compiled if e.error is not None}
