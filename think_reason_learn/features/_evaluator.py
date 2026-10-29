"""Safe evaluation of model-written lambda rules on structured records.

Each rule is checked before it runs: it must be one lambda of one argument, it
may use only the attributes of built-in data types (``.get``, ``.lower``...) that
change nothing, and no name that starts with two underscores (``__import__``).
Any other attribute, such as a generator's ``gi_frame``, could walk out of the
sandbox, and a method such as ``.clear()`` could change the records the next
rules read. A default argument and ``:=`` are both refused, because a default
is evaluated while ``eval`` builds the lambda, so
``lambda r=(len := lambda x: 0): False`` would rebind ``len`` in the scope it
compiles in. Each rule gets its own scope as well, so one rule cannot change
what another rule's names mean. It then runs
with only the built-ins in ``SAFE_BUILTINS`` and the helpers given, so
``import``, ``open``, ``eval``, ``getattr`` and the like are out of reach. Rules
come from a model that read your data: treat them as untrusted code, and keep
these checks.

ponytail: a rule that computes for ever (``lambda r: 10**10**10``) hangs the
fit; run rules in a subprocess with a timeout if that ever matters.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from ._types import HelperFunction, Record, Rule

logger = logging.getLogger(__name__)

type RuleFunction = Callable[[Record], object]

SAFE_BUILTINS: Final[Mapping[str, object]] = {
    # Aggregation and iteration
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
    # Type constructors and checks
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
"""Built-ins available inside rules; everything else is absent."""

_MUTATORS: Final = frozenset(
    {
        "add",
        "append",
        "clear",
        "difference_update",
        "discard",
        "extend",
        "insert",
        "intersection_update",
        "pop",
        "popitem",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "symmetric_difference_update",
        "update",
    }
)
"""Methods that change a list, dict or set: a rule may only read a record."""

_ATTRIBUTES: Final = (
    frozenset(
        name
        for kind in (
            str,
            bytes,
            int,
            float,
            bool,
            list,
            tuple,
            dict,
            set,
            frozenset,
            range,
        )
        for name in dir(kind)
        if not name.startswith("_")
    )
    - {"format", "format_map"}
    - _MUTATORS
)
"""Attributes rules may use; ``format`` reaches any, as ``"{0.__class__}"``."""

_RULE_ERRORS: Final = (
    ArithmeticError,
    LookupError,
    TypeError,
    ValueError,
    AttributeError,
    NameError,
    RecursionError,
    MemoryError,
)
"""What a rule may raise on a record it does not fit; the rule then gives 0."""


class FeatureEvaluator:
    """Compile and evaluate model-written lambda rules on structured records.

    Args:
        rules: Rules whose ``expression`` is a Python lambda of one argument.
        helpers: Helper functions the lambdas may call, by their ``name``.

    Raises:
        ValueError: Two rules share a name.
    """

    def __init__(
        self,
        rules: Sequence[Rule],
        helpers: Sequence[HelperFunction] | None = None,
    ) -> None:
        """Check and compile every rule; a rule that fails is kept, giving 0."""
        names = [rule.name for rule in rules]
        if duplicates := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"Rules need distinct names; repeated: {duplicates}.")
        self._rules = list(rules)
        self._helpers = {h.name: h.func for h in helpers or ()}
        self._compiled = [_compile(rule, self._scope()) for rule in self._rules]

    def _scope(self) -> dict[str, object]:
        """A fresh globals mapping, so one rule cannot rebind a name for another."""
        return {"__builtins__": SAFE_BUILTINS, **self._helpers}

    def evaluate(self, record: Record) -> dict[str, int]:
        """Each rule's result on ``record``: 1 or 0.

        A rule that failed to compile, or raises on this record, gives 0.
        """
        return {entry.rule.name: entry.run(record) for entry in self._compiled}

    def evaluate_df(self, records: Sequence[Record]) -> pd.DataFrame:
        """One column per rule (0 or 1) and one row per record.

        A rule that raises on every record scores 0 everywhere, which reads as
        a rule that is never true, so this warns about those by name.
        """
        before = {e.rule.name: e.failures[0] for e in self._compiled}
        columns = pd.Index([rule.name for rule in self._rules])
        frame = pd.DataFrame([self.evaluate(r) for r in records], columns=columns)
        if records:
            always = [
                e.rule.name
                for e in self._compiled
                if e.function is not None
                and e.failures[0] - before[e.rule.name] == len(records)
            ]
            if always:
                logger.warning(
                    f"These rules raised on every record, so they score 0 "
                    f"everywhere: {', '.join(sorted(always))}."
                )
        return frame

    @property
    def rules(self) -> list[Rule]:
        """The rules, in order."""
        return list(self._rules)

    @property
    def compilation_errors(self) -> dict[str, str]:
        """Why each rule that failed to compile failed, by rule name."""
        return {e.rule.name: e.error for e in self._compiled if e.error is not None}

    @property
    def rule_failures(self) -> dict[str, int]:
        """Records each rule raised on, by rule name, for rules that ever did.

        A rule that compiles can still fail on the data, for instance when a
        field it reads is text in some records and a number in others. It
        scores 0 on those records, which no column tells apart from a rule
        that is simply not true, so count them here.
        """
        return {e.rule.name: e.failures[0] for e in self._compiled if e.failures[0]}


@dataclass(frozen=True, slots=True)
class _Compiled:
    rule: Rule
    function: RuleFunction | None
    error: str | None
    failures: list[int] = field(default_factory=lambda: [0])
    """Records this rule raised on, in a one-item list so the rule can stay frozen."""

    def run(self, record: Record) -> int:
        if self.function is None:
            return 0
        try:
            return 1 if self.function(record) else 0
        except _RULE_ERRORS as error:
            logger.debug(f"Rule {self.rule.name} raised on a record: {error!r}")
            self.failures[0] += 1
            return 0


def _compile(rule: Rule, scope: dict[str, object]) -> _Compiled:
    try:
        tree = _checked(rule.expression)
        function = eval(compile(tree, f"<rule {rule.name}>", "eval"), scope)  # noqa: S307  # checked, restricted builtins
    except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
        # Too deeply nested a rule overflows the parser or the compiler.
        logger.warning(f"Rule {rule.name} does not compile: {error}")
        return _Compiled(rule, None, str(error))
    return _Compiled(rule, function, None)


def _checked(expression: str) -> ast.Expression:
    """The parsed rule, if it is one lambda of one argument using only allowed names.

    Raises:
        SyntaxError: The rule is not valid Python.
        ValueError: The rule is not one lambda, or uses a name it may not.
    """
    tree = ast.parse(expression.strip(), mode="eval")
    body = tree.body
    if not isinstance(body, ast.Lambda) or len(body.args.args) != 1:
        raise ValueError("a rule must be a lambda of one argument")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr not in _ATTRIBUTES:
            raise ValueError(f"a rule may not use .{node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"a rule may not use {node.id}")
        if isinstance(node, ast.NamedExpr):
            raise ValueError("a rule may not use :=")
        if isinstance(node, ast.Lambda) and (
            node.args.defaults or any(d is not None for d in node.args.kw_defaults)
        ):
            raise ValueError("a rule's lambda may not have default arguments")
    return tree
