"""Yes/no checks: built from a policy's criteria by templates, or given, or written.

A check's id is ``"<criterion id>.<form>"``, with the boundary appended for the
forms asked once per boundary (``"C1.reach2"``). Checks of your own and written
checks keep their own id. Each check belongs to a group, the criterion it came
from (or its own id), and explanations add up its points by group.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
import pandas as pd

from ._prompts import applies_check, met_check, reach_check, whole_policy
from ._types import KINDS, Criterion, Kind, Policy

type Form = Literal["reach", "applies", "met", "policy"]
"""How a criterion is asked: whether judging by it alone a case reaches each
outcome (once per boundary), whether it applies, or whether it is met.
``"policy"`` asks the first of these once per boundary for the whole policy."""

type Signs = Literal["criteria", "free"]
"""How weights are bounded: by what each check says, or not at all."""

FORMS: Final[tuple[Form, ...]] = get_args(Form.__value__)
SIGNS: Final[tuple[Signs, ...]] = get_args(Signs.__value__)
POLICY_GROUP: Final = "policy"
"""The group of the checks that judge by the whole policy."""
OWN: Final = "own"
WRITTEN: Final = "written"
COLUMNS: Final = ("id", "group", "form", "kind", "question")


@dataclass(frozen=True, slots=True)
class Check:
    """One yes/no question asked about every case."""

    id: str
    group: str
    form: str
    """A template's form, ``"own"`` for a check of yours, or ``"written"``."""
    kind: Kind
    question: str


def policy_checks(
    policy: Policy, templates: Sequence[Form], *, skip: Collection[str] = ()
) -> list[Check]:
    """The checks ``templates`` build from the criteria not in ``skip``.

    A policy with no criteria is judged as a whole (the ``"policy"`` form),
    whatever the templates.
    """
    outcomes = policy.outcomes[1:]  # boundary k separates outcome k - 1 from k
    checks: list[Check] = []
    for criterion in policy.criteria:
        if criterion.id in skip:
            continue
        cid, kind = criterion.id, criterion.kind
        for form in templates:
            if form == "reach":
                checks += [
                    Check(
                        f"{cid}.reach{k}",
                        cid,
                        form,
                        kind,
                        reach_check(criterion.text, o),
                    )
                    for k, o in enumerate(outcomes, 1)
                ]
            elif form == "applies":
                checks.append(
                    Check(f"{cid}.applies", cid, form, kind, applies_check(criterion))
                )
            elif form == "met":
                checks.append(
                    Check(f"{cid}.met", cid, form, kind, met_check(criterion))
                )
    if "policy" in templates or not policy.criteria:
        text = whole_policy(policy)
        checks += [
            Check(
                f"{POLICY_GROUP}.reach{k}",
                POLICY_GROUP,
                "policy",
                "positive",
                reach_check(text, o),
            )
            for k, o in enumerate(outcomes, 1)
        ]
    return checks


def own_checks(criteria: Sequence[Criterion], form: str) -> list[Check]:
    """Checks asked as written, each its own group."""
    return [Check(c.id, c.id, form, c.kind, c.text) for c in criteria]


def weight_bounds(checks: Sequence[Check], signs: Signs) -> list[tuple[float, float]]:
    """The weights each check may take.

    With ``"criteria"``: a check that a case reaches an outcome may only raise
    it; one that a criterion is met, and a check of your own or a written one,
    follows its kind (positive: raise only, negative: lower only, neutral:
    either way); one that a criterion applies may go either way.
    """
    return [_bounds(check, signs) for check in checks]


def to_frame(checks: Sequence[Check]) -> pd.DataFrame:
    """The checks as a table, one row each."""
    return pd.DataFrame(
        [[getattr(c, name) for name in COLUMNS] for c in checks],
        columns=pd.Index(COLUMNS),
    )


def from_frame(frame: pd.DataFrame) -> list[Check]:
    """The checks of a table made by ``to_frame``."""
    return [
        Check(str(i), str(g), str(f), _kind(k), str(q))
        for i, g, f, k, q in frame[list(COLUMNS)].itertuples(index=False, name=None)
    ]


def _bounds(check: Check, signs: Signs) -> tuple[float, float]:
    free = (-np.inf, np.inf)
    if signs == "free" or check.form == "applies":
        return free
    if check.form in ("reach", "policy"):
        return (0.0, np.inf)
    return {"positive": (0.0, np.inf), "negative": (-np.inf, 0.0)}.get(check.kind, free)


def _kind(value: object) -> Kind:
    for kind in KINDS:
        if value == kind:
            return kind
    raise ValueError(f"Unknown check kind {value!r}.")
