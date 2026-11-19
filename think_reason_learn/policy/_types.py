"""A written policy: its ordered outcomes and criteria, and verdicts on cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal, get_args

type Kind = Literal["positive", "negative", "neutral"]
"""Which way a criterion or check points when it holds: to a higher outcome
(``"positive"``), a lower one (``"negative"``), or either (``"neutral"``)."""

KINDS: Final[tuple[Kind, ...]] = get_args(Kind.__value__)
MIN_OUTCOMES: Final = 2
MAX_OUTCOMES: Final = 4


@dataclass(frozen=True, slots=True)
class Outcome:
    """One outcome of a decision.

    Args:
        name: The label, e.g. ``"YES"``. Labels in ``y`` use it.
        meaning: What it means, as a statement about a case, e.g. ``"the
            recipient will love the gift"``. Checks and the teacher read it.
    """

    name: str
    meaning: str = ""

    def __post_init__(self) -> None:
        """Check the name."""
        if not self.name.strip():
            raise ValueError("An outcome needs a name.")

    @property
    def described(self) -> str:
        """Its meaning, or its name if it has none."""
        return self.meaning.strip() or self.name


@dataclass(frozen=True, slots=True)
class Criterion:
    """A criterion of a policy, or a yes/no check of your own.

    Args:
        id: Your id for it, e.g. ``"C1"``; explanations group reasons by it.
        text: The criterion, e.g. ``"The gift matches a hobby the recipient
            has."``. As a check of your own it is asked as written, yes or no.
        kind: ``"positive"`` if meeting it points to a higher outcome,
            ``"negative"`` if to a lower one, ``"neutral"`` if either.
    """

    id: str
    text: str
    kind: Kind = "neutral"

    def __post_init__(self) -> None:
        """Check the id, text and kind."""
        if not self.id.strip() or not self.text.strip():
            raise ValueError(f"A criterion needs an id and a text, got {self!r}.")
        if self.kind not in KINDS:
            raise ValueError(
                f"Criterion {self.id!r} has kind {self.kind!r}; use one of {KINDS}."
            )


@dataclass(frozen=True, slots=True)
class Policy:
    """A written policy that puts each case in one of 2 to 4 ordered outcomes.

    Args:
        text: The policy in your own words. May be empty when ``criteria`` are
            given.
        outcomes: The outcomes from lowest to highest, e.g. ``NO < MAYBE < YES``.
        criteria: What the policy weighs, each with your own id.
    """

    text: str = ""
    outcomes: tuple[Outcome, ...] = ()
    criteria: tuple[Criterion, ...] = ()

    def __post_init__(self) -> None:
        """Check the outcomes and criteria, and keep them as tuples."""
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "criteria", tuple(self.criteria))
        names = [o.name for o in self.outcomes]
        if not MIN_OUTCOMES <= len(names) <= MAX_OUTCOMES:
            raise ValueError(
                f"A policy needs {MIN_OUTCOMES} to {MAX_OUTCOMES} outcomes, lowest "
                f"first; got {names}."
            )
        if len(set(names)) != len(names):
            raise ValueError(f"Outcome names must differ, got {names}.")
        ids = [c.id for c in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Criterion ids must differ, got {ids}.")
        if not self.text.strip() and not self.criteria:
            raise ValueError("A policy needs a text, criteria, or both.")

    @property
    def names(self) -> tuple[str, ...]:
        """The outcome names, lowest first."""
        return tuple(o.name for o in self.outcomes)


@dataclass(frozen=True, slots=True)
class TeacherVerdict:
    """A teacher's judgement of one case under the policy.

    Args:
        outcome: The outcome's name.
        confidence: How sure the teacher is, from 0 to 1.
        criteria: Ids of the criteria that decided the outcome.
        unjudged: Ids of the criteria the case's text gave no way to judge.
        why: The reason, in a sentence or two.
    """

    outcome: str
    confidence: float = 1.0
    criteria: tuple[str, ...] = ()
    unjudged: tuple[str, ...] = ()
    why: str = ""


type Teacher = Callable[[Policy, str], TeacherVerdict]
"""Your own teacher: the outcome of one case's text under the policy."""


@dataclass(frozen=True, slots=True)
class Mistake:
    """A case the current model gets wrong, shown to the writer of new checks.

    Args:
        case: The case's text.
        outcome: Its outcome, from your label or the teacher's.
        predicted: The outcome the model gave it.
        why: The teacher's reason for ``outcome``, if there is one.
    """

    case: str
    outcome: str
    predicted: str
    why: str = ""


@dataclass(frozen=True, slots=True)
class Explanation:
    """Why the model put one case where it did.

    The model scores a case in points: ``baseline`` is the score of a case
    whose every answer is at its training average, and each criterion (or
    check of your own, or written check) adds or takes away points.
    ``baseline + sum(reasons.values()) == points``, exactly.

    Args:
        outcome: The predicted outcome.
        probabilities: The probability of each outcome, lowest first.
        points: The case's score.
        baseline: The score of an average training case.
        cutoffs: For each outcome above the lowest, the score from which the
            case more likely than not reaches it.
        reasons: Points by criterion id, largest effect first.
        checks: Points by check id, in the order of ``checks_``.
        unassessed: Criteria that could not be judged for this case: their
            checks went unanswered, or the teacher could not judge them at fit.
    """

    outcome: str
    probabilities: Mapping[str, float]
    points: float
    baseline: float
    cutoffs: Mapping[str, float]
    reasons: Mapping[str, float]
    checks: Mapping[str, float] = field(default_factory=dict[str, float])
    unassessed: tuple[str, ...] = ()
