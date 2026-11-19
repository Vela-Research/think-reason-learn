"""Labelling cases: your labels where you give them, a teacher's elsewhere.

The teacher applies the written policy to one case per call, the policy sent
first and marked for prompt caching. Your labels always win: a labelled case is
never sent to the teacher. With two draws the teacher judges each case twice,
and a case it judged two ways counts half in the fit.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_scalar
from pydantic import BaseModel

from think_reason_learn.core._data import LabelInput, TextInput, take, to_texts
from think_reason_learn.exceptions import DataError, NoOutputError
from think_reason_learn.llm import (
    Endpoint,
    LLMSession,
    Models,
    Usage,
    resolve_models,
    run_async,
    run_sync,
)

from ._prompts import TEACHER_INSTRUCTIONS, case_prompt, policy_block
from ._types import Policy, Teacher, TeacherVerdict

logger = logging.getLogger(__name__)

type Codes = npt.NDArray[np.intp]
type TeacherSpec = tuple[Endpoint, ...] | Teacher
"""A resolved teacher: models to call, or your own function."""

UNLABELLED: Final = -1
"""The code of a case with no outcome."""
DISAGREED_WEIGHT: Final = 0.5
"""The weight of a case the teacher judged two ways."""
UNJUDGED_SHARE: Final = 0.8
"""A criterion the teacher could not judge on more than this share of the cases
it labelled is reported, and its checks are dropped."""
MIN_JUDGED: Final = 30
"""Cases the teacher must have labelled before a criterion is dropped: a share
of a handful of cases says too little to give up on a criterion."""
COLUMNS: Final = (
    "outcome",
    "source",
    "confidence",
    "why",
    "criteria",
    "unjudged",
    "second",
    "failure",
    "weight",
)
"""Per case: its outcome (``None`` if it has none), ``"gold"`` for your label or
``"teacher"``, the teacher's confidence, reason, deciding criteria and criteria
it could not judge, its second draw's outcome, why the teacher gave no verdict
(``None`` if it gave one), and the case's weight."""
_POOL_BATCH: Final = 20
TEACHER_REPLY_TOKENS: Final = 400
"""The most a teacher's verdict writes, thinking aside: a short JSON object. It
sizes what each call holds of the cost cap."""


def outcome_codes(policy: Policy, y: LabelInput | None, n: int) -> Codes:
    """Each case's outcome index, lowest 0; ``UNLABELLED`` for a missing label.

    Raises:
        DataError: ``y`` has the wrong length or a label that is no outcome.
    """
    if y is None:
        return np.full(n, UNLABELLED, dtype=np.intp)
    values: list[object] = np.asarray(y, dtype=object).ravel().tolist()
    if len(values) != n:
        raise DataError(f"X has {n} cases and y has {len(values)} labels.")
    index = {name: k for k, name in enumerate(policy.names)}
    codes = np.full(n, UNLABELLED, dtype=np.intp)
    unknown: set[str] = set()
    for i, value in enumerate(values):
        if is_scalar(value) and bool(pd.isna([value])[0]):  # None, NaN, pd.NA
            continue
        if (code := index.get(str(value))) is None:
            unknown.add(str(value))
        else:
            codes[i] = code
    if unknown:
        raise DataError(
            f"y has labels {sorted(unknown)} that are not the policy's outcomes "
            f"{list(policy.names)}; use None for a case with no label."
        )
    return codes


def resolve_teacher(teacher: Models | Teacher) -> TeacherSpec:
    """The teacher's models, checked to be ones that write; or your function."""
    if callable(teacher):
        return teacher
    return resolve_models(teacher, param="teacher", task="generate")


async def label_rows(
    session: LLMSession,
    policy: Policy,
    texts: Sequence[str],
    codes: Codes,
    teacher: TeacherSpec | None,
    *,
    draws: int,
) -> pd.DataFrame:
    """One row per case, ``COLUMNS``: your labels kept, the rest the teacher's."""
    asked = np.flatnonzero(codes == UNLABELLED)
    if len(asked) and teacher is None:
        raise DataError(
            f"{len(asked)} cases have no label: give teacher=... to label them, or "
            "a label for every case."
        )
    rows: list[list[object]] = [
        [policy.names[c], "gold", 1.0, "", (), (), None, None, 1.0]
        if c != UNLABELLED
        else [None, None, math.nan, "", (), (), None, None, 0.0]
        for c in codes.tolist()
    ]
    if teacher is not None and len(asked):
        cases = take(texts, asked)
        first = await verdicts(session, teacher, policy, cases, draw=0)
        second = (
            await verdicts(session, teacher, policy, cases, draw=1)
            if draws > 1
            else [None] * len(cases)
        )
        for i, one, two in zip(asked.tolist(), first, second, strict=True):
            rows[i] = _teacher_row(one, two)
        failures = [v for v in first if isinstance(v, str)]
        if len(failures) == len(cases):
            raise NoOutputError(
                f"The teacher gave no usable verdict for any of the {len(cases)} "
                f"cases. {most_common(failures)}",
                reasons=tuple(failures),
            )
        if failures:
            logger.warning(
                f"The teacher gave no usable verdict for {len(failures)} of "
                f"{len(cases)} cases; they are left out of the fit (see the "
                f"failure column). {most_common(failures)}"
            )
    return pd.DataFrame(rows, columns=pd.Index(COLUMNS))


def most_common(failures: Sequence[str]) -> str:
    """The most common of ``failures``, and how often it came."""
    reason, count = Counter(failures).most_common(1)[0]
    return f"Most common reason ({count} of {len(failures)}): {reason}"


async def verdicts(
    session: LLMSession,
    teacher: TeacherSpec,
    policy: Policy,
    cases: Sequence[str],
    *,
    draw: int,
) -> list[TeacherVerdict | str]:
    """The teacher's verdict on each case, or why it gave none usable."""
    models: tuple[Endpoint, ...] = () if callable(teacher) else teacher

    async def one(case: str) -> TeacherVerdict | str:
        if callable(teacher):
            return _checked(await asyncio.to_thread(teacher, policy, case), policy)
        try:
            reply = await session.generate(
                models,
                case_prompt(case),
                role="teacher",
                schema=_Verdict,
                instructions=TEACHER_INSTRUCTIONS,
                prefix=policy_block(policy),
                draw=draw,
                reply_tokens=TEACHER_REPLY_TOKENS,
            )
        except NoOutputError as error:
            return "; ".join(error.reasons) or str(error)
        return _verdict(reply, policy)

    try:
        async with asyncio.TaskGroup() as group:  # the first error cancels the rest
            tasks = [group.create_task(one(case)) for case in cases]
    except BaseExceptionGroup as errors:
        raise errors.exceptions[0] from None
    return [task.result() for task in tasks]


def unjudged_shares(labels: pd.DataFrame, policy: Policy) -> dict[str, float]:
    """Per criterion, the share of the teacher's cases it could not judge."""
    judged = labels[(labels["source"] == "teacher") & labels["outcome"].notna()]
    if not len(judged):
        return {}
    lists: list[tuple[str, ...]] = judged["unjudged"].tolist()
    return {
        c.id: sum(c.id in unjudged for unjudged in lists) / len(lists)
        for c in policy.criteria
    }


def label_cases(
    policy: Policy,
    X: TextInput,
    teacher: Models | Teacher,
    *,
    y: LabelInput | None = None,
    draws: int = 1,
    pool: TextInput | None = None,
    min_per_outcome: int = 0,
    random_state: int | None = 42,
    usage: Usage | None = None,
) -> pd.DataFrame:
    """Label cases by a written policy: your labels kept, the teacher's elsewhere.

    The teacher judges each case with no label in ``y``, one case per call.
    With ``pool`` and ``min_per_outcome``, an outcome with fewer labels than
    that is topped up: shuffled cases of the pool are labelled in batches, and
    those of a short outcome kept, until each has enough or the pool runs out.
    Topping up over-represents an outcome, so every case gets a sampling
    weight that undoes it: ``n / (n + m)`` for a case of an outcome topped up
    from ``m`` pool cases labelled, with ``n`` the cases of ``X`` labelled.

    Model calls follow the settings in effect (``config_context``): the reply
    cache, and ``max_cost_usd`` for the whole call.

    Args:
        policy: The policy to apply.
        X: The cases: one text each, or a DataFrame (rows as ``column: value``).
        teacher: Reasoning model(s) that apply the policy, or your own function
            ``(policy, case_text) -> TeacherVerdict``.
        y: Your labels, outcome names, with ``None`` for a case to label.
        draws: 2 asks the teacher twice per case: the second outcome is in
            ``second``, and a case judged two ways weighs ``0.5``.
        pool: More cases to top up short outcomes from.
        min_per_outcome: The fewest labels each outcome should have.
        random_state: Seed for the order the pool is labelled in.
        usage: Where to record the calls; read its ``cost_usd`` afterwards.

    Returns:
        One row per case of ``X``, then per pool case kept: ``case`` (its text),
        the columns ``outcome``, ``source`` (``"gold"`` or ``"teacher"``),
        ``confidence``, ``why``, ``criteria``, ``unjudged``, ``second``,
        ``failure`` (why the teacher gave no verdict: the provider's error, an
        empty or cut-off reply, a reply that is not the JSON asked for, or an
        outcome the policy does not have), ``sampling_weight``, ``weight``
        (sampling weight times half for a case judged two ways) and ``pool``.

    Raises:
        NoOutputError: The teacher gave no usable verdict for any case; the
            message names the most common reason, and ``reasons`` has each.
    """
    return run_sync(
        alabel_cases(
            policy,
            X,
            teacher,
            y=y,
            draws=draws,
            pool=pool,
            min_per_outcome=min_per_outcome,
            random_state=random_state,
            usage=usage,
        )
    )


async def alabel_cases(
    policy: Policy,
    X: TextInput,
    teacher: Models | Teacher,
    *,
    y: LabelInput | None = None,
    draws: int = 1,
    pool: TextInput | None = None,
    min_per_outcome: int = 0,
    random_state: int | None = 42,
    usage: Usage | None = None,
) -> pd.DataFrame:
    """Label cases from async code. See ``label_cases``."""
    texts, _ = to_texts(X)
    codes = outcome_codes(policy, y, len(texts))
    spec = resolve_teacher(teacher)
    _check_draws(draws)
    extra = to_texts(pool)[0] if pool is not None else []
    rng = np.random.default_rng(random_state)

    async def work() -> pd.DataFrame:
        session = LLMSession(usage=usage)
        labels = await label_rows(session, policy, texts, codes, spec, draws=draws)
        labels.insert(0, "case", texts)
        return await _topped_up(
            session, policy, labels, extra, spec, draws, min_per_outcome, rng
        )

    return await run_async(work())


async def _topped_up(
    session: LLMSession,
    policy: Policy,
    labels: pd.DataFrame,
    pool: Sequence[str],
    teacher: TeacherSpec,
    draws: int,
    min_per_outcome: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """``labels`` with pool cases of short outcomes added, and sampling weights."""
    outcomes: list[object] = labels["outcome"].tolist()
    short: dict[str, int] = {
        name: min_per_outcome - outcomes.count(name)
        for name in policy.names
        if outcomes.count(name) < min_per_outcome
    }
    screened = dict.fromkeys(policy.names, 0)
    kept: list[pd.DataFrame] = []
    order: list[int] = rng.permutation(len(pool)).tolist()
    while short and order:
        size = max(_POOL_BATCH, sum(short.values()) * len(policy.names))
        batch, order = [pool[i] for i in order[:size]], order[size:]
        for name in short:
            screened[name] += len(batch)
        codes = np.full(len(batch), UNLABELLED, dtype=np.intp)
        rows = await label_rows(session, policy, batch, codes, teacher, draws=draws)
        rows.insert(0, "case", batch)
        found: list[object] = rows["outcome"].tolist()
        kept.append(rows[[o in short for o in found]])
        short = {name: n - found.count(name) for name, n in short.items()}
        short = {name: n for name, n in short.items() if n > 0}
    if short:
        logger.warning(
            f"The pool ran out before every outcome had {min_per_outcome} labels: "
            f"{short} short."
        )
    labelled = int(labels["outcome"].notna().sum())
    combined = pd.concat(
        [labels.assign(pool=False), *(k.assign(pool=True) for k in kept)],
        ignore_index=True,
    )
    sampling = [
        labelled / (labelled + screened[o]) if isinstance(o, str) and labelled else 1.0
        for o in combined["outcome"].tolist()
    ]
    weights = combined.pop("weight")
    combined["sampling_weight"] = sampling
    combined["weight"] = weights * combined["sampling_weight"]
    return combined[["case", *COLUMNS[:-1], "sampling_weight", "weight", "pool"]]


def _teacher_row(
    first: TeacherVerdict | str, second: TeacherVerdict | str | None
) -> list[object]:
    if isinstance(first, str):
        return [None, "teacher", math.nan, "", (), (), None, first, 0.0]
    again = second.outcome if isinstance(second, TeacherVerdict) else None
    weight = DISAGREED_WEIGHT if again is not None and again != first.outcome else 1.0
    return [
        first.outcome,
        "teacher",
        first.confidence,
        first.why,
        first.criteria,
        first.unjudged,
        again,
        None,
        weight,
    ]


def _verdict(reply: _Verdict, policy: Policy) -> TeacherVerdict | str:
    """The reply as a verdict, its outcome matched without regard to case."""
    by_name = {name.lower(): name for name in policy.names}
    outcome = by_name.get(reply.outcome.strip().lower())
    if outcome is None:
        return (
            f"the outcome {reply.outcome!r} is not one of the policy's "
            f"{list(policy.names)}"
        )
    ids = {c.id for c in policy.criteria}
    return TeacherVerdict(
        outcome=outcome,
        confidence=min(max(reply.confidence, 0.0), 1.0),
        criteria=tuple(c for c in reply.criteria if c in ids),
        unjudged=tuple(c for c in reply.unjudged if c in ids),
        why=reply.why.strip(),
    )


def _checked(verdict: TeacherVerdict, policy: Policy) -> TeacherVerdict:
    """Your teacher's verdict, which must name an outcome of the policy."""
    if not isinstance(verdict, TeacherVerdict) or verdict.outcome not in policy.names:  # pyright: ignore[reportUnnecessaryIsInstance]  # your function may return anything
        raise DataError(
            f"The teacher returned {verdict!r}; it must return a TeacherVerdict "
            f"whose outcome is one of {list(policy.names)}."
        )
    return verdict


def _check_draws(draws: int) -> None:
    if draws not in (1, 2):
        raise ValueError(f"draws must be 1 or 2, got {draws}.")


class _Verdict(BaseModel):
    outcome: str
    confidence: float = 1.0
    criteria: list[str] = []
    unjudged: list[str] = []
    why: str = ""
