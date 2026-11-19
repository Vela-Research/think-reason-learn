"""PolicyClassifier: a written policy, labelled cases or both, as ordered outcomes.

Fitting runs four steps, each using the library's sessions, cache and cost cap:

1. Labels: your labels are kept; a teacher applies the policy to the rest.
2. Checks: yes/no checks built from the policy's criteria by templates, plus
   checks of your own; a criterion the teacher could not judge on most cases
   is dropped.
3. Written checks (with a writer): a model reads the cases the current checks
   get wrong, out of fold, and writes new checks; repeated for each round.
4. Weights: the answerer answers every check for every labelled case, and an
   ordinal model (``core.fit_ordinal``) weighs them, one weight per check.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection, Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Final, Literal, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._classifier import Labels, Proba, TRLClassifier, Work
from think_reason_learn.core._data import Classes, LabelInput, TextInput, take, to_texts
from think_reason_learn.core._ordinal import (
    CS,
    OrdinalScorer,
    balanced_accuracy,
    balanced_cuts,
    boundary_aucs,
    fit_ordinal,
    mean_auc,
    quadratic_kappa,
)
from think_reason_learn.core._sklearn import require_fitted
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import (
    LLM,
    QUESTION_TOKENS,
    Endpoint,
    LLMSession,
    Models,
    UsageRecord,
    config_context,
    estimate_answers,
    estimate_generations,
    mean_tokens,
    resolve_models,
    run_async,
    run_sync,
    text_tokens,
)

from ._checks import (
    FORMS,
    OWN,
    POLICY_GROUP,
    SIGNS,
    WRITTEN,
    Check,
    Form,
    Signs,
    from_frame,
    own_checks,
    policy_checks,
    to_frame,
    weight_bounds,
)
from ._labels import (
    UNJUDGED_SHARE,
    UNLABELLED,
    TeacherSpec,
    label_rows,
    outcome_codes,
    resolve_teacher,
    unjudged_shares,
    verdicts,
)
from ._prompts import TEACHER_INSTRUCTIONS, policy_block
from ._types import Criterion, Explanation, Mistake, Policy, Teacher
from ._writer import CHECKS_PER_ROUND, written_checks

logger = logging.getLogger(__name__)

type Decision = Literal["likely", "balanced"]
"""How probabilities become an outcome: the most probable one, or cut-offs on
the expected outcome tuned out of fold for balanced accuracy."""
type Floats = npt.NDArray[np.float64]

DECISIONS: Final[tuple[Decision, ...]] = ("likely", "balanced")
POINTS: Final = 10.0
"""Points per unit of the ordinal model's score, so explanations read as whole
numbers."""
LOPSIDED_SHARE: Final = 0.9
"""Above this share of the training labels on one outcome, fit warns."""
_SHOWN_TOKENS: Final = 25
"""Cases the writer is shown in full, for estimates."""


class PolicyClassifier(TRLClassifier):
    """A decision with 2 to 4 ordered outcomes, from a written policy, labels or both.

    Each case gets yes/no checks built from the policy's criteria (and any of
    your own), a cheap answerer answers them, and an ordinal model weighs the
    answers, one weight per check. The labels it learns from are yours where
    you give them and a teacher's elsewhere: a strong model applying the
    policy. With a writer, a model also writes new checks from the cases the
    current ones get wrong. Every prediction can be explained in points, by
    criterion.

    Args:
        policy: The policy: its text, its ordered outcomes and its criteria.
        answerer: Model(s) that answer the checks. Defaults to Jev.
        teacher: Reasoning model(s) that label the cases with no label by
            applying the policy, or your own function ``(policy, case_text) ->
            TeacherVerdict``. Needed when a case has no label.
        writer: Reasoning model(s) that write new checks. ``None`` writes none.
        templates: The checks built from each criterion (see ``Form``):
            ``"reach"`` (once per boundary), ``"applies"``, ``"met"``, and
            ``"policy"`` (the whole policy, once per boundary); all four by
            default. A policy with no criteria is judged as a whole.
        writer_rounds: Rounds of written checks; ``None`` is 1 with a writer
            and 0 without.
        checks: Yes/no checks of your own, asked as written: ``Criterion(id,
            text, kind)``.
        signs: ``"criteria"`` bounds each weight by what its check says (a
            positive check may only raise the outcome, a negative one only
            lower it; see ``weight_bounds``); ``"free"`` does not.
        decision: ``"likely"`` predicts the most probable outcome;
            ``"balanced"`` cuts the expected outcome where balanced accuracy is
            best, out of fold, which predicts rare outcomes more often.
        cv: Folds for choosing the regularisation and for the report.
        Cs: Inverse L2 strengths to choose from.
        teacher_draws: 2 asks the teacher twice per case; a case it judges two
            ways weighs half, and the report gives its consistency.
        teacher_check: How many of your labelled cases the teacher also
            judges, to measure how well it follows your labels
            (``report_["policy_vs_labels"]``). Only measured: your labels stay
            as they are and its verdicts are not learned from. 0 for none.
        max_cost_usd: A cost cap for each fit and predict, in US dollars; the
            one in ``config_context`` applies when ``None``.
        random_state: Seed for the folds.
        verbose: Show progress when above 0.
    """

    def __init__(
        self,
        policy: Policy | None = None,
        answerer: Models = "jev",
        *,
        teacher: Models | Teacher | None = None,
        writer: Models | None = None,
        templates: tuple[Form, ...] = ("reach", "applies", "met", "policy"),
        writer_rounds: int | None = None,
        checks: tuple[Criterion, ...] = (),
        signs: Signs = "criteria",
        decision: Decision = "likely",
        cv: int = 5,
        Cs: tuple[float, ...] = CS,
        teacher_draws: int = 1,
        teacher_check: int = 50,
        max_cost_usd: float | None = None,
        random_state: int | None = 42,
        verbose: int = 0,
    ) -> None:
        self.policy = policy
        self.answerer = answerer
        self.teacher = teacher
        self.writer = writer
        self.templates = templates
        self.writer_rounds = writer_rounds
        self.checks = checks
        self.signs = signs
        self.decision = decision
        self.cv = cv
        self.Cs = Cs
        self.teacher_draws = teacher_draws
        self.teacher_check = teacher_check
        self.max_cost_usd = max_cost_usd
        self.random_state = random_state
        self.verbose = verbose

    templates: tuple[Form, ...]
    signs: Signs
    decision: Decision
    labels_: pd.DataFrame
    checks_: pd.DataFrame
    answers_: pd.DataFrame
    scorer_: OrdinalScorer
    cut_points_: Floats | None
    unjudged_: dict[str, float]
    policy_vs_labels_: dict[str, float | int | None] | None
    dropped_criteria_: tuple[str, ...]
    report_: dict[str, object]
    _fit_weights: Floats | None = None
    """The running fit's ``sample_weight``; never saved."""

    def fit(
        self,
        X: TextInput,
        y: LabelInput | None = None,
        sample_weight: npt.ArrayLike | None = None,
    ) -> Self:
        """Fit the model.

        Args:
            X: The cases: one text each, or a DataFrame (rows as ``column:
                value``).
            y: Each case's outcome name, ``None`` for a case the teacher
                labels. With no ``y`` the teacher labels every case.
            sample_weight: A weight per case, 1 each by default. It multiplies
                the half weight of a case the teacher judged two ways. Pass
                ``label_cases(...)["weight"]`` to undo a pool top-up.

        Returns:
            The fitted model.
        """
        return run_sync(self._weighted_fit(X, y, sample_weight))

    async def afit(
        self,
        X: TextInput,
        y: LabelInput | None = None,
        sample_weight: npt.ArrayLike | None = None,
    ) -> Self:
        """Fit the model, from async code. See ``fit``."""
        return await run_async(self._weighted_fit(X, y, sample_weight))

    def explain(self, X: TextInput) -> list[Explanation]:
        """Why each case gets its outcome: points by criterion, and what is unknown.

        Answers the checks for ``X`` (those with a weight), as ``predict`` does.
        """
        return self._explanations(self._run(X, "explaining", self._answered))

    async def aexplain(self, X: TextInput) -> list[Explanation]:
        """Explain the cases, from async code. See ``explain``."""
        require_fitted(self)
        answers = await run_async(self._arun(X, "explaining", self._answered))
        return self._explanations(answers)

    def refit(
        self,
        labels: LabelInput | None = None,
        *,
        mute: Collection[str] = (),
        lock: Collection[str] = (),
    ) -> Self:
        """Weigh the checks again from the answers kept at fit: no model is called.

        Args:
            labels: New outcomes for the training cases, in ``labels_`` order
                (``None`` for a case to leave out). A changed label counts as
                yours (``"gold"``). By default the current ones.
            mute: Check or criterion ids to give no weight.
            lock: Check or criterion ids whose weights stay as fitted.

        Returns:
            The refitted model.

        Raises:
            ValueError: An id is unknown, or both muted and locked.
        """
        require_fitted(self)
        policy = self._policy()
        frame = self.labels_
        if labels is not None:
            codes = outcome_codes(policy, labels, len(frame))
            frame = _relabelled(frame, policy, codes)
        checks = from_frame(self.checks_)
        fixed = self._fixed(checks, mute, lock)
        previous, self.labels_ = self.labels_, frame
        try:
            self._weigh(
                checks,
                self.answers_.to_numpy(dtype=float),
                fixed,
                muted=sorted(mute),
                locked=sorted(lock),
                unjudged=self.unjudged_,
            )
        except BaseException:
            self.labels_ = previous  # nothing else was changed yet
            raise
        return self

    def write_instructions(self) -> str:
        """Not used: a ``PolicyClassifier`` follows its written policy instead."""
        raise NotImplementedError(
            "PolicyClassifier follows its written policy and takes no instructions."
        )

    # The fit.

    async def _fit(self, X: TextInput, y: LabelInput) -> Self:
        with self._cost_cap():
            return await super()._fit(X, y)

    async def _weighted_fit(
        self, X: TextInput, y: LabelInput | None, sample_weight: npt.ArrayLike | None
    ) -> Self:
        """``_fit`` with ``sample_weight``, checked first and held for ``_afit``."""
        n = len(to_texts(X)[0])
        self._fit_weights = _checked_weights(sample_weight, n)
        try:
            return await self._fit(X, [None] * n if y is None else y)
        finally:
            del self._fit_weights

    async def _arun[T](self, X: TextInput, doing: str, work: Work[T]) -> T:
        with self._cost_cap():
            return await super()._arun(X, doing, work)

    def _encode_labels(self, texts: list[str], y: LabelInput) -> tuple[Classes, Labels]:
        policy = self._policy()
        codes = outcome_codes(policy, y, len(texts))
        if (codes == UNLABELLED).any() and self.teacher is None:
            raise DataError(
                f"{int((codes == UNLABELLED).sum())} cases have no label: give "
                "teacher=... to label them by the policy, or a label for every case."
            )
        return np.asarray(policy.names), codes

    def _task(self) -> str:
        return self._policy().text

    def _check_classes(self, classes: Classes) -> None:
        """Check every setting before any call."""
        policy = self._policy()
        if unknown := set(self.templates) - set(FORMS):
            raise ValueError(f"templates {sorted(unknown)} are not among {FORMS}.")
        if self.signs not in SIGNS:
            raise ValueError(f"signs must be one of {SIGNS}, got {self.signs!r}.")
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}.")
        if self.teacher_draws not in (1, 2):
            raise ValueError(f"teacher_draws must be 1 or 2, got {self.teacher_draws}.")
        if self.teacher_check < 0:
            raise ValueError(f"teacher_check must be 0 or more: {self.teacher_check}.")
        if self.cv < 2 or not self.Cs or min(self.Cs) <= 0:
            raise ValueError("cv needs 2 or more folds, and Cs values above 0.")
        if self.writer_rounds is not None and self.writer_rounds < 0:
            raise ValueError(f"writer_rounds must be 0 or more: {self.writer_rounds}.")
        checks = self._starting_checks(policy, ())
        if not checks and not self._rounds():
            raise ValueError("No checks to ask: give templates=, checks= or a writer.")
        _check_ids(checks, policy)
        self._models("answerer", "answer")
        self._writer()
        if self.teacher is not None:
            resolve_teacher(self.teacher)

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        policy = self._policy()
        answerer = self._models("answerer", "answer")
        teacher = resolve_teacher(self.teacher) if self.teacher is not None else None
        labels = await label_rows(
            session, policy, texts, y, teacher, draws=self.teacher_draws
        )
        weights = self._fit_weights
        labels.insert(len(labels.columns) - 1, "sample_weight", 1.0)
        if weights is not None:
            labels["sample_weight"] = weights
            labels["weight"] = labels["weight"] * weights
        self.policy_vs_labels_ = await self._check_teacher(
            session, policy, texts, y, teacher
        )
        shares = unjudged_shares(labels, policy)
        dropped = [cid for cid, share in shares.items() if share > UNJUDGED_SHARE]
        if dropped:
            logger.warning(
                f"The teacher could not judge criteria {dropped} on more than "
                f"{UNJUDGED_SHARE:.0%} of the cases; their checks are dropped."
            )
        checks = self._starting_checks(policy, dropped)
        if not checks and not self._rounds():
            raise DataError("Every criterion was dropped: no checks are left to ask.")
        codes = outcome_codes(policy, labels["outcome"], len(labels))
        rows = np.flatnonzero(codes != UNLABELLED)
        if len(np.unique(codes[rows])) < 2:
            raise DataError(
                "The labelled cases need at least two different outcomes to learn "
                f"from; they have {sorted(set(labels['outcome'].dropna()))}."
            )
        answers = np.full((len(texts), len(checks)), np.nan)
        answers[rows] = await self._ask(session, answerer, take(texts, rows), checks)
        self.labels_ = labels
        self.unjudged_, self.dropped_criteria_ = shares, tuple(dropped)
        for round_ in range(1, self._rounds() + 1):
            written = await self._write(
                session, answerer, texts, rows, checks, answers, round_
            )
            if not written:
                break
            more = np.full((len(texts), len(written)), np.nan)
            more[rows] = await self._ask(session, answerer, take(texts, rows), written)
            checks, answers = [*checks, *written], np.hstack([answers, more])
        await asyncio.to_thread(
            self._weigh, checks, answers, {}, muted=[], locked=[], unjudged=shares
        )

    def _gold_checked(self, y: Labels) -> npt.NDArray[np.intp]:
        """The labelled cases the teacher also judges: a seeded sample of them."""
        gold = np.flatnonzero(y != UNLABELLED)
        if self.teacher is None or not self.teacher_check or not len(gold):
            return np.empty(0, dtype=np.intp)
        rng = np.random.default_rng(self.random_state)
        picked = rng.permutation(gold)[: self.teacher_check]
        return np.sort(picked).astype(np.intp)

    async def _check_teacher(
        self,
        session: LLMSession,
        policy: Policy,
        texts: list[str],
        y: Labels,
        teacher: TeacherSpec | None,
    ) -> dict[str, float | int | None] | None:
        """How often the teacher gives your label, on a sample of your cases."""
        rows = self._gold_checked(y)
        if teacher is None or not len(rows):
            return None
        found = await verdicts(session, teacher, policy, take(texts, rows), draw=0)
        pairs = [
            (int(y[r]), policy.names.index(v.outcome))
            for r, v in zip(rows.tolist(), found, strict=True)
            if v is not None
        ]
        if not pairs:
            return None
        labelled = np.array([gold for gold, _ in pairs], dtype=np.intp)
        judged = np.array([verdict for _, verdict in pairs], dtype=np.intp)
        return {
            "n": len(pairs),
            "exact": float((labelled == judged).mean()),
            "kappa": quadratic_kappa(labelled, judged, len(policy.names)),
        }

    async def _write(
        self,
        session: LLMSession,
        answerer: tuple[Endpoint, ...],
        texts: list[str],
        rows: npt.NDArray[np.intp],
        checks: list[Check],
        answers: Floats,
        round_: int,
    ) -> list[Check]:
        """Checks written from the cases the current ones get wrong, out of fold."""
        policy = self._policy()
        codes = outcome_codes(policy, self.labels_["outcome"], len(texts))[rows]
        _, out_of_fold = await asyncio.to_thread(
            self._fit_scorer, checks, answers[rows], codes, {}, rows
        )
        predicted = _decided(out_of_fold, self._oof_cuts(out_of_fold, codes))
        wrong = np.flatnonzero(predicted != codes)
        # The worst first: the least probability on the right outcome.
        wrong = wrong[np.argsort(out_of_fold[wrong, codes[wrong]], kind="stable")]
        why: list[str] = self.labels_["why"].tolist()
        mistakes = [
            Mistake(
                texts[rows[i]],
                policy.names[int(codes[i])],
                policy.names[int(predicted[i])],
                why[rows[i]],
            )
            for i in wrong.tolist()
        ]
        written = await written_checks(
            session,
            self._writer(),
            policy,
            mistakes,
            current=[c.question for c in checks],
            prefix=f"W{round_}-",
        )
        taken = {c.id for c in checks}
        return [c for c in own_checks(written, WRITTEN) if c.id not in taken]

    async def _ask(
        self,
        session: LLMSession,
        answerer: tuple[Endpoint, ...],
        texts: Sequence[str],
        checks: Sequence[Check],
    ) -> Floats:
        return await yes_probabilities(
            session, answerer, texts, [c.question for c in checks], missing=np.nan
        )

    def _weigh(
        self,
        checks: list[Check],
        answers: Floats,
        fixed: Mapping[int, tuple[float, float]],
        *,
        muted: list[str],
        locked: list[str],
        unjudged: Mapping[str, float],
    ) -> None:
        """Fit the weights on the labelled, answered cases; set what depends on them."""
        policy = self._policy()
        codes = outcome_codes(policy, self.labels_["outcome"], len(self.labels_))
        answered = (
            ~np.isnan(answers).all(axis=1)
            if answers.shape[1]
            else np.ones(len(codes), dtype=bool)
        )
        rows = np.flatnonzero((codes != UNLABELLED) & answered)
        if len(np.unique(codes[rows])) < 2:
            raise DataError("The labelled cases need at least two different outcomes.")
        _warn_if_lopsided(codes[rows], policy)
        scorer, out_of_fold = self._fit_scorer(
            checks, answers[rows], codes[rows], fixed, rows
        )
        self.scorer_ = scorer
        self.cut_points_ = self._oof_cuts(out_of_fold, codes[rows])
        frame = to_frame(checks)
        frame["points"] = POINTS * scorer.coef / scorer.scale
        frame["used"] = scorer.coef != 0
        self.checks_ = frame
        self.answers_ = pd.DataFrame(answers, columns=pd.Index(frame["id"]))
        self.report_ = self._report(
            rows,
            codes[rows],
            out_of_fold,
            checks,
            muted=muted,
            locked=locked,
            unjudged=unjudged,
        )
        if not frame["used"].any():
            logger.warning(
                "Every weight is zero, so the model gives every case the same "
                "outcome. Try more labelled cases or other checks."
            )

    def _fit_scorer(
        self,
        checks: list[Check],
        answers: Floats,
        codes: Labels,
        fixed: Mapping[int, tuple[float, float]],
        rows: npt.NDArray[np.intp],
    ) -> tuple[OrdinalScorer, Floats]:
        bounds = weight_bounds(checks, self.signs)
        bounds = [fixed.get(j, b) for j, b in enumerate(bounds)]
        weights = self.labels_["weight"].to_numpy(dtype=float)[rows]
        return fit_ordinal(
            answers,
            codes,
            n_outcomes=len(self._policy().outcomes),
            bounds=bounds,
            Cs=self.Cs,
            cv=self.cv,
            sample_weight=weights,
            random_state=self.random_state,
        )

    def _oof_cuts(self, out_of_fold: Floats, codes: Labels) -> Floats | None:
        """The ``"balanced"`` cut-offs, tuned on out-of-fold expected outcomes."""
        if self.decision != "balanced":
            return None
        return balanced_cuts(_expected(out_of_fold), codes, out_of_fold.shape[1])

    def _report(
        self,
        rows: npt.NDArray[np.intp],
        codes: Labels,
        out_of_fold: Floats,
        checks: list[Check],
        *,
        muted: list[str],
        locked: list[str],
        unjudged: Mapping[str, float],
    ) -> dict[str, object]:
        """How well the model agrees with its labels, out of fold, and with what."""
        policy = self._policy()
        names = policy.names
        source = self.labels_["source"].to_numpy(dtype=object)
        second = self.labels_["second"].to_numpy(dtype=object)
        outcome = self.labels_["outcome"].to_numpy(dtype=object)
        labelled = self.labels_["outcome"].notna().to_numpy(dtype=bool)
        drawn = (source == "teacher") & labelled & self.labels_["second"].notna()
        predicted = _decided(out_of_fold, self.cut_points_)
        by_source = source[rows]

        def agreement(mask: npt.NDArray[np.bool_]) -> dict[str, object] | None:
            if not mask.any():
                return None
            y, p, proba = codes[mask], predicted[mask], out_of_fold[mask]
            reach = np.cumsum(proba[:, ::-1], axis=1)[:, ::-1][:, 1:]  # P(>= k)
            aucs = boundary_aucs(y, reach, len(names))
            return {
                "cases": int(mask.sum()),
                "exact": float((y == p).mean()),
                "balanced": balanced_accuracy(y, p),
                "kappa": quadratic_kappa(y, p, len(names)),
                "boundary_auc": aucs,
                "mean_boundary_auc": mean_auc(aucs),
            }

        def shares(kind: str) -> dict[str, float] | None:
            picked = outcome[(source == kind) & labelled]
            if not len(picked):
                return None
            return {n: float((picked == n).mean()) for n in names}

        used = self.scorer_.coef != 0
        return {
            "cases": {
                "gold": int((source == "gold").sum()),
                "teacher": int(((source == "teacher") & labelled).sum()),
                "unlabelled": int((~labelled).sum()),
            },
            "out_of_fold": {
                "all": agreement(np.ones(len(rows), dtype=bool)),
                "gold": agreement(by_source == "gold"),
                "teacher": agreement(by_source == "teacher"),
            },
            "outcome_shares": {"gold": shares("gold"), "teacher": shares("teacher")},
            "teacher_consistency": (
                float((second[drawn] == outcome[drawn]).mean()) if drawn.any() else None
            ),
            "unjudged": dict(unjudged),
            "dropped_criteria": list(self.dropped_criteria_),
            "checks": {
                "asked": len(checks),
                "used": int(used.sum()),
                "written": sum(c.form == WRITTEN for c in checks),
            },
            "C": self.scorer_.C,
            "cv_auc": self.scorer_.cv_auc,
            **(
                {"policy_vs_labels": self.policy_vs_labels_}
                if self.policy_vs_labels_ is not None
                else {}
            ),
            "muted": muted,
            "locked": locked,
        }

    def _fixed(
        self, checks: list[Check], mute: Collection[str], lock: Collection[str]
    ) -> dict[int, tuple[float, float]]:
        """Bounds that hold muted weights at 0 and locked ones as fitted."""
        known = {c.id for c in checks} | {c.group for c in checks}
        if unknown := (set(mute) | set(lock)) - known:
            raise ValueError(f"No check or criterion has the ids {sorted(unknown)}.")
        if both := set(mute) & set(lock):
            raise ValueError(f"{sorted(both)} cannot be both muted and locked.")
        fixed: dict[int, tuple[float, float]] = {}
        for j, check in enumerate(checks):
            if {check.id, check.group} & set(mute):
                fixed[j] = (0.0, 0.0)
            elif {check.id, check.group} & set(lock):
                weight = float(self.scorer_.coef[j])
                fixed[j] = (weight, weight)
        return fixed

    # Predicting and explaining.

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        return self.scorer_.proba(await self._answered(texts, session))

    async def _answered(self, texts: list[str], session: LLMSession) -> Floats:
        """Answers to every check, asking only those with a weight (NaN elsewhere)."""
        checks = from_frame(self.checks_)
        used = np.flatnonzero(self.checks_["used"].to_numpy(dtype=bool))
        answers = np.full((len(texts), len(checks)), np.nan)
        if len(used):
            answerer = self._models("answerer", "answer")
            asked = [checks[j] for j in used.tolist()]
            answers[:, used] = await self._ask(session, answerer, texts, asked)
        return answers

    def _decide(self, proba: Proba) -> Labels:
        return _decided(proba, self.cut_points_)

    def _explanations(self, answers: Floats) -> list[Explanation]:
        policy = self._policy()
        checks = from_frame(self.checks_)
        used = self.checks_["used"].to_numpy(dtype=bool)
        points = POINTS * self.scorer_.contributions(answers)
        proba = self.scorer_.proba(answers)
        outcomes = self._decide(proba)
        cutoffs = {
            name: float(POINTS * t)
            for name, t in zip(policy.names[1:], self.scorer_.thresholds, strict=True)
        }
        dropped = self.dropped_criteria_
        explanations: list[Explanation] = []
        for i in range(len(answers)):
            by_check = {
                c.id: float(points[i, j]) for j, c in enumerate(checks) if used[j]
            }
            reasons: dict[str, float] = {}
            unanswered: dict[str, bool] = {}
            for j, check in enumerate(checks):
                if not used[j]:
                    continue
                reasons[check.group] = (
                    reasons.get(check.group, 0.0) + by_check[check.id]
                )
                missing = bool(np.isnan(answers[i, j]))
                unanswered[check.group] = unanswered.get(check.group, True) and missing
            ordered = dict(sorted(reasons.items(), key=lambda kv: -abs(kv[1])))
            baseline = 0.0
            explanations.append(
                Explanation(
                    outcome=policy.names[int(outcomes[i])],
                    probabilities={
                        n: float(p) for n, p in zip(policy.names, proba[i], strict=True)
                    },
                    points=baseline + sum(ordered.values()),
                    baseline=baseline,
                    cutoffs=cutoffs,
                    reasons=ordered,
                    checks=by_check,
                    unassessed=(
                        *dropped,
                        *(g for g, gone in unanswered.items() if gone),
                    ),
                )
            )
        return explanations

    # Plans, settings and saving.

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: Classes
    ) -> list[UsageRecord]:
        policy = self._policy()
        answerer = self._models("answerer", "answer")
        checks = self._starting_checks(policy, ())
        planned: list[UsageRecord] = []
        asked = int((y == UNLABELLED).sum())
        checked = len(self._gold_checked(y))
        teacher = resolve_teacher(self.teacher) if self.teacher is not None else None
        if asked + checked and teacher is not None and not callable(teacher):
            prompt = (
                text_tokens(TEACHER_INSTRUCTIONS)
                + text_tokens(policy_block(policy))
                + mean_tokens(texts)
            )
            calls = asked * self.teacher_draws + checked
            planned.append(
                estimate_generations(teacher, prompt, calls=calls, role="teacher")
            )
        planned.append(estimate_answers(answerer, texts, [c.question for c in checks]))
        if rounds := self._rounds():
            prompt = (
                text_tokens(policy_block(policy))
                + _SHOWN_TOKENS * mean_tokens(texts)
                + (len(checks) + rounds * CHECKS_PER_ROUND) * QUESTION_TOKENS
            )
            planned += [
                estimate_generations(
                    self._writer(), prompt, calls=rounds, role="writer"
                ),
                estimate_answers(
                    answerer, texts, [QUESTION_TOKENS] * CHECKS_PER_ROUND, times=rounds
                ),
            ]
        return planned

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        answerer = self._models("answerer", "answer")
        return [estimate_answers(answerer, texts, self._used_questions())]

    def _used_questions(self) -> list[str]:
        """The questions of the checks with a weight, the only ones asked to predict."""
        used: list[bool] = self.checks_["used"].tolist()
        checks = from_frame(self.checks_)
        return [c.question for c, u in zip(checks, used, strict=True) if u]

    def _policy(self) -> Policy:
        if self.policy is None:
            raise ValueError(
                "PolicyClassifier needs policy=Policy(text=..., outcomes=..., "
                "criteria=...)."
            )
        if not isinstance(self.policy, Policy):  # pyright: ignore[reportUnnecessaryIsInstance]  # settings can be set to anything
            raise TypeError(f"policy must be a Policy, got {type(self.policy)}.")
        return self.policy

    def _starting_checks(self, policy: Policy, dropped: Collection[str]) -> list[Check]:
        """The checks from templates and your own, before any are written."""
        built = policy_checks(policy, self.templates, skip=dropped)
        return [*built, *own_checks(self.checks, OWN)]

    def _rounds(self) -> int:
        if self.writer_rounds is None:
            return 1 if self.writer is not None else 0
        return self.writer_rounds

    def _writer(self) -> tuple[Endpoint, ...]:
        """The writer's models; none when no checks are written."""
        if not self._rounds():
            return ()
        if self.writer is None:
            raise ValueError("writer_rounds needs writer=..., a reasoning model.")
        return resolve_models(self.writer, param="writer", task="generate")

    @contextmanager
    def _cost_cap(self) -> Generator[None]:
        """``max_cost_usd`` in effect, if set."""
        cap = self.max_cost_usd
        with config_context(max_cost_usd=cap) if cap is not None else nullcontext():
            yield

    def _saved_params(self) -> Mapping[str, object]:
        """The settings, without a teacher function, which cannot be saved."""
        params = dict(super()._saved_params())
        if params["teacher"] is not None and not isinstance(
            params["teacher"], (str, LLM, tuple)
        ):
            logger.warning(
                "The teacher is a function, which cannot be saved; the saved model "
                "has teacher=None. It predicts and refits without one."
            )
            params["teacher"] = None
        return params

    def _summary(self) -> Mapping[str, object]:
        return {
            "outcomes": list(self._policy().names),
            "checks": self.checks_[["id", "group", "points", "used"]].to_dict(
                "records"
            ),
            "cut_points": self.cut_points_,
            "report": self.report_,
        }


def _warn_if_lopsided(codes: Labels, policy: Policy) -> None:
    """Warn when one outcome has nearly every training label."""
    shares = np.bincount(codes, minlength=len(policy.names)) / len(codes)
    top = int(np.argmax(shares))
    if shares[top] > LOPSIDED_SHARE:
        logger.warning(
            f"{shares[top]:.0%} of the training labels are {policy.names[top]}, so "
            "there is little to learn the other outcomes from: the policy may be "
            "too strict or too loose for these cases."
        )


def _decided(proba: Floats, cuts: Floats | None) -> Labels:
    """Outcome codes: the most probable, or the cut-offs the expected one reaches."""
    if cuts is None:
        return np.argmax(proba, axis=1).astype(np.intp)
    reached = _expected(proba)[:, None] >= cuts[None, :] - 1e-12
    return reached.sum(axis=1).astype(np.intp)


def _expected(proba: Floats) -> Floats:
    """The expected outcome index of each row."""
    return proba @ np.arange(proba.shape[1], dtype=float)


def _relabelled(frame: pd.DataFrame, policy: Policy, codes: Labels) -> pd.DataFrame:
    """``frame`` with new outcomes; a changed one becomes yours, its weight kept."""
    old = outcome_codes(policy, frame["outcome"], len(frame)).tolist()
    rows: list[dict[str, object]] = frame.to_dict("records")  # pyright: ignore[reportAssignmentType]  # string column names
    for i, (new, before) in enumerate(zip(codes.tolist(), old, strict=True)):
        if new == before:
            continue
        weight = rows[i]["sample_weight"]
        rows[i] = {
            "outcome": None if new == UNLABELLED else policy.names[new],
            "source": None if new == UNLABELLED else "gold",
            "confidence": np.nan if new == UNLABELLED else 1.0,
            "why": "",
            "criteria": (),
            "unjudged": (),
            "second": None,
            "sample_weight": weight,
            "weight": 0.0 if new == UNLABELLED else weight,
        }
    return pd.DataFrame(rows, columns=frame.columns)


def _checked_weights(sample_weight: npt.ArrayLike | None, n: int) -> Floats | None:
    """``sample_weight`` as floats, one per case, finite and 0 or more."""
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise DataError(
            f"sample_weight needs one finite weight of 0 or more per case, for {n}; "
            f"got shape {weights.shape}."
        )
    return weights


def _check_ids(checks: Sequence[Check], policy: Policy) -> None:
    ids = [c.id for c in checks]
    if repeated := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"Check ids must differ; repeated: {repeated}.")
    taken = any(c.group == POLICY_GROUP and c.form != "policy" for c in checks)
    if taken or POLICY_GROUP in {c.id for c in policy.criteria}:
        raise ValueError(f"The id {POLICY_GROUP!r} is kept for whole-policy checks.")
