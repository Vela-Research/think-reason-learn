"""Policy Induction: a policy of rules, refined batch by batch, weighed per rule.

Mu, Ternasky, Chen, Alican and Ihlamur, "Policy Induction: Predicting Startup
Success via Explainable Memory-Augmented In-Context Learning" (2025),
https://arxiv.org/abs/2505.21427.

The generator writes a first policy from labelled examples (or you give one) and
refines it with batches of training cases. To predict, the answerer answers each
rule for the sample and an L1 logistic regression weighs them, as in 0.2.

The paper's method is a few parameters away: ``memory=True`` keeps a refinement
only when the answerer, judging the cases under it, scores better than under
the current policy (the paper's memory step), and ``predictor="policy"`` judges
each sample under the whole policy. The paper refines one case at a time over
four rounds (``batch_size=1``, ``n_rounds=4``).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._classifier import (
    BinaryTRLClassifier,
    Labels,
    Proba,
    positive_index,
)
from think_reason_learn.core._combine import (
    LinearCombiner,
    fbeta,
    fit_combiner,
    held_out_rows,
)
from think_reason_learn.core._data import (
    Classes,
    balanced_subset,
    example_batches,
    take,
)
from think_reason_learn.core._prompts import examples_block
from think_reason_learn.exceptions import NoOutputError
from think_reason_learn.llm import (
    QUESTION_TOKENS,
    YES_AT,
    Endpoint,
    LLMSession,
    Models,
    UsageRecord,
    estimate_answers,
    estimate_generations,
    mean_tokens,
)

from ._prompts import (
    INSTRUCTIONS_PURPOSE,
    POLICY_INSTRUCTIONS,
    UPDATE_INSTRUCTIONS,
    initial_prompt,
    policy_question,
    update_prompt,
)

logger = logging.getLogger(__name__)

_HISTORY: Final = pd.Index(["round", "cases", "fbeta_before", "fbeta_after", "kept"])
"""Columns of ``history_``: one row per refinement; scores only with ``memory``."""

type Predictor = Literal["policy", "rules"]
type _Step = tuple[int, int, float, float, bool]


class PolicyInduction(BinaryTRLClassifier):
    """A policy of rules, refined batch by batch, its rules weighed to predict.

    For two classes; the positive class is ``pos_label`` (by default the second
    sorted label).

    Args:
        task: What to predict, in plain words.
        generator: Model(s) that write and refine the policy.
        answerer: Model(s) that judge samples under the policy. Defaults to Jev.
        instructions: Instructions for writing the policy; written from
            ``task`` during ``fit`` when not given.
        initial_policy: Rules to start from, e.g. an expert's. When empty, the
            generator writes them from ``n_examples`` labelled samples. With
            ``n_rounds=0`` the policy is used as given, and no generator is needed.
        n_examples: Labelled samples shown to write the first policy; the paper
            uses 40 (20 of each class).
        n_samples: Training cases used to refine (and with ``memory`` to judge
            refinements), balanced across classes; the paper uses 240.
        memory: Keep a refinement only when the answerer, judging every case
            under it, reaches a higher F-beta than under the current policy:
            the paper's memory step. It answers ``n_samples`` cases for each
            refinement. Off by default, as in 0.2: each refinement is kept.
        n_rounds: Passes over the cases, as in 0.2; the paper uses 4.
        batch_size: Cases per refinement, as in 0.2; the paper uses 1.
        max_rules: Most rules in the policy; the paper keeps it under 20.
        predictor: ``"rules"`` (0.2's: answer each rule, weigh them with L1
            logistic regression, its threshold tuned for F-beta by
            cross-validation) or ``"policy"`` (the paper's: judge each sample
            under the whole policy).
        beta: F-beta for the ``"rules"`` threshold and ``memory``'s comparison.
            The ``"policy"`` predictor takes the answerer's call (probability
            0.5 or more), as the paper does.
        cv: Folds for the ``"rules"`` predictor.
        pos_label: The positive class.
        random_state: Seed for the examples and the order of the cases.
        verbose: Show progress when above 0.
    """

    _instructions_purpose = INSTRUCTIONS_PURPOSE

    def __init__(
        self,
        task: str | None = None,
        generator: Models | None = None,
        answerer: Models = "jev",
        *,
        instructions: str | None = None,
        initial_policy: tuple[str, ...] = (),
        n_examples: int = 40,
        n_samples: int = 240,
        memory: bool = False,
        n_rounds: int = 1,
        batch_size: int = 10,
        max_rules: int = 20,
        predictor: Predictor = "rules",
        beta: float = 0.5,
        cv: int = 5,
        pos_label: str | int | None = None,
        random_state: int | None = 42,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.generator = generator
        self.answerer = answerer
        self.instructions = instructions
        self.initial_policy = initial_policy
        self.n_examples = n_examples
        self.n_samples = n_samples
        self.memory = memory
        self.n_rounds = n_rounds
        self.batch_size = batch_size
        self.max_rules = max_rules
        self.predictor = predictor
        self.beta = beta
        self.cv = cv
        self.pos_label = pos_label
        self.random_state = random_state
        self.verbose = verbose

    predictor: Predictor
    instructions_: str
    policy_: tuple[str, ...]
    weighted_rules_: tuple[str, ...]
    history_: pd.DataFrame
    combiner_: LinearCombiner | None

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        positive = self._positive(y)
        writer = self._writer()
        self._models("answerer", "answer")  # fail before paying for the policy
        rng = np.random.default_rng(self.random_state)
        self.instructions_ = (
            await self._instructions(session, writer)
            if writer
            else self.instructions or ""
        )
        cases = balanced_subset(y, 2, self.n_samples, rng)
        policy = await self._induce(texts, y, cases, writer, session, rng)
        self.policy_, self.combiner_, self.weighted_rules_ = policy, None, ()
        if self.predictor == "policy":
            self.threshold_ = YES_AT  # the paper takes the answerer's call as is
        else:
            await self._fit_rules(texts, positive, cases, session)

    async def _induce(
        self,
        texts: list[str],
        y: Labels,
        cases: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[str, ...]:
        """The first policy, refined batch by batch over the rounds."""
        policy = tuple(self.initial_policy) or await self._first(
            texts, y, cases, writer, session, rng
        )
        if not policy:
            raise NoOutputError("The generator wrote an empty policy.")
        refine = self._remember if self.memory else self._update
        policy, history = await refine(policy, texts, y, cases, writer, session, rng)
        self.history_ = pd.DataFrame(history, columns=_HISTORY)
        return policy

    async def _update(
        self,
        policy: tuple[str, ...],
        texts: list[str],
        y: Labels,
        cases: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[tuple[str, ...], list[_Step]]:
        """0.2's refinement: each batch's rewrite replaces the policy."""
        history: list[_Step] = []
        for round_, batch in _batches(cases, self.n_rounds, self.batch_size, rng):
            rewritten = await self._rewrite(
                policy, texts, y, batch, writer, session, draw=round_
            )
            history.append((round_, len(batch), math.nan, math.nan, bool(rewritten)))
            policy = rewritten or policy
        return policy, history

    async def _remember(
        self,
        policy: tuple[str, ...],
        texts: list[str],
        y: Labels,
        cases: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[tuple[str, ...], list[_Step]]:
        """The paper's memory step: keep a rewrite only if it judges the cases better.

        Each round asks for fresh rewrites (a new draw), so a case seen again under
        an unchanged policy is not answered from the cache. Once the policy is
        perfect on the cases nothing can beat it, so refining stops.
        """
        judged, truth = take(texts, cases), self._positive(y[cases])
        best = self._fbeta(await self._judge(policy, judged, session), truth)
        history: list[_Step] = []
        for round_, batch in _batches(cases, self.n_rounds, self.batch_size, rng):
            if best == 1.0:
                break
            rewritten = await self._rewrite(
                policy, texts, y, batch, writer, session, draw=round_
            )
            tried = await self._judged(rewritten, judged, truth, session)
            after = best if tried is None else tried[1]
            history.append((round_, len(batch), best, after, after > best))
            if tried is not None and after > best:
                policy, best = tried
        return policy, history

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: Classes
    ) -> list[UsageRecord]:
        writer = self._writer()
        answerer = self._models("answerer", "answer")
        rng = np.random.default_rng(self.random_state)
        cases = balanced_subset(y, 2, self.n_samples, rng)
        refinements = self.n_rounds * math.ceil(len(cases) / self.batch_size)
        policy, example = self.max_rules * QUESTION_TOKENS, mean_tokens(texts)
        instructions = self._instructions_tokens()  # every prompt carries them
        planned = self._instructions_plan() if writer else []
        if not self.initial_policy:
            first = instructions + self.n_examples * example
            planned.append(estimate_generations(writer, first))
        if refinements:
            refine = instructions + policy + self.batch_size * example
            planned.append(estimate_generations(writer, refine, calls=refinements))
        if self.memory:
            judged = take(texts, cases)
            planned.append(
                estimate_answers(answerer, judged, [policy], times=1 + refinements)
            )
        if self.predictor == "rules":
            positive = y == positive_index(classes, self.pos_label)
            rows = take(texts, held_out_rows(positive, cases, self.cv))
            rules = [QUESTION_TOKENS] * self.max_rules
            planned.append(estimate_answers(answerer, rows, rules))
        return planned

    def _writer(self) -> tuple[Endpoint, ...]:
        """The generator, unless nothing is written: a given policy, not refined."""
        if self.initial_policy and not self.n_rounds:
            return ()
        return self._models("generator", "generate")

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        answerer = self._models("answerer", "answer")
        if self.combiner_ is None:
            question = self._policy_question(self.policy_)
            return [estimate_answers(answerer, texts, [question])]
        rules = [self._policy_question((rule,)) for rule in self.weighted_rules_]
        return [estimate_answers(answerer, texts, rules)]

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        if self.combiner_ is None:
            return self._binary_proba(await self._judge(self.policy_, texts, session))
        answers = await self._rule_answers(texts, self.weighted_rules_, session)
        return self._binary_proba(self.combiner_.scores(answers))

    async def _first(
        self,
        texts: list[str],
        y: Labels,
        cases: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[str, ...]:
        [shown] = example_batches(
            y[cases],
            self.classes_,
            n_batches=1,
            batch_size=min(self.n_examples, len(cases)),
            class_ratio="balanced",
            rng=rng,
        )
        prompt = initial_prompt(
            self._task(),
            self.instructions_,
            examples_block(texts, y, self.classes_, cases[shown]),
        )
        return await self._write(prompt, POLICY_INSTRUCTIONS, writer, session)

    async def _rewrite(
        self,
        policy: tuple[str, ...],
        texts: list[str],
        y: Labels,
        batch: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        *,
        draw: int,
    ) -> tuple[str, ...] | None:
        """``policy`` rewritten from ``batch``; ``None`` if unchanged or not written."""
        prompt = update_prompt(
            self._task(),
            self.instructions_,
            _numbered(policy),
            examples_block(texts, y, self.classes_, batch),
        )
        try:
            rewritten = await self._write(
                prompt, UPDATE_INSTRUCTIONS, writer, session, draw
            )
        except NoOutputError as error:
            logger.warning(f"A refinement failed ({error}); the policy stays.")
            return None
        return rewritten if rewritten and rewritten != policy else None

    async def _judged(
        self,
        policy: tuple[str, ...] | None,
        judged: Sequence[str],
        truth: npt.NDArray[np.bool_],
        session: LLMSession,
    ) -> tuple[tuple[str, ...], float] | None:
        """A rewritten policy and its F-beta on the cases; ``None`` if not judged."""
        if policy is None:
            return None
        try:
            scores = await self._judge(policy, judged, session)
        except NoOutputError as error:
            logger.warning(f"A refinement was not judged ({error}); the policy stays.")
            return None
        return policy, self._fbeta(scores, truth)

    async def _write(
        self,
        prompt: str,
        instructions: str,
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        draw: int = 0,
    ) -> tuple[str, ...]:
        written = await session.generate(
            writer,
            prompt,
            role="generator",
            schema=_Policy,
            instructions=instructions.format(
                max_rules=self.max_rules, positive=self._positive_label()
            ),
            draw=draw,
        )
        rules = dict.fromkeys(r.strip() for r in written.rules if r.strip())
        return tuple(rules)[: self.max_rules]

    async def _judge(
        self, policy: Sequence[str], texts: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """Probability that each sample is positive under the whole policy."""
        answerer = self._models("answerer", "answer")
        scores = await yes_probabilities(
            session, answerer, texts, [self._policy_question(policy)], missing=0.0
        )
        return scores[:, 0]

    def _policy_question(self, policy: Sequence[str]) -> str:
        """The question that judges a sample under a policy, or one rule."""
        rules = policy[0] if len(policy) == 1 else _numbered(policy)
        return policy_question(self._task(), rules)

    def _fbeta(
        self, scores: npt.NDArray[np.float64], positive: npt.NDArray[np.bool_]
    ) -> float:
        """F-beta of calling a sample positive when its probability is at least 0.5."""
        called = scores >= YES_AT
        tp, fp = float((called & positive).sum()), float((called & ~positive).sum())
        fn = float((~called & positive).sum())
        return float(fbeta(np.array(tp), np.array(fp), np.array(fn), self.beta))

    async def _rule_answers(
        self, texts: Sequence[str], rules: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """Probability that each rule (columns) holds for each sample (rows)."""
        return await yes_probabilities(
            session,
            self._models("answerer", "answer"),
            texts,
            [self._policy_question((rule,)) for rule in rules],
            missing=0.0,
        )

    async def _fit_rules(
        self,
        texts: list[str],
        positive: npt.NDArray[np.bool_],
        cases: npt.NDArray[np.intp],
        session: LLMSession,
    ) -> None:
        """Weigh the rules on samples the policy was not written from, if enough."""
        rows = held_out_rows(positive, cases, self.cv)
        answers = await self._rule_answers(take(texts, rows), self.policy_, session)
        self.combiner_, used = await fit_combiner(
            answers,
            positive[rows],
            l1_ratios=(1.0,),
            cv=self.cv,
            beta=self.beta,
            class_weight=None,
            random_state=self.random_state,
            drop_zero=True,
        )
        self.weighted_rules_ = tuple(take(self.policy_, used))
        self.threshold_ = self.combiner_.threshold

    def _summary(self) -> Mapping[str, object]:
        return {
            "instructions": self.instructions_,
            "policy": list(self.policy_),
            "threshold": self.threshold_,
            "history": self.history_.to_dict("records"),
        }


def _batches(
    cases: npt.NDArray[np.intp], n_rounds: int, size: int, rng: np.random.Generator
) -> Iterator[tuple[int, npt.NDArray[np.intp]]]:
    """(round, cases) for each refinement; each round shuffles the cases afresh."""
    for round_ in range(n_rounds):
        for batch in itertools.batched(rng.permutation(cases), size, strict=False):
            yield round_, np.array(batch)


def _numbered(policy: Sequence[str]) -> str:
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(policy, 1))


class _Policy(BaseModel):
    rules: list[str]
