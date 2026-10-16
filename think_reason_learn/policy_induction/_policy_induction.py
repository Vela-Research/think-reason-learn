"""Policy Induction: a policy of rules, refined case by case, applied to each sample.

Mu, Ternasky, Chen, Alican and Ihlamur, "Policy Induction: Predicting Startup
Success via Explainable Memory-Augmented In-Context Learning" (2025),
https://arxiv.org/abs/2505.21427.

The generator writes a first policy from labelled examples (or you give one) and
refines it with training cases a few at a time, over several rounds. A refined
policy is kept only when the answerer, judging the cases under it, reaches a
higher precision than under the current one: the paper's memory step. To
predict, the answerer judges each sample under the whole policy; with
``predictor="rules"`` each rule is answered separately and a sparse logistic
regression weighs them.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

from think_reason_learn.core._classifier import (
    BinaryTRLClassifier,
    Labels,
    Proba,
    positive_index,
)
from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._combine import LinearCombiner, fit_logistic, held_out_rows
from think_reason_learn.core._data import Classes, balanced_subset, example_batches
from think_reason_learn.core._prompts import examples_block, wrote_nothing
from think_reason_learn.exceptions import LLMError
from think_reason_learn.llm import (
    YES_AT,
    Endpoint,
    LLMSession,
    Models,
    UsageRecord,
    estimate_answers,
    estimate_generations,
)
from think_reason_learn.llm._estimate import QUESTION_TOKENS, mean_tokens

from ._prompts import (
    INSTRUCTIONS_PURPOSE,
    POLICY_INSTRUCTIONS,
    UPDATE_INSTRUCTIONS,
    initial_prompt,
    policy_question,
    rule_question,
    update_prompt,
)

logger = logging.getLogger(__name__)

type Predictor = Literal["policy", "rules"]


class _Policy(BaseModel):
    rules: list[str]


class PolicyInduction(BinaryTRLClassifier):
    """A policy of rules, refined case by case and kept only when it improves.

    For two classes; the positive class is ``pos_label`` (by default the second
    sorted label).

    Args:
        task: What to predict, in plain words.
        generator: Model(s) that write and refine the policy.
        answerer: Model(s) that judge samples under the policy. Defaults to Jev.
        instructions: Instructions for writing the policy; written from
            ``task`` during ``fit`` when not given.
        initial_policy: Rules to start from, e.g. an expert's. When empty, the
            generator writes them from ``n_examples`` labelled samples.
        n_examples: Labelled samples shown to write the first policy; the paper
            uses 40 (20 of each class).
        n_samples: Training cases used to refine and to judge refinements,
            balanced across classes; the paper uses 240.
        n_rounds: Passes over the cases; the paper uses 4.
        batch_size: Cases per refinement; the paper uses 1. Larger is cheaper.
        max_rules: Most rules in the policy; the paper keeps it under 20.
        predictor: ``"policy"`` (the paper's: judge each sample under the whole
            policy) or ``"rules"`` (judge each rule, weigh them with L1
            logistic regression).
        beta: F-beta for the ``"rules"`` predictor's threshold. The ``"policy"``
            predictor takes the answerer's call (probability 0.5 or more), as
            the paper does.
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
        n_rounds: int = 4,
        batch_size: int = 1,
        max_rules: int = 20,
        predictor: Predictor = "policy",
        beta: float = 0.5,
        cv: int = 5,
        pos_label: str | int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.generator = generator
        self.answerer = answerer
        self.instructions = instructions
        self.initial_policy = initial_policy
        self.n_examples = n_examples
        self.n_samples = n_samples
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
        writer = self._models("generator", "generate")
        self._models("answerer", "answer")  # fail before paying for the policy
        rng = np.random.default_rng(self.random_state)
        self.instructions_ = await self._instructions(session, writer)
        cases = balanced_subset(y, 2, self.n_samples, rng)
        policy = await self._induce(texts, y, positive, cases, writer, session, rng)
        self.policy_, self.combiner_, self.weighted_rules_ = policy, None, ()
        if self.predictor == "policy":
            self.threshold_ = YES_AT  # the paper takes the answerer's call as is
        else:
            await self._fit_rules(texts, positive, cases, session)

    async def _induce(
        self,
        texts: list[str],
        y: Labels,
        positive: npt.NDArray[np.bool_],
        cases: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[str, ...]:
        """Refine the policy over the rounds, keeping only refinements that help.

        Each round asks for fresh refinements (a new draw), so a case seen again
        under an unchanged policy is not answered from the cache. Once the policy
        is perfectly precise on the cases nothing can beat it, so refining stops.
        """
        judged = [texts[i] for i in cases]
        policy = self.initial_policy or await self._first(
            texts, y, cases, writer, session, rng
        )
        if not policy:
            raise LLMError("The generator wrote an empty policy.")
        scores = await self._judge(policy, judged, session)
        history: list[tuple[int, int, float, float, bool]] = []
        for round_ in range(self.n_rounds):
            for batch in itertools.batched(rng.permutation(cases), self.batch_size):
                if _precision(scores, positive[cases]) == 1.0:
                    break
                refined = await self._refine(
                    policy, texts, y, np.array(batch), writer, session, draw=round_
                )
                refined_scores = (
                    scores
                    if refined == tuple(policy)
                    else await self._judge(refined, judged, session)
                )
                before = _precision(scores, positive[cases])
                after = _precision(refined_scores, positive[cases])
                history.append((round_, len(batch), before, after, after > before))
                if after > before:
                    policy, scores = refined, refined_scores
        columns = pd.Index(
            ["round", "cases", "precision_before", "precision_after", "kept"]
        )
        self.history_ = pd.DataFrame(history, columns=columns)
        return tuple(policy)

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: Classes
    ) -> list[UsageRecord]:
        writer = self._models("generator", "generate")
        answerer = self._models("answerer", "answer")
        rng = np.random.default_rng(self.random_state)
        cases = balanced_subset(y, 2, self.n_samples, rng)
        refinements = self.n_rounds * math.ceil(len(cases) / self.batch_size)
        policy, example = self.max_rules * QUESTION_TOKENS, mean_tokens(texts)
        planned = self._instructions_plan()
        if not self.initial_policy:
            planned.append(estimate_generations(writer, self.n_examples * example))
        planned += [
            estimate_generations(
                writer, policy + self.batch_size * example, calls=refinements
            ),
            estimate_answers(
                answerer, [texts[i] for i in cases], [policy], times=1 + refinements
            ),
        ]
        if self.predictor == "rules":
            positive = y == positive_index(classes, self.pos_label)
            rows = [texts[i] for i in held_out_rows(positive, cases, self.cv)]
            planned.append(
                estimate_answers(answerer, rows, [QUESTION_TOKENS] * self.max_rules)
            )
        return planned

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        answerer = self._models("answerer", "answer")
        if self.combiner_ is None:
            question = policy_question(_numbered(self.policy_), self._positive_label())
            return [estimate_answers(answerer, texts, [question])]
        rules = [rule_question(rule) for rule in self.weighted_rules_]
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

    async def _refine(
        self,
        policy: Sequence[str],
        texts: list[str],
        y: Labels,
        batch: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        *,
        draw: int,
    ) -> tuple[str, ...]:
        prompt = update_prompt(
            self._task(),
            self.instructions_,
            _numbered(policy),
            examples_block(texts, y, self.classes_, batch),
        )
        try:
            refined = await self._write(
                prompt, UPDATE_INSTRUCTIONS, writer, session, draw
            )
        except LLMError as error:
            if not wrote_nothing(error):
                raise
            logger.warning(f"A refinement wrote nothing ({error}); the policy stays.")
            return tuple(policy)
        return refined or tuple(policy)

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
        question = policy_question(_numbered(policy), self._positive_label())
        answerer = self._models("answerer", "answer")
        scores = await yes_probabilities(
            session, answerer, texts, [question], missing=0.0
        )
        return scores[:, 0]

    async def _rule_answers(
        self, texts: Sequence[str], rules: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """Probability that each rule (columns) holds for each sample (rows)."""
        return await yes_probabilities(
            session,
            self._models("answerer", "answer"),
            texts,
            [rule_question(rule) for rule in rules],
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
        answers = await self._rule_answers(
            [texts[i] for i in rows], self.policy_, session
        )
        fitted = await asyncio.to_thread(
            partial(
                fit_logistic,
                answers,
                positive[rows],
                l1_ratios=(1.0,),
                cv=self.cv,
                beta=self.beta,
                class_weight="balanced",
                random_state=self.random_state,
            )
        )
        used = np.flatnonzero(fitted.coef)
        self.weighted_rules_ = tuple(self.policy_[j] for j in used)
        self.combiner_ = LinearCombiner(
            fitted.coef[used], fitted.intercept, fitted.threshold
        )
        self.threshold_ = fitted.threshold

    def _summary(self) -> Mapping[str, object]:
        return {
            "instructions": self.instructions_,
            "policy": list(self.policy_),
            "threshold": self.threshold_,
            "history": self.history_.to_dict("records"),
        }


def _numbered(policy: Sequence[str]) -> str:
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(policy, 1))


def _precision(
    scores: npt.NDArray[np.float64], positive: npt.NDArray[np.bool_]
) -> float:
    """Precision of calling a sample positive when its probability is at least 0.5."""
    called = scores >= YES_AT
    return float((called & positive).sum() / called.sum()) if called.any() else 0.0
