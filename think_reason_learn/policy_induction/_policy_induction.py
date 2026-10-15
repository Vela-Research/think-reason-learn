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

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

from think_reason_learn.core._classifier import BinaryTRLClassifier, Labels, Proba
from think_reason_learn.core._combine import (
    LinearCombiner,
    best_threshold,
    fit_logistic,
)
from think_reason_learn.core._data import balanced_subset, example_batches
from think_reason_learn.core._prompts import examples_block
from think_reason_learn.llm import Endpoint, LLMSession, Models, YesNoQuestion

from ._prompts import (
    INSTRUCTIONS_PURPOSE,
    POLICY_INSTRUCTIONS,
    UPDATE_INSTRUCTIONS,
    initial_prompt,
    policy_question,
    rule_question,
    update_prompt,
)

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
        beta: F-beta for the decision threshold; the paper reports F0.5.
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
        pos_label: str | None = None,
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
    history_: pd.DataFrame
    combiner_: LinearCombiner | None

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        self._task()
        positive = self._positive(y)
        writer = self._models("generator", "generate")
        rng = np.random.default_rng(self.random_state)
        self.instructions_ = await self._instructions(session, writer)
        cases = balanced_subset(y, 2, self.n_samples, rng)
        policy, scores = await self._induce(
            texts, y, positive, cases, writer, session, rng
        )
        self.policy_, self.combiner_ = policy, None
        if self.predictor == "policy":
            self.threshold_, _ = best_threshold(scores, positive[cases], self.beta)
        else:
            self.combiner_ = await self._fit_rules(texts, positive, session)
            self.threshold_ = self.combiner_.threshold

    async def _induce(
        self,
        texts: list[str],
        y: Labels,
        positive: npt.NDArray[np.bool_],
        cases: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
        """Refine the policy over the rounds, keeping only refinements that help."""
        judged = [texts[i] for i in cases]
        policy = self.initial_policy or await self._first(
            texts, y, cases, writer, session, rng
        )
        scores = await self._judge(policy, judged, session)
        history: list[tuple[int, int, float, float, bool]] = []
        for round_ in range(self.n_rounds):
            order = rng.permutation(cases)
            for batch in (
                order[i : i + self.batch_size]
                for i in range(0, len(order), self.batch_size)
            ):
                refined = await self._refine(policy, texts, y, batch, writer, session)
                refined_scores = await self._judge(refined, judged, session)
                before = _precision(scores, positive[cases])
                after = _precision(refined_scores, positive[cases])
                history.append((round_, len(batch), before, after, after > before))
                if after > before:
                    policy, scores = refined, refined_scores
        columns = pd.Index(
            ["round", "cases", "precision_before", "precision_after", "kept"]
        )
        self.history_ = pd.DataFrame(history, columns=columns)
        return tuple(policy), scores

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        if self.combiner_ is None:
            return self._binary_proba(await self._judge(self.policy_, texts, session))
        return self._binary_proba(
            self.combiner_.scores(await self._rule_answers(texts, session))
        )

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
    ) -> tuple[str, ...]:
        prompt = update_prompt(
            self._task(),
            self.instructions_,
            _numbered(policy),
            examples_block(texts, y, self.classes_, batch),
        )
        refined = await self._write(prompt, UPDATE_INSTRUCTIONS, writer, session)
        return refined or tuple(policy)

    async def _write(
        self,
        prompt: str,
        instructions: str,
        writer: tuple[Endpoint, ...],
        session: LLMSession,
    ) -> tuple[str, ...]:
        written = await session.generate(
            writer,
            prompt,
            role="generator",
            schema=_Policy,
            instructions=instructions.format(
                max_rules=self.max_rules, positive=self._positive_label()
            ),
        )
        return tuple(r.strip() for r in written.rules if r.strip())[: self.max_rules]

    async def _judge(
        self, policy: Sequence[str], texts: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """Probability that each sample is positive under the whole policy."""
        question = YesNoQuestion(
            policy_question(_numbered(policy), self._positive_label())
        )
        answers = await session.answer(
            self._models("answerer", "answer"),
            texts,
            {"policy": question},
            role="answerer",
        )
        return np.array(
            [
                a.probability() if (a := row["policy"]) is not None else 0.0
                for row in answers
            ]
        )

    async def _rule_answers(
        self, texts: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        asked = {
            f"r{j}": YesNoQuestion(rule_question(rule))
            for j, rule in enumerate(self.policy_)
        }
        with self._progress(len(texts) * len(asked), "rule answers") as progress:
            answers = await session.answer(
                self._models("answerer", "answer"),
                texts,
                asked,
                role="answerer",
                progress=progress,
            )
        return np.array(
            [
                [a.probability() if (a := row[q]) is not None else 0.0 for q in asked]
                for row in answers
            ]
        )

    async def _fit_rules(
        self, texts: list[str], positive: npt.NDArray[np.bool_], session: LLMSession
    ) -> LinearCombiner:
        return fit_logistic(
            await self._rule_answers(texts, session),
            positive,
            l1_ratios=(1.0,),
            cs=10,
            cv=self.cv,
            beta=self.beta,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def _positive_label(self) -> str:
        return str(self.classes_[self._pos_index()])

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
    called = scores >= 0.5
    return float((called & positive).sum() / called.sum()) if called.any() else 0.0
