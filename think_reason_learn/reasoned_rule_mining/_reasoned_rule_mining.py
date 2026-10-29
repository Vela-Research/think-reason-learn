"""Reasoned Rule Mining: rules mined from reasoning, applied in a two-stage cascade.

Preuveneers and Ihlamur, "Reasoned Rule Mining: Precision-Optimised Weighted
LogProb Classification" (IEEE CSCloud 2025),
https://vela.partners/research/reasoned-rule-mining.

The generator reasons about labelled samples and states one IF-THEN rule per
sample; the rules are compiled into one policy. To classify, the answerer judges
each sample under the policy (an ensemble of draws for a reasoning model, one
probability from a decision model). Samples it rates highly get a harsh second
look, by a stronger model if you give one. A logistic regression, fitted on
samples the miner never saw, calibrates and weighs both stages and its threshold
maximises F0.5. The paper weighs token log probabilities; this library reads
probabilities from decision models or vote shares instead, so any provider works.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._classifier import (
    BinaryTRLClassifier,
    Labels,
    Proba,
    positive_index,
)
from think_reason_learn.core._combine import LinearCombiner, fit_combiner, held_out_rows
from think_reason_learn.core._data import Classes, balanced_subset, take
from think_reason_learn.core._prompts import most_written
from think_reason_learn.exceptions import NoOutputError
from think_reason_learn.llm import (
    GENERATE_OUTPUT_TOKENS,
    QUESTION_TOKENS,
    Endpoint,
    LLMSession,
    Models,
    UsageRecord,
    estimate_answers,
    estimate_generations,
    mean_tokens,
    resolve_models,
)

from ._prompts import (
    COMPILE_INSTRUCTIONS,
    MINING_INSTRUCTIONS,
    HarshLevel,
    compile_prompt,
    harsh_question,
    mining_prompt,
    vote_question,
)

logger = logging.getLogger(__name__)

_CLIP: Final = 1e-3
"""Probabilities are clipped to [_CLIP, 1 - _CLIP] before taking log-odds."""


class ReasonedRuleMining(BinaryTRLClassifier):
    """Rules mined from reasoning about labelled samples, applied in a cascade.

    For two classes; the positive class is ``pos_label`` (by default the second
    sorted label).

    Args:
        task: What to predict, in plain words.
        generator: Model(s) that reason, state rules and compile the policy.
        answerer: Model(s) that judge samples under the policy. Defaults to Jev.
        harsh_answerer: Model(s) for the second look at highly rated samples;
            defaults to ``answerer``. The paper uses a stronger model here.
        n_samples: Labelled samples to mine rules from, balanced across
            classes; the rest calibrate the cascade. ``None`` mines every
            sample and calibrates on them too.
        ensemble_size: Judgements per sample by a reasoning answerer, averaged.
            A decision model gives a probability in one call.
        use_harsh: Run the second look.
        harsh_gate: First-pass probability at which a sample gets the second
            look: 0.2's 0.80; the paper uses 0.95.
        harsh_level: How severe the second look is.
        beta: F-beta for the threshold; the paper uses 0.5.
        cv: Folds for the calibration.
        pos_label: The positive class.
        random_state: Seed for the mined samples and the folds.
        verbose: Show progress when above 0.
    """

    def __init__(
        self,
        task: str | None = None,
        generator: Models | None = None,
        answerer: Models = "jev",
        *,
        harsh_answerer: Models | None = None,
        n_samples: int | None = 500,
        ensemble_size: int = 3,
        use_harsh: bool = True,
        harsh_gate: float = 0.8,
        harsh_level: HarshLevel = "strict",
        beta: float = 0.5,
        cv: int = 5,
        pos_label: str | int | None = None,
        random_state: int | None = 42,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.generator = generator
        self.answerer = answerer
        self.harsh_answerer = harsh_answerer
        self.n_samples = n_samples
        self.ensemble_size = ensemble_size
        self.use_harsh = use_harsh
        self.harsh_gate = harsh_gate
        self.harsh_level = harsh_level
        self.beta = beta
        self.cv = cv
        self.pos_label = pos_label
        self.random_state = random_state
        self.verbose = verbose

    harsh_level: HarshLevel
    mined_rules_: tuple[tuple[str, str], ...]
    policy_: str
    combiner_: LinearCombiner
    harsh_share_: float
    """Share of the calibration samples that got the second look."""

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        positive = self._positive(y)
        rng = np.random.default_rng(self.random_state)
        mined = self._mined_samples(y, rng)
        writer = self._models("generator", "generate")
        self._models("answerer", "answer")  # fail before paying for mining
        self._harsh_models()
        self.mined_rules_ = await self._mine(texts, y, mined, writer, session)
        self.policy_ = await self._compile(writer, session)
        calibrate = self._calibration_samples(positive, mined)
        features = await self._features(
            take(texts, calibrate), session, harsh=self.use_harsh
        )
        self.harsh_share_ = float(features[:, 2].mean()) if len(features) else 0.0
        self.combiner_, _ = await fit_combiner(
            features,
            positive[calibrate],
            l1_ratios=(0.0,),
            cv=self.cv,
            beta=self.beta,
            class_weight=None,  # unweighted, so predict_proba stays calibrated
            random_state=self.random_state,
        )
        self.threshold_ = self.combiner_.threshold

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: Classes
    ) -> list[UsageRecord]:
        """Mining, compiling, then judging; any sample may get the second look."""
        writer = self._models("generator", "generate")
        mined = self._mined_samples(y, np.random.default_rng(self.random_state))
        positive = y == positive_index(classes, self.pos_label)
        calibrate = take(texts, self._calibration_samples(positive, mined))
        policy = GENERATE_OUTPUT_TOKENS  # the compiled policy is a whole reply
        return [
            estimate_generations(writer, mean_tokens(texts), calls=len(mined)),
            estimate_generations(writer, len(mined) * QUESTION_TOKENS),
            *self._judging_plan(calibrate, calibrate, policy, policy),
        ]

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        task, share = self._task(), self.harsh_share_ if self._harsh_counts() else 0.0
        return self._judging_plan(
            texts,
            texts[: round(len(texts) * share)],
            vote_question(task, self.policy_),
            harsh_question(task, self.policy_, self.harsh_level),
        )

    def _judging_plan(
        self,
        texts: Sequence[str],
        promoted: Sequence[str],
        question: str | int,
        harsh: str | int,
    ) -> list[UsageRecord]:
        answerer = self._models("answerer", "answer")
        planned = [
            estimate_answers(answerer, texts, [question], times=self._draws(answerer))
        ]
        if self.use_harsh and promoted:
            models = self._harsh_models()
            planned.append(
                estimate_answers(models, promoted, [harsh], times=self._draws(models))
            )
        return planned

    def _calibration_samples(
        self, positive: npt.NDArray[np.bool_], mined: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.intp]:
        """Samples the miner never saw, if there are enough of each class."""
        if self.n_samples is None:  # every sample was mined, as asked
            return np.arange(len(positive), dtype=np.intp)
        return held_out_rows(positive, mined, self.cv)

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        features = await self._features(texts, session, harsh=self._harsh_counts())
        return self._binary_proba(self.combiner_.scores(features))

    def _harsh_counts(self) -> bool:
        """Whether the second look carries weight; if not, it is not paid for."""
        return self.use_harsh and bool(np.any(self.combiner_.coef[1:]))

    def _mined_samples(
        self, y: Labels, rng: np.random.Generator
    ) -> npt.NDArray[np.intp]:
        if self.n_samples is None:
            return np.arange(len(y), dtype=np.intp)
        # At most half of each class, so the rest can calibrate the cascade.
        return balanced_subset(y, 2, self.n_samples, rng, max_share=0.5)

    async def _mine(
        self,
        texts: list[str],
        y: Labels,
        mined: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
    ) -> tuple[tuple[str, str], ...]:
        """(class, rule) for each mined sample, reasoned from its true class.

        A sample the generator writes nothing for is skipped.
        """

        async def mine(i: int) -> tuple[str, str]:
            label = str(self.classes_[y[i]])
            reply = await session.generate(
                writer,
                mining_prompt(self._task(), texts[i], label),
                role="generator",
                schema=_Mined,
                instructions=MINING_INSTRUCTIONS,
            )
            return label, reply.rule.strip()

        rules = await most_written([mine(int(i)) for i in mined], "mined samples")
        usable = tuple((label, rule) for label, rule in rules if rule)
        if not usable:
            # The replies parsed, so most_written was satisfied, but every rule
            # was empty. Compiling a policy from nothing gives one the model
            # invented rather than one it mined, which reads as a real result.
            raise NoOutputError(
                f"No rule was mined from {len(mined)} samples: the generator "
                "replied, but every rule was empty."
            )
        return usable

    async def _compile(self, writer: tuple[Endpoint, ...], session: LLMSession) -> str:
        by_class: dict[str, list[str]] = {}
        for label, rule in self.mined_rules_:
            by_class.setdefault(label, []).append(rule)
        return await session.generate(
            writer,
            compile_prompt(self._task(), by_class),
            role="generator",
            instructions=COMPILE_INSTRUCTIONS,
        )

    async def _features(
        self, texts: Sequence[str], session: LLMSession, *, harsh: bool
    ) -> npt.NDArray[np.float64]:
        """Log-odds of the first pass and of the second look, and whether it ran.

        With ``harsh=False`` no sample gets the second look.
        """
        task = self._task()
        first = await self._judge(
            self._models("answerer", "answer"),
            vote_question(task, self.policy_),
            texts,
            session,
        )
        second = np.full(len(texts), 0.5)
        promoted = first >= self.harsh_gate if harsh else np.zeros(len(texts), bool)
        if promoted.any():
            question = harsh_question(task, self.policy_, self.harsh_level)
            reviewed = take(texts, np.flatnonzero(promoted))
            second[promoted] = await self._judge(
                self._harsh_models(), question, reviewed, session
            )
        return np.column_stack(
            [
                _logit(first),
                np.where(promoted, _logit(second), 0.0),
                promoted.astype(float),
            ]
        )

    async def _judge(
        self,
        models: tuple[Endpoint, ...],
        question: str,
        texts: Sequence[str],
        session: LLMSession,
    ) -> npt.NDArray[np.float64]:
        """Mean probability of yes over the ensemble (one draw for decision models).

        Draws keep the model's default temperature, so they differ; an unanswered
        draw counts as NO.
        """
        draws = self._draws(models)
        runs = await asyncio.gather(
            *(
                yes_probabilities(
                    session,
                    models,
                    texts,
                    [question],
                    missing=0.0,
                    draw=d,
                    ensemble=draws > 1,
                )
                for d in range(draws)
            )
        )
        return np.mean([run[:, 0] for run in runs], axis=0)

    def _draws(self, models: Sequence[Endpoint]) -> int:
        """One call when the first model decides; ``ensemble_size`` to reason."""
        return 1 if models[0].kind == "decision" else self.ensemble_size

    def _harsh_models(self) -> tuple[Endpoint, ...]:
        if self.harsh_answerer is None:
            return self._models("answerer", "answer")
        return resolve_models(
            self.harsh_answerer, param="harsh_answerer", task="answer"
        )

    def _summary(self) -> Mapping[str, object]:
        return {
            "policy": self.policy_,
            "mined_rules": [list(r) for r in self.mined_rules_],
            "threshold": self.threshold_,
        }


def _logit(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(p, _CLIP, 1 - _CLIP)
    return np.log(clipped / (1 - clipped))


class _Mined(BaseModel):
    reasoning: str
    rule: str
