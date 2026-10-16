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

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from think_reason_learn.core._classifier import BinaryTRLClassifier, Labels, Proba
from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._combine import LinearCombiner, fit_logistic, n_folds
from think_reason_learn.core._data import balanced_subset
from think_reason_learn.llm import (
    Endpoint,
    LLMSession,
    Models,
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

_CLIP = 1e-3
"""Probabilities are clipped to [_CLIP, 1 - _CLIP] before taking log-odds."""


class _Mined(BaseModel):
    reasoning: str
    rule: str


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
            look; the paper uses 0.95.
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
        harsh_gate: float = 0.95,
        harsh_level: HarshLevel = "strict",
        beta: float = 0.5,
        cv: int = 5,
        pos_label: str | int | None = None,
        random_state: int | None = None,
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

    instructions: str | None = None
    harsh_level: HarshLevel
    rules_: tuple[tuple[str, str], ...]
    policy_: str
    combiner_: LinearCombiner

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        self._task()
        positive = self._positive(y)
        rng = np.random.default_rng(self.random_state)
        mined = self._mined_samples(y, rng)
        writer = self._models("generator", "generate")
        self.rules_ = await self._mine(texts, y, mined, writer, session)
        self.policy_ = await self._compile(writer, session)
        calibrate = self._calibration_samples(positive, mined)
        features = await self._features([texts[i] for i in calibrate], session)
        self.combiner_ = fit_logistic(
            features,
            positive[calibrate],
            l1_ratios=(0.0,),
            cs=10,
            cv=self.cv,
            beta=self.beta,
            class_weight=None,  # unweighted, so predict_proba stays calibrated
            random_state=self.random_state,
        )
        self.threshold_ = self.combiner_.threshold

    def _calibration_samples(
        self, positive: npt.NDArray[np.bool_], mined: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.intp]:
        """Samples the miner never saw, if there are enough of each class."""
        held_out = np.setdiff1d(np.arange(len(positive)), mined)
        if n_folds(self.cv, positive[held_out]) >= 2:
            return held_out
        logger.warning(
            "Too few unmined samples of each class to calibrate on; calibrating "
            "on every sample, so the threshold may be optimistic. Lower n_samples."
        )
        return np.arange(len(positive))

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        return self._binary_proba(
            self.combiner_.scores(await self._features(texts, session))
        )

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
        """(class, rule) for each mined sample, reasoned from its true class."""
        replies = await asyncio.gather(
            *(
                session.generate(
                    writer,
                    mining_prompt(self._task(), texts[i], str(self.classes_[y[i]])),
                    role="generator",
                    schema=_Mined,
                    instructions=MINING_INSTRUCTIONS,
                )
                for i in mined
            )
        )
        return tuple(
            (str(self.classes_[y[i]]), reply.rule.strip())
            for i, reply in zip(mined, replies)
            if reply.rule.strip()
        )

    async def _compile(self, writer: tuple[Endpoint, ...], session: LLMSession) -> str:
        by_class: dict[str, list[str]] = {}
        for label, rule in self.rules_:
            by_class.setdefault(label, []).append(rule)
        return await session.generate(
            writer,
            compile_prompt(self._task(), by_class),
            role="generator",
            instructions=COMPILE_INSTRUCTIONS,
        )

    async def _features(
        self, texts: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """Log-odds of the first pass and of the second look, and whether it ran."""
        positive = self._positive_label()
        first = await self._judge(
            self._models("answerer", "answer"),
            vote_question(self.policy_, positive),
            texts,
            session,
        )
        harsh = np.full(len(texts), 0.5)
        promoted = (
            first >= self.harsh_gate
            if self.use_harsh
            else np.zeros(len(texts), dtype=bool)
        )
        if promoted.any():
            question = harsh_question(self.policy_, positive, self.harsh_level)
            reviewed = [texts[int(i)] for i in np.flatnonzero(promoted)]
            harsh[promoted] = await self._judge(
                self._harsh_models(), question, reviewed, session
            )
        return np.column_stack(
            [
                _logit(first),
                np.where(promoted, _logit(harsh), 0.0),
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
        """Mean probability of yes over the ensemble (one draw for decision models)."""
        draws = 1 if all(m.kind == "decision" for m in models) else self.ensemble_size
        runs = await asyncio.gather(
            *(
                yes_probabilities(
                    session, models, texts, [question], missing=0.5, draw=d
                )
                for d in range(draws)
            )
        )
        return np.mean([run[:, 0] for run in runs], axis=0)

    def _harsh_models(self) -> tuple[Endpoint, ...]:
        if self.harsh_answerer is None:
            return self._models("answerer", "answer")
        return resolve_models(
            self.harsh_answerer, param="harsh_answerer", task="answer"
        )

    def _summary(self) -> Mapping[str, object]:
        return {
            "policy": self.policy_,
            "rules": [list(r) for r in self.rules_],
            "threshold": self.threshold_,
        }


def _logit(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(p, _CLIP, 1 - _CLIP)
    return np.log(clipped / (1 - clipped))
