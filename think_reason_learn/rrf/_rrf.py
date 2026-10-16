"""Random Rule Forest (RRF): a vote of YES/NO questions written by a language model.

Griffin, Yin, Vidaurre, Koyluoglu, Ternasky, Alican and Ihlamur, "Random Rule
Forest: Interpretable Ensembles of LLM-Generated Questions for Predicting
Startup Success" (2025), https://arxiv.org/abs/2505.24622.

The generator writes questions from small class-balanced batches of a held-out
generation set. The answerer answers every question for the other training
samples. Questions are ranked by F-beta (F0.5 by default), weak and redundant
ones dropped, and the prediction is positive when at least T of the top N
questions say YES, with (N, T) tuned by cross-validation. With
``combiner="elasticnet"`` a regularised logistic regression weighs the answers.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel

from think_reason_learn.core._answers import yes_probabilities
from think_reason_learn.core._classifier import BinaryTRLClassifier, Labels, Proba
from think_reason_learn.core._combine import LinearCombiner, fit_logistic
from think_reason_learn.core._data import (
    Classes,
    ClassRatio,
    TextInput,
    balanced_subset,
    count,
    example_batches,
    take,
)
from think_reason_learn.core._prompts import examples_block, most_written
from think_reason_learn.exceptions import NoOutputError
from think_reason_learn.llm import (
    Endpoint,
    LLMSession,
    Models,
    UsageRecord,
    estimate_answers,
    estimate_generations,
)
from think_reason_learn.llm._estimate import QUESTION_TOKENS, mean_tokens

from ._prompts import INSTRUCTIONS_PURPOSE, QUESTION_INSTRUCTIONS, question_prompt
from ._selection import as_yes, question_stats, rank, tune_vote

type Combiner = Literal["vote", "elasticnet"]


class RRF(BinaryTRLClassifier):
    """Random Rule Forest: a tuned vote of YES/NO questions.

    For two classes. The positive class is ``pos_label``, by default the second
    of the sorted labels (``"YES"`` of ``"NO"``/``"YES"``).

    Args:
        task: What to predict, in plain words.
        generator: Model(s) that write the instructions and the questions.
        answerer: Model(s) that answer the questions. Defaults to Jev.
        instructions: Instructions for writing questions; written from ``task``
            during ``fit`` when not given (see ``write_instructions``).
        expert_questions: Questions of your own, scored like the written ones.
        n_prompts: Question-writing prompts; the paper uses 25.
        questions_per_prompt: Questions per prompt; the paper uses 10.
        n_examples: Labelled samples shown per prompt; the paper uses 20.
        example_class_ratio: Class shares among them; the paper balances them.
        generation_size: Samples set aside for writing questions (int), or that
            fraction of the training samples (float). They are not used to score
            questions, so selection is not tuned on samples the generator saw.
            Each class keeps at least half its samples for scoring.
        beta: F-beta used to rank questions and tune the vote; the paper uses 0.5.
        drop_below_base_rate: Drop questions whose precision is at most the
            positive rate.
        hamming_threshold: Drop a question whose answers differ from a better
            question's on fewer than this share of samples.
        combiner: ``"vote"`` (the paper's) or ``"elasticnet"``.
        max_rules: Most questions the combiner may use.
        cv: Folds for tuning the vote or the elastic net.
        screening_size: Score every question on this many samples (int) or
            share of samples (float) first, and answer only the best
            ``screened_questions`` for the rest. ``None`` answers all.
        screened_questions: Questions kept after screening.
        pos_label: The positive class.
        random_state: Seed for examples, splits and folds.
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
        expert_questions: tuple[str, ...] = (),
        n_prompts: int = 25,
        questions_per_prompt: int = 10,
        n_examples: int = 20,
        example_class_ratio: ClassRatio = "balanced",
        generation_size: int | float = 500,
        beta: float = 0.5,
        drop_below_base_rate: bool = True,
        hamming_threshold: float = 0.15,
        combiner: Combiner = "vote",
        max_rules: int | None = None,
        cv: int = 5,
        screening_size: int | float | None = None,
        screened_questions: int = 20,
        pos_label: str | int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.generator = generator
        self.answerer = answerer
        self.instructions = instructions
        self.expert_questions = expert_questions
        self.n_prompts = n_prompts
        self.questions_per_prompt = questions_per_prompt
        self.n_examples = n_examples
        self.example_class_ratio = example_class_ratio
        self.generation_size = generation_size
        self.beta = beta
        self.drop_below_base_rate = drop_below_base_rate
        self.hamming_threshold = hamming_threshold
        self.combiner = combiner
        self.max_rules = max_rules
        self.cv = cv
        self.screening_size = screening_size
        self.screened_questions = screened_questions
        self.pos_label = pos_label
        self.random_state = random_state
        self.verbose = verbose

    example_class_ratio: ClassRatio
    combiner: Combiner
    instructions_: str
    questions_: pd.DataFrame
    rules_: tuple[str, ...]
    min_votes_: int
    combiner_: LinearCombiner | None
    threshold_: float

    def answers(self, X: TextInput) -> pd.DataFrame:
        """Probability of YES for each rule (columns) and sample (rows)."""
        matrix = self._run(
            X,
            "answering",
            lambda texts, session: self._answer(texts, self.rules_, session),
        )
        return pd.DataFrame(matrix, columns=pd.Index(self.rules_))

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        positive = self._positive(y)
        writer = self._models("generator", "generate")
        self._models("answerer", "answer")  # fail before paying for questions
        rng = np.random.default_rng(self.random_state)
        self.instructions_ = await self._instructions(session, writer)
        generation, scoring = self._split_generation(y, len(self.classes_), rng)
        questions = await self._write(texts, y, generation, writer, session, rng)
        if not questions:
            raise NoOutputError("The generator wrote no usable questions.")
        questions, answers = await self._score(
            texts, scoring, positive[scoring], questions, session, rng
        )
        # Ranking and cross-validation are CPU work: off the shared event loop.
        await asyncio.to_thread(self._select, questions, answers, positive[scoring])

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        answers = await self._answer(texts, self.rules_, session)
        return self._binary_proba(self._positive_scores(answers))

    def _positive_scores(
        self, answers: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self.combiner_ is not None:
            return self.combiner_.scores(np.nan_to_num(answers, nan=0.0))
        if not self.rules_:
            return np.zeros(len(answers))
        return as_yes(answers).mean(axis=1)

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: Classes
    ) -> list[UsageRecord]:
        writer = self._models("generator", "generate")
        answerer = self._models("answerer", "answer")
        rng = np.random.default_rng(self.random_state)
        _, scoring = self._split_generation(y, len(classes), rng)
        rows = take(texts, scoring)
        written = self.n_prompts * self.questions_per_prompt
        n_questions = written + len(self.expert_questions)
        planned = [
            *self._instructions_plan(),
            estimate_generations(
                writer, self.n_examples * mean_tokens(texts), calls=self.n_prompts
            ),
        ]
        if self.screening_size is None or n_questions <= self.screened_questions:
            return [
                *planned,
                estimate_answers(answerer, rows, [QUESTION_TOKENS] * n_questions),
            ]
        screen = count(self.screening_size, len(rows))
        return [
            *planned,
            estimate_answers(answerer, rows[:screen], [QUESTION_TOKENS] * n_questions),
            estimate_answers(
                answerer, rows[screen:], [QUESTION_TOKENS] * self.screened_questions
            ),
        ]

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        answerer = self._models("answerer", "answer")
        return [estimate_answers(answerer, texts, list(self.rules_))]

    def _split_generation(
        self, y: Labels, n_classes: int, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Samples for writing questions, and the rest for scoring them."""
        size = count(self.generation_size, len(y))
        generation = balanced_subset(y, n_classes, size, rng, max_share=0.5)
        scoring = np.setdiff1d(np.arange(len(y)), generation).astype(np.intp)
        return generation, scoring

    async def _write(
        self,
        texts: list[str],
        y: Labels,
        generation: npt.NDArray[np.intp],
        writer: tuple[Endpoint, ...],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> list[str]:
        """Questions from every prompt, deduplicated, then the expert's."""
        prompts = self._prompts(texts, y, generation, rng)
        written = await most_written(
            [
                session.generate(
                    writer,
                    p,
                    role="generator",
                    schema=_Questions,
                    instructions=QUESTION_INSTRUCTIONS,
                )
                for p in prompts
            ],
            "prompts",
        )
        candidates = [
            q for batch in written for q in batch.questions[: self.questions_per_prompt]
        ]
        return _unique([*candidates, *self.expert_questions])

    def _prompts(
        self,
        texts: list[str],
        y: Labels,
        generation: npt.NDArray[np.intp],
        rng: np.random.Generator,
    ) -> list[str]:
        """One prompt per batch of class-balanced examples from ``generation``."""
        batches = example_batches(
            y[generation],
            self.classes_,
            n_batches=self.n_prompts,
            batch_size=min(self.n_examples, len(generation)),
            class_ratio=self.example_class_ratio,
            rng=rng,
        )
        return [
            question_prompt(
                self._task(),
                self.instructions_,
                self.questions_per_prompt,
                examples_block(texts, y, self.classes_, generation[batch]),
            )
            for batch in batches
        ]

    async def _score(
        self,
        texts: list[str],
        scoring: npt.NDArray[np.intp],
        positive: npt.NDArray[np.bool_],
        questions: list[str],
        session: LLMSession,
        rng: np.random.Generator,
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        """Answers of the scoring samples, after screening if it is on."""
        rows = take(texts, scoring)
        if self.screening_size is None or len(questions) <= self.screened_questions:
            return questions, await self._answer(rows, questions, session)
        screen = rng.permutation(len(rows))[: count(self.screening_size, len(rows))]
        screened = await self._answer(take(rows, screen), questions, session)
        _, _, scores = question_stats(as_yes(screened), positive[screen], self.beta)
        best = np.argsort(-scores, kind="stable")[: self.screened_questions]
        kept = take(questions, best)
        rest = np.setdiff1d(np.arange(len(rows)), screen).astype(np.intp)
        answers = np.empty((len(rows), len(kept)))
        answers[screen] = screened[:, best]
        answers[rest] = await self._answer(take(rows, rest), kept, session)
        return kept, answers

    async def _answer(
        self, texts: Sequence[str], questions: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """P(YES) per sample (rows) and question (columns); NaN where unanswered."""
        return await yes_probabilities(
            session,
            self._models("answerer", "answer"),
            texts,
            questions,
            missing=np.nan,
        )

    def _select(
        self,
        questions: list[str],
        answers: npt.NDArray[np.float64],
        positive: npt.NDArray[np.bool_],
    ) -> None:
        yes = as_yes(answers)
        precision, recall, scores = question_stats(yes, positive, self.beta)
        self.questions_ = pd.DataFrame(
            {
                "question": questions,
                "precision": precision,
                "recall": recall,
                f"f{self.beta:g}": scores,
            }
        )
        self.combiner_, self.min_votes_ = None, 0
        if self.combiner == "vote":
            ranked, n_rules, min_votes = tune_vote(
                yes,
                positive,
                beta=self.beta,
                drop_below_base_rate=self.drop_below_base_rate,
                hamming_threshold=self.hamming_threshold,
                max_rules=self.max_rules,
                cv=self.cv,
                random_state=self.random_state,
            )
            used = ranked[:n_rules]
            self.min_votes_ = min_votes
            self.threshold_ = min_votes / n_rules if n_rules else math.inf
        else:
            used, self.threshold_ = self._fit_elasticnet(yes, answers, positive)
        self.rules_ = tuple(questions[j] for j in used)
        self.questions_["used"] = [j in used for j in range(len(questions))]

    def _fit_elasticnet(
        self,
        yes: npt.NDArray[np.bool_],
        answers: npt.NDArray[np.float64],
        positive: npt.NDArray[np.bool_],
    ) -> tuple[list[int], float]:
        """Rules weighted by elastic net, and its threshold.

        Questions are ranked on every scoring row before the combiner tunes its
        out-of-fold threshold, so the threshold is slightly optimistic.
        """
        ranked = rank(
            yes,
            positive,
            beta=self.beta,
            drop_below_base_rate=self.drop_below_base_rate,
            hamming_threshold=self.hamming_threshold,
        )[: self.max_rules]
        if not ranked:
            return [], math.inf
        fitted = fit_logistic(
            np.nan_to_num(answers[:, ranked], nan=0.0),
            positive,
            l1_ratios=(0.1, 0.5, 0.9),
            cv=self.cv,
            beta=self.beta,
            class_weight="balanced",
            random_state=self.random_state,
        )
        nonzero = np.flatnonzero(fitted.coef)
        self.combiner_ = LinearCombiner(
            fitted.coef[nonzero], fitted.intercept, fitted.threshold
        )
        return [ranked[j] for j in nonzero], fitted.threshold

    def _summary(self) -> Mapping[str, object]:
        return {
            "instructions": self.instructions_,
            "rules": list(self.rules_),
            "min_votes": self.min_votes_,
            "threshold": self.threshold_,
            "questions": self.questions_.to_dict("records"),
        }


def _unique(questions: Sequence[str]) -> list[str]:
    """Questions in order, without blanks or repeats (ignoring case)."""
    seen: set[str] = set()
    unique: list[str] = []
    for question in (q.strip() for q in questions):
        if question and question.lower() not in seen:
            seen.add(question.lower())
            unique.append(question)
    return unique


class _Questions(BaseModel):
    questions: list[str]
