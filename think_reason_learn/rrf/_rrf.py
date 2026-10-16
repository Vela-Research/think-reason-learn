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

from think_reason_learn.core._classifier import Labels, Proba, TRLClassifier
from think_reason_learn.core._combine import LinearCombiner, fit_logistic
from think_reason_learn.core._data import (
    ClassRatio,
    TextInput,
    count,
    example_batches,
    to_texts,
)
from think_reason_learn.core._prompts import examples_block
from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import Endpoint, LLMSession, Models, YesNoQuestion, run_sync

from ._prompts import INSTRUCTIONS_PURPOSE, QUESTION_INSTRUCTIONS, question_prompt
from ._selection import as_yes, question_stats, rank, tune_vote

type Combiner = Literal["vote", "elasticnet"]


class _Questions(BaseModel):
    questions: list[str]


class RRF(TRLClassifier):
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
        pos_label: str | None = None,
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
        texts = to_texts(X)[0]
        matrix = run_sync(self._answer(texts, self.rules_, self._session()))
        return pd.DataFrame(matrix, columns=pd.Index(self.rules_))

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        self._task()
        positive = self._positive(y)
        writer = self._models("generator", "generate")
        rng = np.random.default_rng(self.random_state)
        self.instructions_ = await self._instructions(session, writer)
        generation, scoring = self._split_generation(y, rng)
        questions = await self._write(texts, y, generation, writer, session, rng)
        questions, answers = await self._score(
            texts, scoring, positive[scoring], questions, session, rng
        )
        self._select(questions, answers, positive[scoring])

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        p = self._positive_scores(await self._answer(texts, self.rules_, session))
        proba = np.zeros((len(texts), 2))
        proba[:, self._pos_index()] = p
        proba[:, 1 - self._pos_index()] = 1 - p
        return proba

    def _decide(self, proba: Proba) -> Labels:
        positive = proba[:, self._pos_index()] >= self.threshold_ - 1e-9
        return np.where(positive, self._pos_index(), 1 - self._pos_index()).astype(
            np.intp
        )

    def _positive_scores(
        self, answers: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if not self.rules_:
            return np.zeros(len(answers))
        if self.combiner_ is not None:
            return self.combiner_.scores(np.nan_to_num(answers, nan=0.0))
        return as_yes(answers).mean(axis=1)

    def _positive(self, y: Labels) -> npt.NDArray[np.bool_]:
        if len(self.classes_) != 2:
            raise DataError(f"RRF predicts two classes; y has {len(self.classes_)}.")
        return y == self._pos_index()

    def _pos_index(self) -> int:
        if self.pos_label is None:
            return 1
        labels = [str(c) for c in self.classes_]
        if self.pos_label not in labels:
            raise ValueError(f"pos_label={self.pos_label!r} is not one of {labels}.")
        return labels.index(self.pos_label)

    def _split_generation(
        self, y: Labels, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Samples for writing questions, and the rest for scoring them."""
        size = count(self.generation_size, len(y))
        share = math.ceil(size / len(self.classes_))
        chosen = [
            rng.permutation(members)[: min(share, len(members) // 2)]
            for members in (np.flatnonzero(y == c) for c in range(len(self.classes_)))
        ]
        generation = np.sort(np.concatenate(chosen)).astype(np.intp)
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
        batches = example_batches(
            y[generation],
            self.classes_,
            n_batches=self.n_prompts,
            batch_size=min(self.n_examples, len(generation)),
            class_ratio=self.example_class_ratio,
            rng=rng,
        )
        prompts = [
            question_prompt(
                self._task(),
                self.instructions_,
                self.questions_per_prompt,
                examples_block(texts, y, self.classes_, generation[batch]),
            )
            for batch in batches
        ]
        written = await asyncio.gather(
            *(
                session.generate(
                    writer,
                    p,
                    role="generator",
                    schema=_Questions,
                    instructions=QUESTION_INSTRUCTIONS,
                )
                for p in prompts
            )
        )
        candidates = [
            q for batch in written for q in batch.questions[: self.questions_per_prompt]
        ]
        return _unique([*candidates, *self.expert_questions])

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
        rows = [texts[i] for i in scoring]
        if self.screening_size is None or len(questions) <= self.screened_questions:
            return questions, await self._answer(rows, questions, session)
        screen = rng.permutation(len(rows))[: count(self.screening_size, len(rows))]
        screened = await self._answer([rows[i] for i in screen], questions, session)
        _, _, scores = question_stats(as_yes(screened), positive[screen], self.beta)
        kept = [
            questions[j]
            for j in np.argsort(-scores, kind="stable")[: self.screened_questions]
        ]
        return kept, await self._answer(rows, kept, session)

    async def _answer(
        self, texts: Sequence[str], questions: Sequence[str], session: LLMSession
    ) -> npt.NDArray[np.float64]:
        """P(YES) per sample (rows) and question (columns); NaN where unanswered."""
        if not questions:
            return np.zeros((len(texts), 0))
        asked = {f"q{j}": YesNoQuestion(q) for j, q in enumerate(questions)}
        with self._progress(len(texts) * len(asked), "RRF answers") as progress:
            answers = await session.answer(
                self._models("answerer", "answer"),
                texts,
                asked,
                role="answerer",
                progress=progress,
            )
        return np.array(
            [
                [
                    a.probability() if (a := row[qid]) is not None else np.nan
                    for qid in asked
                ]
                for row in answers
            ]
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
            cs=10,
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
