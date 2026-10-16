"""GPTree: a decision tree whose splits are questions written by a language model.

Xiong, Ihlamur, Alican and Yin, "GPTree: Towards Explainable Decision-Making via
LLM-powered Decision Trees" (2024), https://arxiv.org/abs/2411.08257.

At each node the generator writes candidate questions from class-balanced
examples, the path so far and insights about each class; the answerer answers
them for the node's samples; and the question that most decreases impurity
splits the node. Nodes of one level grow concurrently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import graphviz
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from think_reason_learn.core._classifier import (
    Labels,
    Proba,
    TRLClassifier,
    binary_decision,
    positive_index,
)
from think_reason_learn.core._combine import best_threshold
from think_reason_learn.exceptions import LLMError
from think_reason_learn.core._data import (
    ClassRatio,
    TextInput,
    count,
    example_batches,
    to_texts,
)
from think_reason_learn.core._prompts import examples_block
from think_reason_learn.llm import (
    Answer,
    ChoiceQuestion,
    Endpoint,
    LLMSession,
    Model,
    Models,
    Question,
    UsageRecord,
    YesNoQuestion,
    estimate_answers,
    estimate_generations,
    resolve_models,
    run_sync,
)
from think_reason_learn.llm._estimate import QUESTION_TOKENS, mean_tokens, text_tokens

from ._prompts import (
    CHOICE,
    CRITIC_INSTRUCTIONS,
    INSIGHT_INSTRUCTIONS,
    INSTRUCTIONS_PURPOSE,
    MERGE_INSIGHT_INSTRUCTIONS,
    QUESTION_INSTRUCTIONS,
    YES_NO,
)
from ._split import Criterion, Split, best_split
from ._tree import Node, Tree

logger = logging.getLogger(__name__)

_MAX_INSIGHT_BATCHES: Final = 4
"""Insight batches per class.

ponytail: caps insight cost on large classes; raise if insights miss rare patterns.
"""


class _Written(BaseModel):
    text: str
    options: list[str] = Field(default_factory=list)


class _Questions(BaseModel):
    questions: list[_Written]


class GPTree(TRLClassifier):
    """A decision tree whose splits are questions written by a language model.

    Args:
        task: What to predict, in plain words.
        generator: Model(s) that write the instructions, insights and questions.
        answerer: Model(s) that answer the questions about each sample. Defaults
            to Jev; any reasoning model works too.
        critic: Review each node's questions before answering them: ``True``
            uses the generator, or give a model.
        instructions: Instructions for writing questions. Written from ``task``
            during ``fit`` when not given; see ``write_instructions``.
        expert_advice: Guidance added to every question-writing prompt.
        criterion: Impurity to minimise: ``"gini"`` or ``"entropy"``.
        max_depth: Deepest level; the paper uses 18. ``None`` for no limit.
        min_samples_split: Fewest samples a node needs to split (int), or that
            fraction of the training samples (float).
        min_samples_leaf: Fewest samples in each child (int), or that fraction
            of the training samples (float). The paper uses 0.5%.
        min_impurity_decrease: Smallest weighted impurity decrease a split needs.
        max_leaf_nodes: Most leaves; the best splits of each level go first.
        max_questions: Candidate questions written per node.
        max_choices: Answers per question: 2 for yes/no only, more to allow
            questions with that many options.
        n_examples: Labelled examples shown when writing a node's questions.
        example_class_ratio: Class shares among those examples.
        class_weight: Class weights in the impurity: ``"balanced"``, or a
            weight per label.
        insights: Summarise each class before growing the tree, as the paper does.
        insight_batch_size: Samples per insight prompt.
        pos_label: For two classes, the positive one (by default the second
            sorted label).
        beta: For two classes, leaves are called positive above the share of
            positives that maximises F-beta on the training samples, as the
            paper does (F0.5). More classes take the most likely class.
        random_state: Seed for the examples shown.
        verbose: Show progress when above 0.
    """

    def __init__(
        self,
        task: str | None = None,
        generator: Models | None = None,
        answerer: Models = "jev",
        *,
        critic: bool | Model = False,
        instructions: str | None = None,
        expert_advice: str | None = None,
        criterion: Criterion = "gini",
        max_depth: int | None = 18,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 0.005,
        min_impurity_decrease: float = 0.0,
        max_leaf_nodes: int | None = None,
        max_questions: int = 10,
        max_choices: int = 2,
        n_examples: int = 30,
        example_class_ratio: ClassRatio = "balanced",
        class_weight: Literal["balanced"] | Mapping[str, float] | None = None,
        insights: bool = True,
        insight_batch_size: int = 250,
        pos_label: str | int | None = None,
        beta: float = 0.5,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.task = task
        self.generator = generator
        self.answerer = answerer
        self.critic = critic
        self.instructions = instructions
        self.expert_advice = expert_advice
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.max_questions = max_questions
        self.max_choices = max_choices
        self.n_examples = n_examples
        self.example_class_ratio = example_class_ratio
        self.class_weight = class_weight
        self.insights = insights
        self.insight_batch_size = insight_batch_size
        self.pos_label = pos_label
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose

    _instructions_purpose = INSTRUCTIONS_PURPOSE
    criterion: Criterion
    example_class_ratio: ClassRatio
    class_weight: Literal["balanced"] | Mapping[str, float] | None
    instructions_: str
    insights_: str
    tree_: Tree
    threshold_: float | None

    def apply(self, X: TextInput) -> npt.NDArray[np.intp]:
        """The leaf each sample reaches."""
        return run_sync(self._route(to_texts(X)[0], self._session()))

    def plot(self) -> graphviz.Source:
        """The tree as a Graphviz figure (needs the Graphviz program to render)."""
        return graphviz.Source(self.tree_.export_graphviz(list(self.classes_)))

    def export_graphviz(self) -> str:
        """The tree in Graphviz DOT."""
        return self.tree_.export_graphviz(list(self.classes_))

    def prune(self, node_id: int) -> None:
        """Turn a node into a leaf, dropping everything below it."""
        self.tree_.prune(node_id)

    async def _afit(self, texts: list[str], y: Labels, session: LLMSession) -> None:
        self._task()
        grower = _Grower.start(self, texts, y, session)
        self.instructions_ = await self._instructions(session, grower.writer)
        self.insights_ = await grower.insights() if self.insights else ""
        self.tree_ = await grower.grow()
        self.threshold_ = grower.threshold()

    def _fit_plan(
        self, texts: list[str], y: Labels, classes: npt.NDArray[np.object_]
    ) -> list[UsageRecord]:
        """Insights, then each level's questions, answered by every sample."""
        writer = self._models("generator", "generate")
        n, example = len(texts), mean_tokens(texts)
        smallest = max(
            2, count(self.min_samples_split, n), 2 * count(self.min_samples_leaf, n)
        )
        levels = self._levels(n, smallest)
        nodes = sum(min(2**k, n // smallest) for k in range(levels))
        prompt = self.n_examples * example
        planned = [
            *self._instructions_plan(),
            *self._insights_plan(writer, y, example),
            estimate_generations(writer, prompt, calls=nodes),
        ]
        if (critic := _critic(self.critic, writer)) is not None:
            planned.append(
                estimate_generations(critic, prompt, calls=nodes, role="critic")
            )
        questions = [QUESTION_TOKENS] * self.max_questions
        answerer = self._models("answerer", "answer")
        return [*planned, estimate_answers(answerer, texts, questions, times=levels)]

    def _predict_plan(self, texts: list[str]) -> list[UsageRecord]:
        """One question per level along each path, as often as in training."""
        nodes = self.tree_.nodes
        asked = [node.question.text for node in nodes if node.question is not None]
        if not asked:
            return []
        visits = sum(n.n_samples for n in nodes if not n.is_leaf) / nodes[0].n_samples
        question = round(sum(map(text_tokens, asked)) / len(asked))
        answerer = self._models("answerer", "answer")
        return [estimate_answers(answerer, texts, [question], times=visits)]

    def _levels(self, n: int, smallest: int) -> int:
        """Levels split, halving nodes until they are too small to split."""
        if n < smallest:
            return 0
        limits = [int(math.log2(n / smallest)) + 1]
        if self.max_depth is not None:
            limits.append(self.max_depth)
        if self.max_leaf_nodes is not None:
            limits.append(math.ceil(math.log2(max(self.max_leaf_nodes, 2))))
        return min(limits)

    def _insights_plan(
        self, writer: tuple[Endpoint, ...], y: Labels, example: float
    ) -> list[UsageRecord]:
        if not self.insights:
            return []
        size = self.insight_batch_size
        batches = [
            min(math.ceil(int(np.count_nonzero(y == c)) / size), _MAX_INSIGHT_BATCHES)
            for c in np.unique(y)
        ]
        calls = sum(b + 1 if b > 1 else b for b in batches)  # several are merged
        return [estimate_generations(writer, size * example, calls=calls)]

    def _decide(self, proba: Proba) -> Labels:
        if self.threshold_ is None:
            return super()._decide(proba)
        pos = positive_index(self.classes_, self.pos_label)
        return binary_decision(proba, pos, self.threshold_)

    async def _apredict_proba(self, texts: list[str], session: LLMSession) -> Proba:
        leaves = await self._route(texts, session)
        return np.stack([self.tree_.proba(int(leaf)) for leaf in leaves])

    async def _route(
        self, texts: list[str], session: LLMSession
    ) -> npt.NDArray[np.intp]:
        """Walk each sample down the tree, one level at a time."""
        answerer = self._models("answerer", "answer")
        at = np.zeros(len(texts), dtype=np.intp)
        frontier = {0: np.arange(len(texts))}
        while frontier := {
            k: v for k, v in frontier.items() if not self.tree_.nodes[k].is_leaf
        }:
            moves = await asyncio.gather(
                *(
                    self._descend(node_id, idx, texts, answerer, session)
                    for node_id, idx in frontier.items()
                )
            )
            frontier = {}
            for move in moves:
                for child, idx in move.items():
                    at[idx] = child
                    frontier[child] = idx
        return at

    async def _descend(
        self,
        node_id: int,
        idx: npt.NDArray[np.intp],
        texts: list[str],
        answerer: tuple[Endpoint, ...],
        session: LLMSession,
    ) -> dict[int, npt.NDArray[np.intp]]:
        node = self.tree_.nodes[node_id]
        assert node.question is not None and node.default is not None
        try:
            answers = await session.answer(
                answerer, [texts[i] for i in idx], {"q": node.question}, role="answerer"
            )
        except LLMError:
            logger.warning(
                f"No answers at node {node_id}; following its largest branch."
            )
            return {node.children[node.default]: idx}
        children = np.array(
            [
                node.children.get(
                    _label(a["q"]) or node.default,
                    node.children[node.default],
                )
                for a in answers
            ]
        )
        return {int(c): idx[children == c] for c in np.unique(children)}

    def _summary(self) -> Mapping[str, object]:
        frame = self.tree_.to_frame(list(self.classes_)).reset_index()
        nodes: object = json.loads(frame.to_json(orient="records") or "[]")
        return {
            "instructions": self.instructions_,
            "insights": self.insights_,
            "nodes": nodes,
        }


@dataclass(slots=True)
class _Grower:
    """The state of one ``fit``: data, models and the tree being grown."""

    model: GPTree
    texts: list[str]
    y: Labels
    weights: npt.NDArray[np.float64]
    session: LLMSession
    writer: tuple[Endpoint, ...]
    critic: tuple[Endpoint, ...] | None
    answerer: tuple[Endpoint, ...]
    rng: np.random.Generator
    min_leaf: int
    min_split: int
    leaf_of: npt.NDArray[np.intp]
    tree: Tree = field(default_factory=Tree)

    @classmethod
    def start(
        cls, model: GPTree, texts: list[str], y: Labels, session: LLMSession
    ) -> _Grower:
        writer = model._models("generator", "generate")
        n = len(texts)
        return cls(
            model,
            texts,
            y,
            _sample_weights(y, model.classes_, model.class_weight),
            session,
            writer,
            _critic(model.critic, writer),
            model._models("answerer", "answer"),
            np.random.default_rng(model.random_state),
            max(1, count(model.min_samples_leaf, n)),
            max(2, count(model.min_samples_split, n)),
            np.zeros(n, dtype=np.intp),
        )

    async def insights(self) -> str:
        """A short summary of each class, as the paper's insight step."""
        summaries = await asyncio.gather(
            *(self._summarise(c) for c in range(len(self.model.classes_)))
        )
        return "\n\n".join(
            f"{label}:\n{summary}"
            for label, summary in zip(self.model.classes_, summaries)
            if summary
        )

    async def grow(self) -> Tree:
        """Grow level by level; the best splits of a level go first."""
        root = self.tree.add(
            depth=0,
            value=self._value(np.arange(len(self.texts))),
            n_samples=len(self.texts),
        )
        frontier = [(root, np.arange(len(self.texts)))]
        budget = self.model.max_leaf_nodes
        while frontier and (budget is None or self.tree.n_leaves < budget):
            proposals = await asyncio.gather(
                *(self._propose(node, idx) for node, idx in frontier)
            )
            chosen = sorted(
                (p for p in proposals if p is not None), key=lambda p: -p.split.decrease
            )
            frontier = [
                child
                for p in self._within_leaf_budget(chosen)
                for child in self._apply(p)
            ]
        return self.tree

    async def _summarise(self, c: int) -> str:
        members = self.rng.permutation(np.flatnonzero(self.y == c))
        size = self.model.insight_batch_size
        batches = [members[i : i + size] for i in range(0, len(members), size)][
            :_MAX_INSIGHT_BATCHES
        ]
        if not batches:
            return ""
        label = self.model.classes_[c]
        summaries = await asyncio.gather(
            *(
                self.session.generate(
                    self.writer,
                    f"Task: {self.model.task}\n\nSamples labelled {label}:\n\n"
                    + "\n\n".join(self.texts[i] for i in batch),
                    role="generator",
                    instructions=INSIGHT_INSTRUCTIONS,
                )
                for batch in batches
            )
        )
        if len(summaries) == 1:
            return summaries[0]
        return await self.session.generate(
            self.writer,
            "\n\n".join(summaries),
            role="generator",
            instructions=MERGE_INSIGHT_INSTRUCTIONS,
        )

    async def _propose(self, node: Node, idx: npt.NDArray[np.intp]) -> _Proposal | None:
        if self._is_final(node, idx):
            return None
        questions = await self._questions(node, idx)
        node.candidates = tuple(q.text for q in questions.values())
        if not questions:
            return None
        try:
            answers = await self.session.answer(
                self.answerer, [self.texts[i] for i in idx], questions, role="answerer"
            )
        except LLMError:
            logger.warning(f"No answers at node {node.id}; it stays a leaf.")
            return None
        labels = {qid: [_label(a.get(qid)) for a in answers] for qid in questions}
        split = best_split(
            labels,
            self.y[idx],
            self.weights[idx],
            n_classes=len(self.model.classes_),
            criterion=self.model.criterion,
            min_samples_leaf=self.min_leaf,
            min_impurity_decrease=self.model.min_impurity_decrease,
            total_weight=float(self.weights.sum()),
        )
        if split is None:
            return None
        return _Proposal(node, idx, split, questions[split.qid], labels[split.qid])

    def _is_final(self, node: Node, idx: npt.NDArray[np.intp]) -> bool:
        max_depth = self.model.max_depth
        return bool(
            (max_depth is not None and node.depth >= max_depth)
            or len(idx) < max(self.min_split, 2 * self.min_leaf)
            or np.count_nonzero(node.value) <= 1
        )

    async def _questions(
        self, node: Node, idx: npt.NDArray[np.intp]
    ) -> dict[str, Question]:
        prompt = self._node_prompt(node, idx)
        instructions = QUESTION_INSTRUCTIONS.format(
            options=YES_NO
            if self.model.max_choices <= 2
            else CHOICE.format(max_choices=self.model.max_choices)
        )
        written = await self.session.generate(
            self.writer,
            prompt,
            role="generator",
            schema=_Questions,
            instructions=instructions,
        )
        if self.critic is not None:
            reviewed = await self.session.generate(
                self.critic,
                f"{prompt}\n\nCandidate questions:\n{written.model_dump_json()}",
                role="critic",
                schema=_Questions,
                instructions=CRITIC_INSTRUCTIONS,
            )
            written = (
                reviewed
                if 0 < len(reviewed.questions) <= len(written.questions)
                else written
            )
        return self._valid(node, written)

    def _node_prompt(self, node: Node, idx: npt.NDArray[np.intp]) -> str:
        model = self.model
        [chosen] = example_batches(
            self.y[idx],
            model.classes_,
            n_batches=1,
            batch_size=min(model.n_examples, len(idx)),
            class_ratio=model.example_class_ratio,
            rng=self.rng,
        )
        path = (
            "\n".join(f"- {q} -> {a}" for q, a in self.tree.path(node.id))
            or "(this is the root)"
        )
        parts = [
            f"Task: {model.task}",
            f"Instructions:\n{model.instructions_}",
            f"What tells the classes apart overall:\n{model.insights_}"
            if model.insights_
            else "",
            f"Expert advice:\n{model.expert_advice}" if model.expert_advice else "",
            f"Path to this node:\n{path}",
            "Examples at this node:\n\n"
            + examples_block(self.texts, self.y, model.classes_, idx[chosen]),
            f"Write up to {model.max_questions} questions that best separate the "
            "classes here.",
        ]
        return "\n\n".join(part for part in parts if part)

    def _valid(self, node: Node, written: _Questions) -> dict[str, Question]:
        """Usable, distinct questions not on the path, at most ``max_questions``."""
        asked = {q for q, _ in self.tree.path(node.id)}
        valid: dict[str, Question] = {}
        for item in written.questions:
            text = item.text.strip()
            if not text or text in asked or text in {q.text for q in valid.values()}:
                continue
            options = tuple(dict.fromkeys(o.strip() for o in item.options if o.strip()))
            if not options:
                valid[f"n{node.id}q{len(valid)}"] = YesNoQuestion(text)
            elif 3 <= len(options) <= self.model.max_choices:
                valid[f"n{node.id}q{len(valid)}"] = ChoiceQuestion(text, options)
        return dict(list(valid.items())[: self.model.max_questions])

    def _within_leaf_budget(self, proposals: Sequence[_Proposal]) -> list[_Proposal]:
        budget = self.model.max_leaf_nodes
        if budget is None:
            return list(proposals)
        leaves, kept = self.tree.n_leaves, []
        for proposal in proposals:
            added = len(proposal.split.branches) - 1
            if leaves + added <= budget:
                kept.append(proposal)
                leaves += added
        return kept

    def _apply(self, proposal: _Proposal) -> list[tuple[Node, npt.NDArray[np.intp]]]:
        node, idx, split = proposal.node, proposal.idx, proposal.split
        node.question, node.default = proposal.question, split.default
        routed = np.array(
            [split.default if a is None else a for a in proposal.labels], dtype=object
        )
        children = []
        for branch in split.branches:
            child_idx = idx[routed == branch]
            child = self.tree.add(
                depth=node.depth + 1,
                value=self._value(child_idx),
                n_samples=len(child_idx),
                parent=node.id,
                branch=branch,
            )
            node.children[branch] = child.id
            self.leaf_of[child_idx] = child.id
            children.append((child, child_idx))
        return children

    def threshold(self) -> float | None:
        """For two classes, the leaf share of positives that maximises F-beta."""
        if len(self.model.classes_) != 2:
            return None
        pos = positive_index(self.model.classes_, self.model.pos_label)
        shares = np.array([self.tree.proba(int(leaf))[pos] for leaf in self.leaf_of])
        return best_threshold(shares, self.y == pos, self.model.beta)[0]

    def _value(self, idx: npt.NDArray[np.intp]) -> tuple[float, ...]:
        counts = np.bincount(
            self.y[idx], weights=self.weights[idx], minlength=len(self.model.classes_)
        )
        return tuple(float(c) for c in counts)


@dataclass(frozen=True, slots=True)
class _Proposal:
    node: Node
    idx: npt.NDArray[np.intp]
    split: Split
    question: Question
    labels: list[str | None]


def _critic(
    critic: bool | Model, writer: tuple[Endpoint, ...]
) -> tuple[Endpoint, ...] | None:
    if critic is False:
        return None
    if critic is True:
        return writer
    return resolve_models(critic, param="critic", task="generate")


def _sample_weights(
    y: Labels,
    classes: npt.NDArray[np.object_],
    class_weight: Literal["balanced"] | Mapping[str, float] | None,
) -> npt.NDArray[np.float64]:
    if class_weight is None:
        return np.ones(len(y))
    if class_weight == "balanced":
        counts = np.bincount(y, minlength=len(classes))
        per_class = len(y) / (len(classes) * np.maximum(counts, 1))
    else:
        per_class = np.array(
            [class_weight.get(str(c), 1.0) for c in classes], dtype=float
        )
    return per_class[y]


def _label(answer: Answer | None) -> str | None:
    return answer.label if answer is not None else None
