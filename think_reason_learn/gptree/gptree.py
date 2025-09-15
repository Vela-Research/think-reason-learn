from __future__ import annotations

import asyncio
from typing import List, Dict, Literal, cast, Type, Tuple
import logging
import contextlib
from dataclasses import dataclass, field
from uuid import uuid4
from copy import deepcopy

from pydantic import BaseModel, create_model
import numpy as np
import numpy.typing as npt
import pandas as pd

from ._types import QuestionType, Criterion
from ._prompts import INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS
from ._prompts import num_questions_tag, QUESTION_ANSWER_INSTRUCTIONS
from think_reason_learn.core.llms.schemas import LLMPriority, TokenCount
from think_reason_learn.core.llms import llm
from think_reason_learn.core._exceptions import DataError


logger = logging.getLogger(__name__)

IndexArray = npt.NDArray[np.intp]


@dataclass(slots=True)
class NodeQuestion:
    value: str
    choices: List[str]
    question_type: QuestionType
    df_column: str = field(default_factory=lambda: str(uuid4()))
    score: float | None = None

    def __hash__(self):
        return hash((self.value, self.question_type))

    def __eq__(self, other: object):
        if not isinstance(other, NodeQuestion):
            return False
        return self.value == other.value and self.question_type == other.question_type


class Question(BaseModel):
    value: str
    choices: List[str]
    question_type: QuestionType


class Questions(BaseModel):
    questions: List[Question]
    memory_context: str


class Answer(BaseModel):
    answer: str


@dataclass(slots=True)
class Node:
    """
    A Node represents a decision point in GPTree.

    Attributes
    ----------
    id: int
        The id of the node.

    label: str
        The label of the node.

    question: NodeQuestion, default=None
        The chosen question at this node.

    questions: List[NodeQuestion], default=[]
        The questions that have been generated for this node.

    memory_context: str | None = None
        The memory context of the node.

    split_ratios: Tuple[int, ...], default=None
        The split ratios of the samples that are associated with this node.

    gini: float, default=0.0
        The Gini impurity of the node.

    class_distribution: Dict[str, int], default=None
        Distribution of classes at this node. Class label as key and number of samples as value.

    children: List[Node], default=None
        The children of this node.
    """

    id: int
    label: str
    question: NodeQuestion | None = None
    questions: List[NodeQuestion] = field(default_factory=list)
    memory_context: str | None = None
    split_ratios: Tuple[int, ...] | None = None
    gini: float = 0.0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    children: List[Node] | None = None
    parent_id: int | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children or []) == 0


class GPTree:
    """LLM based decision tree classifier."""

    def __init__(
        self,
        question_gen_llmp: List[LLMPriority],
        critic_llmp: List[LLMPriority],
        question_gen_instructions_llmp: List[LLMPriority],
        question_gen_temperature: float = 0.0,
        critic_temperature: float = 0.0,
        question_gen_instructions_gen_temperature: float = 0.0,
        criterion: Criterion = "gini",
        max_depth: int | None = None,
        max_width: int = 3,
        min_samples_leaf: int = 1,
        max_clusters: int | None = None,
        llm_semaphore_limit: int = 3,
        min_question_candidates: int = 3,
        max_question_candidates: int = 10,
        expert_advice: str | None = None,
        context_samples_n: int = 30,
        class_ratio: Dict[str, int] | Literal["balanced"] = "balanced",
        use_critic: bool = False,
    ):
        """
        GPTree.

        Attributes
        ----------
        question_gen_llmp: List[LLMPriority]
            The LLM priority for question generation.

        critic_llmp: List[LLMPriority]
            The LLM priority for critic LLM.

        question_gen_instructinos_llmp: List[LLMPriority]
            The LLM priority for generating question generation instructions.

        question_gen_temperature: float = 0.0,
            The temperature for question generation.

        critic_temperature: float = 0.0,
            The temperature for critic.

        question_gen_instructions_gen_temperature: float = 0.0,
            The temperature for generating question generation instructions.

        criterion: Criterion, default="gini"
            The function to measure the quality of a split. Currently the only
            supported criteria is "gini" for the Gini impurity,
            see :ref:`tree_mathematical_formulation`.

        max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until further splitting will cause leaves to
            contain less than min_samples_leaf samples.

        max_width: int, default=2
            The maximum width of the tree

        min_samples_leaf: int, default=1
            The minimum number of samples required to be at a leaf node.

        max_clusters: int, default=None
            The maximum out degree of a node in the case of cluster. If None, then
            nodes of the tree can have arbitrary out degree.

        llm_semaphore_limit: int, default=3
            The maximum number of concurrent LLM calls on the fly.

        X_column: str, default="data"
            The column name of the X dataframe that contains the data.

        questions: Dict[int, Question], default={}
            The questions that have been generated for the tree. The key is the node id.

        num_classes: int | None
            The number of classes in the dataset.

        token_usage: List[TokenCount]
            The token usage for the tree.

        min_question_candidates: int, default=3
            The minimum number of questions to generate for a node.

        max_question_candidates: int, default=10
            The maximum number of questions to generate for a node.

        expert_advice: str | None = None
            Long term advice from a human expert for question generations.

        context_samples_n: int, default=30
            The number of samples to use for context in question generations.
            Could be lower if the number of samples is less than this number.

        class_ratio: Dict[str, int] | Literal["balanced"], default="balanced"
            The ratio of classes to use for context in question generations.
            If "balanced", then the ratio will be balanced.
            If a dictionary, then the ratio will be the ratio of classes.

        use_critic: bool, default=False
            Whether to use a critic to evaluate the quality of the questions.
        """
        self.question_gen_llmp = question_gen_llmp
        self.question_gen_instructions_llmp = question_gen_instructions_llmp
        self.critic_llmp = critic_llmp
        self.question_gen_temperature = question_gen_temperature
        self.critic_temperature = critic_temperature
        self.question_gen_instructions_gen_temperature = (
            question_gen_instructions_gen_temperature
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_clusters = max_clusters
        self.llm_semaphore_limit = llm_semaphore_limit
        self.token_usage: List[TokenCount] = []
        self.min_question_candidates = min_question_candidates
        self.max_question_candidates = max_question_candidates
        self.expert_advice = expert_advice
        self.context_samples_n = context_samples_n
        self.class_ratio = class_ratio
        self.use_critic = use_critic

        self._num_classes: int | None = None
        self._X: pd.DataFrame | None = None
        self._y: npt.NDArray[np.str_] | None = None
        self._nodes: Dict[int, Node] = {}
        self._root: Node | None = None
        self._node_counter = 0
        self._llm_semaphore = asyncio.Semaphore(llm_semaphore_limit)
        self._question_gen_instructions_template: str | None = None
        self._critic_instructions_template: str | None = None
        self._X_column = "data"

    @property
    def num_classes(self) -> int | None:
        return self.num_classes

    def _gini(self, indices: IndexArray) -> float:
        _, counts = np.unique(self._y[indices], return_counts=True)  # type: ignore
        probs = counts / counts.sum()
        return float(1 - np.sum(probs**2))

    def _get_next_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def _get_num_questions(
        self,
        node_depth: int,
        node_size: int,
        max_candidates: int,
        min_candidates: int,
    ):
        if self._X is None:
            raise ValueError("X must be set")

        size_factor = node_size / self._X.shape[0] if self._X.shape[0] > 0 else 1
        if self.max_depth is not None:
            # Linear decay by depth, adjusted by size
            depth_factor = 1 - (node_depth / self.max_depth)
            scale = max(depth_factor * size_factor, min_candidates / max_candidates)
        else:
            # Fallback: Pure size-based linear decay (from max at full size to min at 0 size)
            scale = max(size_factor, min_candidates / max_candidates)
        return max(min_candidates, int(max_candidates * scale))

    async def init_question_gen_instructions_template(
        self,
        instructions_template: str | None = None,
        task_description: str | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Set the question generation instructions template or provide a task description to generate the instructions template using an LLM.

        Parameters
        ----------
        instructions_template: str
            The instructions template to set. Must contain the tag '<number_of_questions>'. If None, the task description will be used to generate the instructions template using an LLM.

        task_description: str
            The task description to use to generate the instructions template using an LLM.

        verbose: bool
            Whether to print verbose output.

        Returns
        -------
        None
            The function sets the question generation instructions template.
        """
        assert (
            instructions_template is not None or task_description is not None
        ), "Either instructions_template or task_description must be provided"

        if instructions_template:
            if num_questions_tag not in instructions_template:
                raise ValueError(
                    f"Instructions_template must contain the tag '{num_questions_tag}'"
                    "This tag will be replaced with the number of questions to generate at each generation run"
                )
            else:
                self._question_gen_instructions_template = instructions_template
                return

        async with self._llm_semaphore:
            response = await llm.respond(
                query=f"The task is to build a decision tree model for this: {task_description}",
                llm_priority=self.question_gen_instructions_llmp,
                response_format=str,
                instructions=INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS,
                temperature=self.question_gen_instructions_gen_temperature,
                verbose=verbose,
            )
        if not response.response:
            raise ValueError(
                "Failed to generate question generation instructions"
                "Try refining the task description or change the models for generating question generation instructions, `self.question_gen_instructions_llmp`"
            )
        if response.average_confidence is not None:
            logger.info(
                f"Generated question generation instructions with confidence {response.average_confidence}"
            )
        else:
            logger.info(
                "Generated question generation instructions. Could not track confidence of instructions."
            )

        self._question_gen_instructions_template = response.response

    def _get_question_gen_instructions(self, num_questions: int) -> str:

        if not self._question_gen_instructions_template:
            raise ValueError(
                "Question generation instructions template is not set"
                "Set the template using `init_question_gen_instructions_template`"
            )

        return self._question_gen_instructions_template.replace(
            num_questions_tag, str(num_questions)
        )

    async def _generate_questions(
        self,
        sample_indices: IndexArray,
        context_memory: str | None,
        node_depth: int,
        verbose: bool,
    ) -> Questions:
        if self._X is None or self._y is None:
            raise ValueError("X and y must be set")
        if sample_indices.shape[0] == 0:
            raise ValueError("Sample indices is empty")

        num_questions = self._get_num_questions(
            node_depth=node_depth,
            node_size=sample_indices.shape[0],
            min_candidates=self.min_question_candidates,
            max_candidates=self.max_question_candidates,
        )

        instructions = self._get_question_gen_instructions(num_questions)
        query = ""

        X = cast(pd.DataFrame, self._X.iloc[sample_indices])
        y = self._y[sample_indices]
        y_unique = np.unique(y)

        class_ratio_fractions: Dict[str, float] = {}  # type linter not to say unbound
        if isinstance(self.class_ratio, dict):
            class_ratio = {k: v for k, v in self.class_ratio.items() if k in y_unique}
            total_ratio = sum(class_ratio.values())
            class_ratio_fractions = {k: v / total_ratio for k, v in class_ratio.items()}

        grouped = dict(tuple(X.groupby(y)))  # type: ignore
        for label in y_unique:
            if isinstance(self.class_ratio, dict):
                n_samples = self.context_samples_n * class_ratio_fractions[label]
            elif self.class_ratio == "balanced":
                n_samples = self.context_samples_n / y_unique.shape[0]

            samples = grouped[label].sample(int(n_samples))  # type: ignore
            samples_array = samples[self._X_column].to_numpy(dtype=str)  # type: ignore
            samples_array = cast(npt.NDArray[np.str_], samples_array)
            samples_str = "\n".join(
                [f"{i}: {s}" for i, s in enumerate(samples_array, 1)]
            )
            query += f"Samples with label {label.capitalize()}:\n{samples_str}\n"

        if self.expert_advice is not None:
            query += f"Consider this expert advice: {self.expert_advice}\n"
        if context_memory is not None:
            query += f"Context from previous nodes: {context_memory}"

        async with self._llm_semaphore:
            response = await llm.respond(
                query=query,
                llm_priority=self.question_gen_llmp,
                response_format=Questions,
                instructions=instructions,
                temperature=self.question_gen_temperature,
                verbose=verbose,
            )
        self.token_usage.append(
            TokenCount(
                provider=response.provider_model.provider,
                model=response.provider_model.model,
                value=response.total_tokens,
            )
        )
        questions = response.response
        if questions is None:
            raise ValueError(
                f"Could not generate questions\nQuery: {query}\n\nInstructions: {instructions}"
            )

        if self.use_critic:
            # TODO: Implement critic
            pass

        return questions

    def _make_answer_model(self, choices: List[str]) -> Type[Answer]:
        field_type = Literal[tuple(choices)]
        model = create_model("Answer", answer=(field_type, ...))
        return cast(Type[Answer], model)

    async def _answer_question_for_row(
        self,
        idx: int,
        row: pd.Series,
        question: NodeQuestion,
        verbose: bool,
    ) -> Tuple[int, Answer] | None:
        AnswerModel = self._make_answer_model(question.choices)
        sample = cast(str, row[self._X_column])

        try:
            response = await llm.respond(
                llm_priority=self.question_gen_llmp,
                query=question.value,
                instructions=QUESTION_ANSWER_INSTRUCTIONS,
                response_format=AnswerModel,
                temperature=self.question_gen_temperature,
                verbose=verbose,
            )
            if response.response is None:
                raise ValueError("No response from LLM")
            if response.average_confidence is not None:
                logger.debug(
                    f"Confidence: {response.average_confidence} Answered question: '{question.value}' for sample: {sample}"
                )
            else:
                logger.debug(
                    f"Could not track confidence of answer. Answered question: '{question.value}' for sample: {sample}"
                )
            return idx, response.response

        except Exception as _:
            logger.warning(
                f"Error answering question: '{question.value}' for sample: {sample}",
                exc_info=True,
            )

    async def _answer_question(
        self,
        question: NodeQuestion,
        sample_indices: IndexArray,
        verbose: bool,
    ) -> None:
        """Answer a question for a subset of self._X inplace."""
        if self._X is None or self._y is None:
            raise ValueError("X and y must be set")

        X = cast(pd.DataFrame, self._X.iloc[sample_indices])
        tasks: List[asyncio.Task[Tuple[int, Answer] | None]] = []
        try:
            tasks = [
                asyncio.create_task(
                    self._answer_question_for_row(
                        idx=int(str(row[0])),
                        row=row[1],
                        question=question,
                        verbose=verbose,
                    )
                )
                for row in X.iterrows()
            ]
            for task in asyncio.as_completed(tasks):
                with contextlib.suppress(asyncio.CancelledError):
                    result = await task
                    if result is None:
                        continue
                    idx, answer = result
                    self._X.at[idx, question.df_column] = answer.answer
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _build_tree(
        self,
        id: int,
        parent_id: int | None,
        depth: int,
        label: str,
        sample_indices: IndexArray,
        verbose: bool,
    ) -> Node:
        X = self._X
        y = self._y
        if X is None or y is None or self._num_classes is None:
            raise ValueError("X, y, and num_classes must be set")

        sample_y = y[sample_indices]

        uniq, counts = np.unique(sample_y, return_counts=True)
        class_distribution: Dict[str, int] = {
            str(u): int(c) for u, c in zip(uniq, counts)
        }

        # NOTE: Could just calculate, but we want to be consistent
        gini = self._gini(sample_indices)

        if (
            depth >= (self.max_depth or float("inf"))
            or sample_y.shape[0] < self._num_classes * self.min_samples_leaf
            or uniq.shape[0] == 1
        ):
            logger.info(f"Terminal node at depth {depth} with id {id}")
            return Node(
                id=id,
                parent_id=parent_id,
                label=label,
                gini=gini,
                class_distribution=class_distribution,
                question=None,
                questions=[],
                memory_context=None,
                split_ratios=None,
                children=None,
            )

        memory_context = (
            self._nodes[parent_id].memory_context if parent_id is not None else None
        )
        questions = await self._generate_questions(
            sample_indices=sample_indices,
            context_memory=memory_context,
            node_depth=depth,
            verbose=verbose,
        )
        logger.info(f"Generated {len(questions.questions)} questions for node {id}")
        chosen_question: NodeQuestion | None = None

        for llm_question in questions.questions:
            node_question = NodeQuestion(**llm_question.model_dump())
            logger.info(f"Answering question (Node {id}): {node_question.value}")

            if node_question.question_type == "INFERENCE":
                await self._answer_question(node_question, sample_indices, verbose)
                groups = X.groupby(node_question.df_column).indices  # type: ignore
                df_split_indices = [
                    np.array(groups.get(val, []), dtype=np.intp)
                    for val in node_question.choices
                ]

            elif node_question.question_type == "CODE":
                # TODO: Get code execution environment
                continue
            else:
                logger.warning(
                    f"Invalid question type: {node_question.question_type}. Skipping..."
                )
                continue

            min_gini, total, skip = 1, 0, False

            for sub_indices in df_split_indices:
                if len(sub_indices) >= self.min_samples_leaf:
                    total += len(sub_indices)
                else:
                    skip = True
                    logger.debug(f"Not enough samples to split. Terminating node {id}.")

            if not skip:
                weighted_gini = sum(
                    ((len(si) * self._gini(si) / total) for si in df_split_indices)
                )
                if weighted_gini < min_gini:
                    min_gini = weighted_gini
                    chosen_question = node_question
                    logger.debug(f"New minimum found: {min_gini} for Node {id}")

                node_question.score = weighted_gini

        if chosen_question is None:
            logger.debug(f"Terminating at Node {id}. No valid split found.")
            return Node(
                id=id,
                parent_id=parent_id,
                label=label,
                gini=gini,
                class_distribution=class_distribution,
                split_ratios=None,
                question=None,
                questions=[NodeQuestion(**q.model_dump()) for q in questions.questions],
                memory_context=questions.memory_context,
                children=None,
            )

        children: List[Node] = []
        split_ratios: List[int] = []
        for choice in chosen_question.choices:
            choice_df = cast(pd.DataFrame, X[X[chosen_question.df_column] == choice])
            split_ratios.append(choice_df.shape[0])
            if choice_df.empty:
                continue

            indices: IndexArray = choice_df.index.to_numpy(dtype=np.intp)  # type: ignore
            new_id = self._get_next_node_id()

            child_node = await self._build_tree(
                id=new_id,
                parent_id=id,
                depth=depth + 1,
                label=choice,
                sample_indices=indices,
                verbose=verbose,
            )
            children.append(child_node)

        return Node(
            id=id,
            parent_id=parent_id,
            label=label,
            gini=gini,
            question=chosen_question,
            questions=[NodeQuestion(**q.model_dump()) for q in questions.questions],
            memory_context=questions.memory_context,
            split_ratios=tuple(split_ratios),
            class_distribution=class_distribution,
            children=children,
        )

    async def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[np.str_],
        copy_data: bool = True,
        verbose: bool = True,
    ):
        if X.columns.tolist() != [self._X_column]:
            raise DataError(
                f"X must be a pandas DataFrame with a single column named {self._X_column}"
            )

        if copy_data:
            self._X = deepcopy(X)  # type: ignore
            self._y = deepcopy(y)
        else:
            self._X = X  # type: ignore
            self._y = y

        self._root = await self._build_tree(
            id=0,
            parent_id=None,
            depth=0,
            label="root",
            sample_indices=X.index.to_numpy(dtype=np.intp),  # type: ignore
            verbose=verbose,
        )
