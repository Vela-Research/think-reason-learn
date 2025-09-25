"""GPTree.

LLM based decision tree classifier.
"""

from __future__ import annotations

import os
from os import PathLike
import asyncio
from typing import List, Dict, Literal, Sequence, cast, Type, Tuple
from typing import Set, Any, AsyncGenerator
import logging
from dataclasses import dataclass, field, asdict
from uuid import uuid4
from copy import deepcopy
from pathlib import Path
import datetime
import re

import orjson
from pydantic import BaseModel, create_model, Field
import numpy as np
import numpy.typing as npt
import pandas as pd

from think_reason_learn.core.llms import LLMChoice, TokenCounter, llm
from think_reason_learn.core.exceptions import DataError, LLMError, CorruptionError
from ._types import QuestionType, Criterion
from ._prompts import INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS
from ._prompts import num_questions_tag, QUESTION_ANSWER_INSTRUCTIONS
from ._prompts import CUMULATIVE_MEMORY_INSTRUCTIONS


logger = logging.getLogger(__name__)

IndexArray = npt.NDArray[np.intp]


@dataclass(slots=True)
class NodeQuestion:
    """A question for generated at a node."""

    value: str = field(metadata={"description": "The question text."})
    choices: List[str] = field(
        metadata={"description": "The answer choices for the question."}
    )
    question_type: QuestionType = field(
        metadata={"description": "The type of the question."}
    )
    df_column: str = field(
        default_factory=lambda: str(uuid4()),
        metadata={
            "description": "The name of the column in the tree's training data that "
            "contains the question."
        },
    )
    score: float | None = field(
        default=None,
        metadata={
            "description": "The score of the question in the node. E.g, gini impurity."
        },
    )

    def __hash__(self):
        return hash((self.value, self.question_type))

    def __eq__(self, other: object):
        if not isinstance(other, NodeQuestion):
            return False
        return self.value == other.value and self.question_type == other.question_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node question to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NodeQuestion:
        """Convert a dictionary to a node question."""
        return cls(**d)


class Question(BaseModel):
    value: str
    choices: List[str]
    question_type: QuestionType


class Questions(BaseModel):
    questions: List[Question]
    cumulative_memory: str = Field(..., description=CUMULATIVE_MEMORY_INSTRUCTIONS)


class Answer(BaseModel):
    answer: str


@dataclass(slots=True)
class Node:
    """A Node represents a decision point in GPTree."""

    id: int = field(metadata={"description": "The id of the node."})
    label: str = field(metadata={"description": "The label of the node."})
    question: NodeQuestion | None = field(
        default=None,
        metadata={
            "description": "The chosen question at this node. E.g, "
            "if criterion is gini, then the question with the lowest gini impurity."
        },
    )
    questions: List[NodeQuestion] = field(
        default_factory=list,
        metadata={
            "description": "All questions that have been generated at this node."
        },
    )
    cumulative_memory: str | None = field(
        default=None,
        metadata={
            "description": "The cumulative memory context generated at this node."
        },
    )
    split_ratios: Tuple[int, ...] | None = field(
        default=None,
        metadata={
            "description": "Samples split at this node per answer choice of the "
            "chosen question."
        },
    )
    gini: float = field(
        default=0.0,
        metadata={"description": "The Gini impurity of the node if criterion is gini."},
    )
    class_distribution: Dict[str, int] = field(
        default_factory=dict,
        metadata={"description": "Distribution of classes at this node."},
    )
    children: List[Node] = field(
        default_factory=list, metadata={"description": "The children of this node."}
    )
    parent_id: int | None = field(
        default=None, metadata={"description": "The id of the parent node."}
    )

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return len(self.children or []) == 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Node:
        """Convert a dictionary to a node."""
        if question := d.get("question"):
            d["question"] = NodeQuestion.from_dict(question)
        if questions := d.get("questions"):
            d["questions"] = [NodeQuestion.from_dict(q) for q in questions]
        if children := d.get("children"):
            d["children"] = [cls.from_dict(c) for c in children]
        return cls(**d)


@dataclass(slots=True)
class BuildTask:
    node_id: int
    parent_id: int | None
    depth: int
    label: str
    sample_indices: IndexArray

    def to_dict(self) -> Dict[str, Any]:
        """Convert the build task to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BuildTask:
        """Convert a dictionary to a build task."""
        if sample_indices := d.get("sample_indices"):
            d["sample_indices"] = np.array(sample_indices, dtype=np.intp)
        return cls(**d)


class GPTree:
    """LLM based decision tree classifier.

    Note that GPTree auto saves the tree after each node is built.

    Args:
        qgen_llmc: LLMs to use for question generation, in priority order.
        critic_llmc: LLMs to use for question critique, in priority order.
        qgen_instr_llmc: LLMs for generating instructions.
        qanswer_llmc: LLMs to use for answering questions, in priority order.
            If None, use qgen_llmc.
        qgen_temperature: Sampling temperature for question generation.
        critic_temperature: Sampling temperature for critique.
        qgen_instr_gen_temperature: Sampling temperature for generating
            instructions.
        qanswer_temperature: Sampling temperature for answering questions.
        criterion: Splitting criterion. Currently only "gini".
        max_depth: Maximum tree depth. If None, grow until pure/min samples.
        max_node_width: Maximum children per node.
        min_samples_leaf: Minimum samples per leaf.
        llm_semaphore_limit: Max concurrent LLM calls.
        min_question_candidates: Min number of questions per node.
        max_question_candidates: Max number of questions per node. Max 15
        expert_advice: Human-provided hints for generation.
        n_samples_as_context: Number of samples used as context in generation.
        class_ratio: Strategy for class sampling ("balanced" or dict of ratios).
        use_critic: Whether to critique generated questions.
        save_path: Directory to save checkpoints/models.
        name: Name of the tree instance.
        random_state: Random seed.
    """

    def __init__(
        self,
        qgen_llmc: List[LLMChoice],
        critic_llmc: List[LLMChoice],
        qgen_instr_llmc: List[LLMChoice],
        qanswer_llmc: List[LLMChoice] | None = None,
        qgen_temperature: float = 0.0,
        critic_temperature: float = 0.0,
        qgen_instr_gen_temperature: float = 0.0,
        qanswer_temperature: float = 0.0,
        criterion: Criterion = "gini",
        max_depth: int | None = None,
        max_node_width: int = 3,
        min_samples_leaf: int = 1,
        llm_semaphore_limit: int = 5,
        min_question_candidates: int = 3,
        max_question_candidates: int = 10,
        expert_advice: str | None = None,
        n_samples_as_context: int = 30,
        class_ratio: Dict[str, int] | Literal["balanced"] = "balanced",
        use_critic: bool = False,
        save_path: str | PathLike[str] | None = None,
        name: str | None = None,
        random_state: int | None = None,
    ):
        locals_dict = deepcopy(locals())
        del locals_dict["self"]
        self._verify_input_data(**locals_dict)

        self.name: str = self._get_name(name)
        self.save_path: Path = self._set_save_path(save_path)

        self.qgen_llmc = qgen_llmc
        self.critic_llmc = critic_llmc
        self.qgen_instr_llmc = qgen_instr_llmc
        self.qanswer_llmc = qanswer_llmc or qgen_llmc
        self.qgen_temperature = qgen_temperature
        self.qanswer_temperature = qanswer_temperature
        self.critic_temperature = critic_temperature
        self.qgen_instr_gen_temperature = qgen_instr_gen_temperature
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_node_width = max_node_width
        self.min_samples_leaf = min_samples_leaf
        self.llm_semaphore_limit = llm_semaphore_limit
        self.min_question_candidates = min_question_candidates
        self.max_question_candidates = max_question_candidates
        self._expert_advice = expert_advice
        self.n_samples_as_context = n_samples_as_context
        self.class_ratio = class_ratio
        self.use_critic = use_critic
        self.random_state = random_state

        self._token_counter: TokenCounter = TokenCounter()

        self._classes: List[str] | None = None
        self._X: pd.DataFrame | None = None
        self._y: npt.NDArray[np.str_] | None = None
        self._nodes: Dict[int, Node] = {}
        self._node_counter = 0
        self.__llm_semaphore: asyncio.Semaphore | None = None
        self._qgen_instructions_template: str | None = None
        self._critic_instructions_template: str | None = None
        self._stop_training: bool = False
        self._task_description: str | None = None

        self._frontier: List[BuildTask] = []  # Frontier for resumable training

        self._llm_semaphore = asyncio.Semaphore(llm_semaphore_limit)

    def _verify_input_data(self, **kwargs: Any) -> None:
        """Verify the input data."""
        qgen_temperature = kwargs["qgen_temperature"]
        critic_temperature = kwargs["critic_temperature"]
        qgen_instr_gen_temperature = kwargs["qgen_instr_gen_temperature"]
        qanswer_temperature = kwargs["qanswer_temperature"]
        max_depth = kwargs["max_depth"]
        max_node_width = kwargs["max_node_width"]
        min_samples_leaf = kwargs["min_samples_leaf"]
        llm_semaphore_limit = kwargs["llm_semaphore_limit"]
        min_question_candidates = kwargs["min_question_candidates"]
        max_question_candidates = kwargs["max_question_candidates"]
        n_samples_as_context = kwargs["n_samples_as_context"]
        class_ratio = kwargs["class_ratio"]

        if qgen_temperature < 0 or qgen_temperature > 2:
            raise ValueError("qgen_temperature must be >= 0 and <= 2")
        if critic_temperature < 0 or critic_temperature > 2:
            raise ValueError("critic_temperature must be >= 0 and <= 2")
        if qgen_instr_gen_temperature < 0 or qgen_instr_gen_temperature > 2:
            raise ValueError("qgen_instr_gen_temperature must be >= 0 and <= 2")
        if qanswer_temperature < 0 or qanswer_temperature > 2:
            raise ValueError("qanswer_temperature must be >= 0 and <= 2")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 if provided")
        if max_node_width < 2 or max_node_width > 10:
            raise ValueError("max_node_width must be >= 2 and <= 10")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if llm_semaphore_limit < 1:
            raise ValueError("llm_semaphore_limit must be >= 1")
        if min_question_candidates < 1:
            raise ValueError("min_question_candidates must be >= 1")
        if max_question_candidates < min_question_candidates:
            raise ValueError(
                "min_question_candidates must be <= max_question_candidates"
            )
        if max_question_candidates < 1 or max_question_candidates > 15:
            raise ValueError("max_question_candidates must be >= 1 and <= 15")
        if n_samples_as_context < 1:
            raise ValueError("n_samples_as_context must be >= 1")
        if isinstance(class_ratio, dict):
            if not class_ratio:
                raise ValueError("class_ratio dict cannot be empty")
            if not all(isinstance(k, str) for k in class_ratio):  # type: ignore
                raise ValueError("class_ratio keys must be strings")
            if not all(isinstance(v, int) and v > 0 for v in class_ratio.values()):  # type: ignore
                raise ValueError("class_ratio values must be positive integers")
        elif class_ratio != "balanced":
            raise ValueError("class_ratio must be 'balanced' or a dict[str, int]")

    def get_root_id(self) -> int | None:
        """Get the root node id."""
        return (
            0
            if 0 in self._nodes
            else next(
                (nid for nid, n in self._nodes.items() if n.parent_id is None), None
            )
        )

    def get_node(self, node_id: int) -> Node | None:
        """Get the node by id."""
        return self._nodes.get(node_id)

    def get_training_data(self) -> pd.DataFrame | None:
        """Get the training data."""
        if self._X is None or self._y is None:
            return None
        return pd.concat([self._X, pd.Series(self._y, name="y")], axis=1)

    def get_questions(self) -> pd.DataFrame | None:
        """Get all questions generated in the tree."""
        questions = [
            {"node_id": n.id, **asdict(q)}
            for n in self._nodes.values()
            for q in n.questions
        ]
        return pd.DataFrame(questions) if questions else None

    @property
    def token_usage(self) -> TokenCounter:
        """Get the token counter for the GPTree."""
        return self._token_counter

    @property
    def question_gen_instructions_template(self) -> str | None:
        """Get the question generation instructions template."""
        return self._qgen_instructions_template

    @property
    def critic_instructions_template(self) -> str | None:
        """Get the critic instructions template."""
        return self._critic_instructions_template

    @property
    def task_description(self) -> str | None:
        """Description of the classification task."""
        return self._task_description

    def view_node(
        self,
        node_id: int,
        format: Literal["png", "svg"] = "png",
        add_all_questions: bool = False,
        truncate_length: int | None = 140,
    ) -> bytes:
        """Render subtree rooted at node_id as PNG/SVG bytes.

        Args:
            node_id: Root node ID for the subtree visualization.
            format: Output image format ('png' or 'svg').
            add_all_questions: Include all generated questions in node display.
            truncate_length: Maximum text length before truncation.
                None disables truncation.

        Returns:
            Rendered subtree image data as bytes.

        Raises:
            ValueError: If node_id doesn't exist in the tree.
            ImportError: If graphviz package is not installed.
        """
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found in the tree")

        try:
            from graphviz import Digraph  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'graphviz' Python package is required. "
                "Install it with `pip install graphviz`."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Graphviz failed to load. Ensure both the Python "
                "package (`pip install graphviz`) "
                "and the system Graphviz binary are installed."
            ) from e

        def _escape(text: str) -> str:
            return (
                text.replace("\\", "\\\\")
                .replace("\n", "\\n")
                .replace("\r", " ")
                .replace("\t", " ")
                .replace('"', '\\"')
            )

        def _truncate(text: str) -> str:
            if truncate_length is None:
                return text
            if len(text) <= truncate_length:
                return text
            return text[: truncate_length - 3] + "..."

        dot = Digraph(
            name=f"GPTree_{self.name}_{node_id}",
            format=format,
            graph_attr={"rankdir": "TB", "concentrate": "true"},
        )  # type: ignore

        visited: set[int] = set()
        added_edges: set[tuple[int, int]] = set()
        queue: List[Node] = [node]

        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)

            label_lines: List[str] = [
                f"id={current.id}",
                f"label={current.label}",
                f"gini={current.gini:.3f}",
            ]
            if add_all_questions and current.questions:
                all_questions = [q.value for q in current.questions]
                label_lines.append(f"[{_truncate('; '.join(all_questions))}]")
            if current.question is not None:
                q = current.question
                label_lines.append(_truncate(f"Q: {q.value}"))
                label_lines.append(f"choices: {', '.join(q.choices)}")
            if current.class_distribution:
                dist_str = ", ".join(
                    f"{k}:{v}" for k, v in current.class_distribution.items()
                )
                label_lines.append(_truncate(f"Label distribution: {dist_str}"))
            if current.split_ratios is not None:
                label_lines.append(
                    "Answer distribution: "
                    f"{', '.join(str(x) for x in current.split_ratios)}"
                )

            node_label = _escape("\n".join(label_lines))
            dot.node(  # type: ignore
                str(current.id),
                node_label,
                shape="box",
                style="rounded,filled",
                fillcolor="lightgrey",
                fontsize="10",
            )

            seen_child_ids: set[int] = set()
            for child in current.children or []:
                if child.id in seen_child_ids:
                    if child.id not in visited:
                        queue.append(child)
                    continue
                seen_child_ids.add(child.id)

                edge_key = (current.id, child.id)
                if edge_key not in added_edges:
                    added_edges.add(edge_key)
                    edge_label = _escape(str(child.label))
                    dot.edge(str(current.id), str(child.id), label=edge_label)  # type: ignore

                if child.id not in visited:
                    queue.append(child)

        return dot.pipe(format=format)  # type: ignore

    def stop(self) -> None:
        """Stop training process."""
        self._stop_training = True

    def advice(self, advice: str | None) -> Literal["Advice taken", "Advice cleared"]:
        """Set context/advice for question generations.

        Args:
            advice: The advice to set. If None, the advice is cleared.

        Returns:
            "Advice taken" if advice is set, "Advice cleared" if advice is cleared.
        """
        if advice is None:
            self._expert_advice = None
            return "Advice cleared"

        self._expert_advice = advice
        return "Advice taken"

    @property
    def expert_advice(self) -> str | None:
        """Expert advice set on the tree."""
        return self._expert_advice

    def _get_name(self, name: str | None) -> str:
        if name is None:
            name = str(uuid4()).replace("-", "_")
            logger.debug(f"No name provided. Assigned name: {name}")

        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError("Name must be only alphanumeric and underscores")
        return name

    def _set_save_path(self, save_path: str | PathLike[str] | None) -> Path:
        if save_path is None:
            return (Path(os.getcwd()) / "gptrees" / self.name).resolve()
        else:
            save_path = Path(save_path).resolve()
            if save_path.is_file():
                raise ValueError("Please provide a directory, not a file.")
            return save_path

    @property
    def classes(self) -> List[str] | None:
        """Classes the tree is trying to classify."""
        return self._classes

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
            # Fallback: Pure size-based linear decay (max at full size to min at 0 size)
            scale = max(size_factor, min_candidates / max_candidates)
        return max(min_candidates, int(max_candidates * scale))

    async def set_tasks(
        self,
        instructions_template: str | None = None,
        task_description: str | None = None,
    ) -> str:
        """Initialize question generation instructions template.

        This sets the task description for the tree.
        Either sets a custom template or generates one from task description using LLM.
        For most users, LLM generation is recommended over custom templates.

        Args:
            instructions_template: Custom template to use. Must contain
                '<number_of_questions>' tag. If None, generates template
                from task_description using LLM.
            task_description: Description of classification task to help LLM generate
                the template.

        Returns:
            The question generation instructions template.

        Raises:
            ValueError: If template missing required tag or generation fails.
            AssertionError: If both parameters are None.
        """
        assert (
            instructions_template is not None or task_description is not None
        ), "Either instructions_template or task_description must be provided"

        if instructions_template:
            if num_questions_tag not in instructions_template:
                raise ValueError(
                    f"instructionstemplate must contain the tag '{num_questions_tag}' "
                    "This tag will be replaced with the number of questions to "
                    "generate at each generation run"
                )
            else:
                self._qgen_instructions_template = instructions_template
                return instructions_template

        async with self._llm_semaphore:
            response = await llm.respond(
                query=f"Build a decision tree for:\n{task_description}",
                llm_priority=self.qgen_llmc,
                response_format=str,
                instructions=INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS,
                temperature=self.qgen_instr_gen_temperature,
            )
        await self._token_counter.append(
            provider=response.provider_model.provider,
            model=response.provider_model.model,
            value=response.total_tokens,
            caller="GPTree.set_tasks",
        )
        if not response.response:
            raise ValueError(
                "Failed to generate question generation instructions. "
                "Try refining the task description or change the models "
                "for generating question generation instructions (self.qgen_llmc)."
            )
        elif num_questions_tag not in response.response:
            raise ValueError(
                "Failed to generate a valid question generation "
                "instructions template. Please try again."
            )

        if response.average_confidence is not None:
            logger.info(
                "Generated question generation instructions with "
                f"confidence {response.average_confidence}"
            )
        else:
            logger.info(
                "Generated question generation instructions. Could not track "
                "confidence of instructions."
            )

        self._task_description = task_description
        self._qgen_instructions_template = response.response
        return response.response

    def _get_question_gen_instructions(self, num_questions: int) -> str:
        if not self._qgen_instructions_template:
            raise ValueError(
                "Question generation instructions template is not set"
                "Set the template using `set_tasks`"
            )

        instructions = self._qgen_instructions_template.replace(
            num_questions_tag, str(num_questions)
        )
        instructions += (
            "\n\nIMPORTANT: Limit the number of choices per question "
            f"to {self.max_node_width}."
        )
        return instructions

    async def _generate_questions(
        self,
        sample_indices: IndexArray,
        cumulative_memory: str | None,
        node_depth: int,
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
            total_ratio = sum(self.class_ratio.values())
            class_ratio_fractions = {
                k: v / total_ratio for k, v in self.class_ratio.items()
            }

        grouped = dict(tuple(X.groupby(y)))  # type: ignore
        for label in y_unique:
            if isinstance(self.class_ratio, dict):
                n_samples = self.n_samples_as_context * class_ratio_fractions[label]
            elif self.class_ratio == "balanced":
                n_samples = self.n_samples_as_context / y_unique.shape[0]
            else:
                raise ValueError(f"Invalid class ratio: {self.class_ratio}")

            n_samples = min(n_samples, len(grouped[label]))
            samples = grouped[label].sample(  # type: ignore
                int(n_samples), random_state=self.random_state
            )

            samples_str = ""
            for sample in samples.to_dict(orient="records"):  # type: ignore
                sample_str = "\n".join([f"{col}: {val}" for col, val in sample.items()])  # type: ignore
                samples_str += f"\n{sample_str};"
            query += f"Samples with label {label}".upper() + f"{samples_str}\n\n"

        # Query woould be like:
        # SAMPLES WITH LABEL NO
        # data: Bob, a 25-year-old man based in New York City...
        # ages: 25;
        # data: David, a 40-year-old man living in Chicago...
        # ages: 35;

        # SAMPLES WITH LABEL YES
        # data: Charlie, a 35-year-old man in Los Angeles...
        # ages: 30;
        # data: Grace, a 27-year-old woman living in Austin, Texas...
        # ages: 27;

        if self._expert_advice is not None:
            query += f"Consider this expert advice: {self._expert_advice}\n"
        cumulative_memory = (
            cumulative_memory or "No cumulative memory yet. This is the root node."
        )
        query += f"Cumulative advice from previous nodes: {cumulative_memory}"

        async with self._llm_semaphore:
            response = await llm.respond(
                query=query,
                llm_priority=self.qgen_llmc,
                response_format=Questions,
                instructions=instructions,
                temperature=self.qgen_temperature,
            )
        await self._token_counter.append(
            provider=response.provider_model.provider,
            model=response.provider_model.model,
            value=response.total_tokens,
            caller="GPTree._generate_questions",
        )
        questions = response.response
        if questions is None:
            raise LLMError(
                "Could not generate questions. Please try "
                "again or with a different llm."
                f"\nQuery: {query[:200]}..."
                f"\n\nCumulative memory: {cumulative_memory[:200]}..."
                f"\n\nInstructions: {instructions[:200]}..."
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
        idx: Any,
        sample: str,
        question: NodeQuestion,
        token_counter: TokenCounter,
    ) -> Tuple[Any, Answer] | None:
        AnswerModel = self._make_answer_model(question.choices)

        try:
            async with self._llm_semaphore:
                response = await llm.respond(
                    llm_priority=self.qanswer_llmc,
                    query=f"Query: {question.value}\n\nSample: {sample}",
                    instructions=QUESTION_ANSWER_INSTRUCTIONS,
                    response_format=AnswerModel,
                    temperature=self.qanswer_temperature,
                )
            await token_counter.append(
                provider=response.provider_model.provider,
                model=response.provider_model.model,
                value=response.total_tokens,
                caller="GPTree._answer_question_for_row",
            )
            if response.response is None:
                raise LLMError("No response from LLM")
            if response.average_confidence is not None:
                logger.debug(
                    f"Confidence: {response.average_confidence} "
                    f"Answered question: '{question.value}' for sample: {sample}"
                )
            else:
                logger.debug(
                    "Could not track confidence of answer. "
                    f"Answered question: '{question.value}' for sample: {sample}"
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
    ) -> None:
        """Answer a question for a subset of self._X inplace."""
        if self._X is None or self._y is None:
            raise ValueError("X and y must be set")

        X = cast(pd.DataFrame, self._X.iloc[sample_indices])

        answers_buffer: Dict[Any, str] = {}
        completion_queue: asyncio.Queue[None] = asyncio.Queue()

        async def worker(idx_label: Any, sample_text: str) -> None:
            try:
                result = await self._answer_question_for_row(
                    idx=idx_label,
                    sample=sample_text,
                    question=question,
                    token_counter=self._token_counter,
                )
                if result is not None:
                    idx_done, answer = result
                    answers_buffer[idx_done] = answer.answer
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            finally:
                completion_queue.put_nowait(None)

        row_iter = iter(X.itertuples(index=True, name=None))
        in_flight = 0

        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(self.llm_semaphore_limit):
                    try:
                        row_tuple = next(row_iter)
                    except StopIteration:
                        break

                    idx_label = row_tuple[0]
                    sample_text = "\n".join(
                        [f"{col}: {val}" for col, val in zip(X.columns, row_tuple[1:])]  # type: ignore
                    )
                    tg.create_task(worker(idx_label, sample_text))
                    in_flight += 1

                while in_flight > 0:
                    await completion_queue.get()
                    in_flight -= 1
                    try:
                        row_tuple = next(row_iter)
                    except StopIteration:
                        continue
                    idx_label = row_tuple[0]
                    sample_text = "\n".join(
                        [f"{col}: {val}" for col, val in zip(X.columns, row_tuple[1:])]  # type: ignore
                    )
                    tg.create_task(worker(idx_label, sample_text))
                    in_flight += 1
        finally:
            if answers_buffer:
                series_update = pd.Series(answers_buffer, dtype=object)
                index_update = cast(pd.Index, series_update.index)
                self._X.loc[index_update, question.df_column] = series_update

    async def _build_tree(
        self,
        id: int,
        parent_id: int | None,
        depth: int,
        label: str,
        sample_indices: IndexArray,
    ) -> AsyncGenerator[Node, None]:
        if not any(t.node_id == id for t in self._frontier):
            self._frontier.append(
                BuildTask(
                    node_id=id,
                    parent_id=parent_id,
                    depth=depth,
                    label=label,
                    sample_indices=sample_indices,
                )
            )
            self.save()

        if self._stop_training:
            return

        X = self._X
        y = self._y

        if X is None or y is None:
            raise ValueError("X, y,  must be set")

        sample_y = y[sample_indices]

        uniq, counts = np.unique(sample_y, return_counts=True)
        class_distribution: Dict[str, int] = {
            str(u): int(c) for u, c in zip(uniq, counts)
        }

        if self._classes is None:
            self._classes = list(uniq)

        gini = self._gini(sample_indices)

        if (
            depth >= (self.max_depth or float("inf"))
            or sample_y.shape[0] < len(self._classes) * self.min_samples_leaf
            or uniq.shape[0] == 1
        ):
            logger.info(f"Terminal node at depth {depth} with id {id}")
            existing = self._nodes.get(id)
            if existing is None:
                node = Node(
                    id=id,
                    parent_id=parent_id,
                    label=label,
                    gini=gini,
                    class_distribution=class_distribution,
                    question=None,
                    questions=[],
                    cumulative_memory=None,
                    split_ratios=None,
                    children=[],
                )
                self._nodes[id] = node
            else:
                node = existing
                node.parent_id = parent_id
                node.label = label
                node.gini = gini
                node.class_distribution = class_distribution
                node.question = None
                node.questions = []
                node.cumulative_memory = None
                node.split_ratios = None
                node.children = []

            # Ensure the parent points to the current node instance
            if parent_id is not None:
                parent = self._nodes.get(parent_id)
                if parent is not None:
                    replaced = False
                    parent.children = parent.children or []
                    for i, ch in enumerate(parent.children):
                        if ch.id == node.id:
                            parent.children[i] = node
                            replaced = True
                            break
                    if not replaced:
                        parent.children.append(node)
            self._frontier = [t for t in self._frontier if t.node_id != id]
            self.save()
            yield node
            return

        cumulative_memory = (
            self._nodes[parent_id].cumulative_memory if parent_id is not None else None
        )
        questions = await self._generate_questions(
            sample_indices=sample_indices,
            cumulative_memory=cumulative_memory,
            node_depth=depth,
        )
        logger.info(f"Generated {len(questions.questions)} questions for node {id}")
        chosen_question: NodeQuestion | None = None

        node_questions: List[NodeQuestion] = []
        for llm_question in questions.questions:
            node_question = NodeQuestion(**llm_question.model_dump())
            node_questions.append(node_question)
            logger.info(f"Answering question (Node {id}): {node_question.value}")

            if node_question.question_type == "INFERENCE":
                await self._answer_question(node_question, sample_indices)
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
            existing = self._nodes.get(id)
            if existing is None:
                node = Node(
                    id=id,
                    parent_id=parent_id,
                    label=label,
                    gini=gini,
                    class_distribution=class_distribution,
                    split_ratios=None,
                    question=None,
                    questions=node_questions,
                    cumulative_memory=questions.cumulative_memory,
                    children=[],
                )
                self._nodes[id] = node
            else:
                node = existing
                node.parent_id = parent_id
                node.label = label
                node.gini = gini
                node.class_distribution = class_distribution
                node.split_ratios = None
                node.question = None
                node.questions = node_questions
                node.cumulative_memory = questions.cumulative_memory
                node.children = []

            # Ensure the parent points to the current node instance
            if parent_id is not None:
                parent = self._nodes.get(parent_id)
                if parent is not None:
                    replaced = False
                    parent.children = parent.children or []
                    for i, ch in enumerate(parent.children):
                        if ch.id == node.id:
                            parent.children[i] = node
                            replaced = True
                            break
                    if not replaced:
                        parent.children.append(node)

            self._frontier = [t for t in self._frontier if t.node_id != id]
            self.save()
            yield node
            return

        choice_dfs = {
            choice: cast(pd.DataFrame, X[X[chosen_question.df_column] == choice])
            for choice in chosen_question.choices
        }
        split_ratios = tuple([df.shape[0] for df in choice_dfs.values()])

        existing = self._nodes.get(id)
        if existing is None:
            node = Node(
                id=id,
                parent_id=parent_id,
                label=label,
                gini=gini,
                question=chosen_question,
                questions=node_questions,
                cumulative_memory=questions.cumulative_memory,
                split_ratios=split_ratios,
                class_distribution=class_distribution,
                children=[],
            )
            self._nodes[id] = node
        else:
            node = existing
            node.parent_id = parent_id
            node.label = label
            node.gini = gini
            node.question = chosen_question
            node.questions = node_questions
            node.cumulative_memory = questions.cumulative_memory
            node.split_ratios = split_ratios
            node.class_distribution = class_distribution
            node.children = []

        # Ensure the parent points to the current node instance
        if parent_id is not None:
            parent = self._nodes.get(parent_id)
            if parent is not None:
                replaced = False
                parent.children = parent.children or []
                for i, ch in enumerate(parent.children):
                    if ch.id == node.id:
                        parent.children[i] = node
                        replaced = True
                        break
                if not replaced:
                    parent.children.append(node)

        choice_indices_list: List[Tuple[str, IndexArray]] = [
            (
                choice,
                df.index.to_numpy(dtype=np.intp),  # type: ignore
            )
            for choice, df in choice_dfs.items()
            if not df.empty
        ]

        new_id_map: Dict[str, int] = {
            choice: self._get_next_node_id() for choice, _ in choice_indices_list
        }

        # Replace current node frontier with children frontiers
        self._frontier = [t for t in self._frontier if t.node_id != id]
        for choice, indices in choice_indices_list:
            if not any(t.node_id == new_id_map[choice] for t in self._frontier):
                self._frontier.append(
                    BuildTask(
                        node_id=new_id_map[choice],
                        parent_id=id,
                        depth=depth + 1,
                        label=choice,
                        sample_indices=indices,
                    )
                )
        self.save()

        for choice, indices in choice_indices_list:
            if self._stop_training:
                yield node
                return

            async for child_node in self._build_tree(
                id=new_id_map[choice],
                parent_id=id,
                depth=depth + 1,
                label=choice,
                sample_indices=indices,
            ):
                node.children = node.children or []
                # Replace same-id child with latest instance, or append if new
                replaced_child = False
                for i, ch in enumerate(node.children):
                    if ch.id == child_node.id:
                        node.children[i] = child_node
                        replaced_child = True
                        break
                if not replaced_child:
                    node.children.append(child_node)
                yield node

        return

    def _set_data(self, X: pd.DataFrame, y: Sequence[str], copy_data: bool) -> None:
        if not all(isinstance(item, str) for item in y):  # type: ignore
            raise DataError("y must be a sequence of strings")

        if len(y) != X.shape[0]:
            raise DataError("y and X must have the same number of rows")

        if isinstance(self.class_ratio, dict):
            if set(y) != set(self.class_ratio.keys()):
                raise DataError("y must have the same values as class_ratio keys")

        y_array = np.array(y, dtype=np.str_)

        if copy_data:
            self._X = deepcopy(X).reset_index(drop=True)  # type: ignore
            self._y = deepcopy(y_array)
        else:
            self._X = X.reset_index(drop=True)  # type: ignore
            self._y = y_array

        self._nodes = {}
        self._node_counter = 0
        self._frontier = []
        self._stop_training = False

    async def fit(
        self,
        X: pd.DataFrame | None = None,
        y: Sequence[str] | None = None,
        *,
        copy_data: bool = True,
        reset: bool = False,
    ) -> AsyncGenerator[Node, None]:
        """Train or resume tree construction as an async generator.

        Args:
            X: Training features. Required on first run or with reset=True.
            y: Training labels. Required on first run or with reset=True.
            copy_data: Whether to copy input data.
            reset: Clear existing state and restart from root.

        Yields:
            Node: Updated nodes during tree construction.

        Raises:
            ValueError: If data requirements aren't met or invalid reset usage.
        """
        if reset:
            if X is None or y is None:
                raise ValueError("reset=True requires X and y")
            self._set_data(X, y, copy_data)
        else:
            if X is not None or y is not None:
                if self._X is not None or self._y is not None:
                    raise ValueError(
                        "Data already set on tree. Explicitly pass reset=True "
                        "to replace data and restart training."
                    )
                if X is None or y is None:
                    raise ValueError("Both X and y must be provided together")
                self._set_data(X, y, copy_data)

        if self._X is None or self._y is None:
            raise ValueError(
                "No data found on tree. Provide X and y (or reset=True with X,y)"
            )

        self._stop_training = False

        if len(self._nodes) == 0:  # train from scratch
            indices: IndexArray = self._X.index.to_numpy(dtype=np.intp)  # type: ignore
            async for updated_root in self._build_tree(
                id=0,
                parent_id=None,
                depth=0,
                label="root",
                sample_indices=indices,
            ):
                if self._stop_training:
                    break
                yield updated_root
            return

        while self._frontier:  # resume from frontier
            task = self._frontier.pop(0)
            async for updated_node in self._build_tree(
                id=task.node_id,
                parent_id=task.parent_id,
                depth=task.depth,
                label=task.label,
                sample_indices=task.sample_indices,
            ):
                if self._stop_training:
                    return
                yield updated_node
        return

    async def _predict(
        self,
        sample_index: Any,
        sample: str,
        token_counter: TokenCounter,
    ) -> AsyncGenerator[Tuple[Any, str, str, int], None]:
        """Predict a single sample data point."""
        node_id = self.get_root_id()
        node = self.get_node(node_id) if node_id is not None else None
        if node is None:
            raise ValueError("Tree is empty. Fit or load a tree before predicting.")

        while not node.is_leaf:
            question = node.question
            if question is None:
                raise ValueError(f"Node {node.id} has not question. Tree is corrupted.")
            idx_answer = await self._answer_question_for_row(
                idx=0,
                sample=sample,
                question=question,
                token_counter=token_counter,
            )
            if idx_answer is None:
                raise LLMError(f"Failed to answer question: {question.value}!")
            _, answer = idx_answer

            pre_node_id = node.id
            yield sample_index, question.value, answer.answer, pre_node_id
            node = next(
                (c for c in node.children or [] if c.label == answer.answer), None
            )
            if node is None:
                raise CorruptionError(
                    f"Node with label {answer.answer} not found in "
                    f"children of node {pre_node_id}"
                )
        yield sample_index, "No Question", "No Answer", node.id

    async def predict(
        self,
        samples: pd.DataFrame,
    ) -> AsyncGenerator[Tuple[int, str, str, int, TokenCounter], None]:
        """Predict labels for samples with concurrent processing.

        Args:
            samples: DataFrame with single column matching training data format.

        Yields:
            Tuple of (sample_index, question, answer, node_id, token_usage)
        """
        token_counter = TokenCounter()
        queue: asyncio.Queue[Literal["DONE"] | Tuple[Any, str, str, int]] = (
            asyncio.Queue()
        )

        async def worker(sample_index: Any, sample: str) -> None:
            try:
                async for record in self._predict(sample_index, sample, token_counter):
                    await queue.put(record)
            finally:
                await queue.put("DONE")

        tasks: List[asyncio.Task[None]] = []
        try:
            for sample_index, row in samples.iterrows():
                sample = "\n".join([f"{col}: {val}" for col, val in row.items()])
                tasks.append(asyncio.create_task(worker(sample_index, sample)))

            remaining = len(tasks)
            while remaining > 0:
                item = await queue.get()
                if item == "DONE":
                    remaining -= 1
                    continue
                else:
                    yield item + (token_counter,)
            return
        except asyncio.CancelledError:
            pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    def prune_tree(self, node_id: int) -> None:
        """Prune the tree from the node with the given ID.

        Args:
            node_id: The ID of the node to prune.

        Raises:
            ValueError: If the node with the given ID is not found on the tree.
            ValueError: If the node with the given ID is a leaf node.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found on tree.")
        if node.is_leaf:
            raise ValueError(f"Node with id {node_id} is a leaf node. Cannot prune.")

        to_remove: Set[int] = set()
        stack: List[Node] = list(node.children)
        while stack:
            n = stack.pop()
            to_remove.add(n.id)
            if n.children:
                stack.extend(n.children)

        for rid in to_remove:
            self._nodes.pop(rid, None)

        self._frontier = [
            t
            for t in self._frontier
            if t.node_id not in to_remove
            and (t.parent_id is None or t.parent_id not in to_remove)
        ]

        node.children = []
        node.question = None
        node.questions = []
        node.split_ratios = None

        self.save()

    def _get_path_to_node(self, node_id: int) -> List[Node]:
        """Return path from root to node (inclusive)."""
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found on tree.")
        path_nodes: List[Node] = []
        current: Node | None = node
        while current is not None:
            path_nodes.append(current)
            if current.parent_id is None:
                break
            current = self.get_node(current.parent_id)
        return list(reversed(path_nodes))

    def _compute_sample_indices_for_node(self, node_id: int) -> IndexArray:
        """Compute training sample indices that reach a given node.

        Walks from root to the node using parent questions' recorded
        df_column and each node's label.
        """
        if self._X is None or self._y is None:
            raise ValueError("Tree has no training data loaded. Fit or load first.")

        path_nodes = self._get_path_to_node(node_id)

        indices = np.asarray(self._X.index, dtype=np.intp)
        # Skip the first node (root). For each step, filter by parent's chosen split.
        for i in range(1, len(path_nodes)):
            parent = path_nodes[i - 1]
            child = path_nodes[i]
            if parent.question is None:
                raise ValueError(
                    (
                        f"Parent node {parent.id} has no question. "
                        "Cannot reconstruct samples."
                    )
                )
            df_column = parent.question.df_column
            sub_df = cast(pd.DataFrame, self._X.loc[indices])
            mask_series = cast(pd.Series, sub_df[df_column] == child.label)
            mask_np = np.asarray(mask_series, dtype=bool)
            next_indices = np.asarray(sub_df.index[mask_np], dtype=np.intp)
            indices = next_indices
        return indices

    async def resume_fit(self, node_id: int) -> AsyncGenerator[Node, None]:
        """Enqueue a node to resume (re)building its subtree from current data.

        Typical usage:
            - Call prune_tree(node_id) to clear the subtree
            - await resume_fit(node_id) to continue building from that node

        Args:
            node_id: The ID of the node to resume building from.

        Yields:
            Node: Updated nodes during tree construction.

        Raises:
            ValueError: If the node with the given ID is not found on the tree.
            ValueError: If the tree has no training data loaded.
        """
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found on tree.")
        if self._X is None or self._y is None:
            raise ValueError("Tree has no training data loaded. Fit or load first.")

        logger.debug(f"Getting path to node {node_id}")
        path_nodes = self._get_path_to_node(node_id)
        depth = len(path_nodes) - 1
        logger.debug(f"Depth: {depth}")
        logger.debug("Computing sample indices for node")
        sample_indices = self._compute_sample_indices_for_node(node_id)

        logger.debug("Updating frontier...")
        # Ensure no duplicate task for this node
        self._frontier = [t for t in self._frontier if t.node_id != node_id]
        self._frontier.append(
            BuildTask(
                node_id=node_id,
                parent_id=node.parent_id,
                depth=depth,
                label=node.label,
                sample_indices=sample_indices,
            )
        )
        self.save()
        logger.info("Re-fitting tree...")
        async for updated_node in self.fit():
            yield updated_node

    @classmethod
    def _load(cls, path: str | PathLike[str]) -> GPTree:
        base = Path(path)
        if base.is_dir():
            tree_json_path = base / "gptree.json"
            if not tree_json_path.exists():
                raise FileNotFoundError(f"'gptree.json' not found in directory: {base}")
        else:
            raise ValueError("Please provide a directory, not a file.")

        manifest = orjson.loads(tree_json_path.read_bytes())

        name: str = manifest["tree_name"]
        params = manifest["params"]
        llm_choices = manifest["llm_choices"]
        templates = manifest["templates"]

        inst = cls(
            qgen_llmc=llm_choices["qgen_llmc"],
            critic_llmc=llm_choices["critic_llmc"],
            qgen_instr_llmc=llm_choices["qgen_instr_llmc"],
            qanswer_llmc=llm_choices["qanswer_llmc"],
            qgen_temperature=params["qgen_temperature"],
            critic_temperature=params["critic_temperature"],
            qgen_instr_gen_temperature=params["qgen_instr_gen_temperature"],
            qanswer_temperature=params["qanswer_temperature"],
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            max_node_width=params["max_node_width"],
            min_samples_leaf=params["min_samples_leaf"],
            llm_semaphore_limit=params["llm_semaphore_limit"],
            min_question_candidates=params["min_question_candidates"],
            max_question_candidates=params["max_question_candidates"],
            expert_advice=manifest["expert_advice"],
            n_samples_as_context=params["n_samples_as_context"],
            class_ratio=params["class_ratio"],
            use_critic=params["use_critic"],
            save_path=base,
            name=name,
            random_state=params["random_state"],
        )

        inst._classes = manifest["classes"]
        inst._task_description = manifest["task_description"]
        inst._qgen_instructions_template = templates["qgen_instr_template"]
        inst._critic_instructions_template = templates["critic_instructions_template"]

        inst._llm_semaphore = asyncio.Semaphore(inst.llm_semaphore_limit)

        inst._node_counter = int(manifest["node_counter"])

        id_to_node: Dict[int, Node] = {}
        for nd in manifest["nodes"]:
            id_to_node[nd["id"]] = Node.from_dict(nd)

        # Link children by parent_id
        for node in id_to_node.values():
            if node.parent_id is not None:
                id_to_node[node.parent_id].children.append(node)

        inst._nodes = id_to_node

        # frontier
        inst._frontier = [
            BuildTask.from_dict(
                {
                    **ft,
                    "sample_indices": np.array(ft["sample_indices"], dtype=np.intp),
                }
            )
            for ft in manifest["frontier"]
        ]

        # token usage
        if tk_dict := manifest.get("token_counter"):
            inst._token_counter = TokenCounter.from_dict(tk_dict)

        # load training data if present
        data_parquet_path = base / "data.parquet"
        if data_parquet_path.exists():
            df = pd.read_parquet(data_parquet_path)  # type: ignore
            inst._y = df["y"].to_numpy(dtype=np.str_)  # type: ignore
            inst._X = df.drop(columns=["y"])  # type: ignore

        # Ensure save_path aligns with loaded data
        inst.save_path = base

        return inst

    @classmethod
    def load(cls, path: str | PathLike[str]) -> GPTree:
        """Load a GPTree from saved state.

        Args:
            path: Directory containing "gptree.json" or the JSON file directly.

        Returns:
            Reconstructed GPTree instance.
        """
        try:
            return cls._load(path)
        except KeyError as e:
            raise CorruptionError(
                f"Failed to load GPTree. Tree json is probably corrupted: {e}"
            )
        except Exception as e:
            raise e

    def save(
        self,
        dir_path: str | PathLike[str] | None = None,
        for_production: bool = False,
    ) -> None:
        """Save model config to JSON and dataframes to parquet in a directory.

        If dir_path is None, uses `<self.save_path>/<self.name>`.
        If for_production is True, does not save the training dataframe.

        Args:
            dir_path: The directory to save the tree to.
            for_production: Whether to save the tree for production.
        """
        base = Path(dir_path) if dir_path is not None else (self.save_path / self.name)
        if base.is_file():
            raise ValueError("Please provide a directory, not a file.")
        base.mkdir(parents=True, exist_ok=True)

        def _serialize_node(node: Node) -> Dict[str, Any]:
            dict_ = asdict(node)
            dict_.pop("children")
            return dict_

        payload: Dict[str, object] = {
            "tree_name": self.name,
            "created_at": datetime.datetime.now().isoformat(),
            "classes": list(self._classes) if self._classes is not None else None,
            "save_path": str(self.save_path) if not for_production else None,
            "params": {
                "criterion": self.criterion,
                "max_depth": self.max_depth,
                "max_node_width": self.max_node_width,
                "min_samples_leaf": self.min_samples_leaf,
                "llm_semaphore_limit": self.llm_semaphore_limit,
                "min_question_candidates": self.min_question_candidates,
                "max_question_candidates": self.max_question_candidates,
                "n_samples_as_context": self.n_samples_as_context,
                "class_ratio": self.class_ratio,
                "use_critic": self.use_critic,
                "qgen_temperature": self.qgen_temperature,
                "critic_temperature": self.critic_temperature,
                "qgen_instr_gen_temperature": self.qgen_instr_gen_temperature,
                "qanswer_temperature": self.qanswer_temperature,
                "random_state": self.random_state,
            },
            "llm_choices": {
                "qgen_llmc": [
                    llmc if isinstance(llmc, dict) else llmc.model_dump()
                    for llmc in self.qgen_llmc
                ],
                "critic_llmc": [
                    llmc if isinstance(llmc, dict) else llmc.model_dump()
                    for llmc in self.critic_llmc
                ],
                "qanswer_llmc": [
                    llmc if isinstance(llmc, dict) else llmc.model_dump()
                    for llmc in self.qanswer_llmc
                ],
                "qgen_instr_llmc": [
                    llmc if isinstance(llmc, dict) else llmc.model_dump()
                    for llmc in self.qgen_instr_llmc
                ],
            },
            "templates": {
                "qgen_instr_template": self._qgen_instructions_template,
                "critic_instructions_template": self._critic_instructions_template,
            },
            "expert_advice": self._expert_advice,
            "task_description": self._task_description,
            "node_counter": self._node_counter,
            "nodes": [_serialize_node(node) for node in self._nodes.values()],
            "frontier": [task.to_dict() for task in self._frontier],
            "token_counter": None if for_production else self._token_counter.to_dict(),
        }
        payload_json = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

        with (base / "gptree.json").open("w", encoding="utf-8") as f:
            f.write(payload_json.decode("utf-8"))

        if not for_production and self._X is not None and self._y is not None:
            df_to_save: pd.DataFrame = self._X.copy()
            df_to_save["y"] = self._y
            data_parquet_path = base / "data.parquet"
            df_to_save.to_parquet(str(data_parquet_path), index=True)  # type: ignore

    def __repr__(self) -> str:
        return f"GPTree(name={self.name})"

    def __str__(self) -> str:
        return f"GPTree(name={self.name})"
