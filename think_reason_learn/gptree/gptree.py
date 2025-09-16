from __future__ import annotations

import asyncio
from typing import List, Dict, Literal, cast, Type, Tuple, AsyncGenerator
import logging
import contextlib
from dataclasses import dataclass, field, asdict
from uuid import uuid4
from copy import deepcopy
from pathlib import Path
import datetime
import re

import orjson
from pydantic import BaseModel, create_model
import numpy as np
import numpy.typing as npt
import pandas as pd

from ._types import QuestionType, Criterion
from ._prompts import INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS
from ._prompts import num_questions_tag, QUESTION_ANSWER_INSTRUCTIONS
from think_reason_learn.core.llms.schemas import LLMChoice, TokenCount
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


# New: task descriptor for resumable building
@dataclass(slots=True)
class BuildTask:
    node_id: int
    parent_id: int | None
    depth: int
    label: str
    sample_indices: IndexArray


class GPTree:
    """LLM based decision tree classifier."""

    def __init__(
        self,
        question_gen_llmc: List[LLMChoice],
        critic_llmc: List[LLMChoice],
        question_gen_instructions_llmc: List[LLMChoice],
        question_gen_temperature: float = 0.0,
        critic_temperature: float = 0.0,
        question_gen_instructions_gen_temperature: float = 0.0,
        criterion: Criterion = "gini",
        max_depth: int | None = None,
        max_node_width: int = 3,
        min_samples_leaf: int = 1,
        max_clusters: int | None = None,
        llm_semaphore_limit: int = 3,
        min_question_candidates: int = 3,
        max_question_candidates: int = 10,
        expert_advice: str | None = None,
        context_samples_n: int = 30,
        class_ratio: Dict[str, int] | Literal["balanced"] = "balanced",
        use_critic: bool = False,
        save_path: str | Path | None = None,
        save_training_data: bool = False,
        name: str | None = None,
    ):
        """
        GPTree.

        Attributes
        ----------
        question_gen_llmc: List[LLMChoice]
            The LLM choice for question generation.

        critic_llmc: List[LLMChoice]
            The LLM choice for critic LLM.

        question_gen_instructinos_llmc: List[LLMChoice]
            The LLM choice for generating question generation instructions.

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

        max_node_width: int, default=3
            The maximum number of immediate children of a node.

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

        save_path: str | Path | None = None,
            The path to save the tree.

        save_training_data: bool, default=False
            Whether to save the training data.
        """
        self.question_gen_llmc = question_gen_llmc
        self.question_gen_instructions_llmc = question_gen_instructions_llmc
        self.critic_llmc = critic_llmc
        self.question_gen_temperature = question_gen_temperature
        self.critic_temperature = critic_temperature
        self.question_gen_instructions_gen_temperature = (
            question_gen_instructions_gen_temperature
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_node_width = max_node_width
        self.min_samples_leaf = min_samples_leaf
        self.max_clusters = max_clusters
        self.llm_semaphore_limit = llm_semaphore_limit
        self.token_usage: List[TokenCount] = []
        self.min_question_candidates = min_question_candidates
        self.max_question_candidates = max_question_candidates
        self._expert_advice = expert_advice
        self.context_samples_n = context_samples_n
        self.class_ratio = class_ratio
        self.use_critic = use_critic
        self.name: str = self._get_name(name)
        self.save_path: Path = self._set_save_path(save_path)
        self.save_training_data = save_training_data

        self._classes: List[str] | None = None
        self._X: pd.DataFrame | None = None
        self._y: npt.NDArray[np.str_] | None = None
        self._nodes: Dict[int, Node] = {}
        self._root: Node | None = None
        self._node_counter = 0
        self._llm_semaphore = asyncio.Semaphore(llm_semaphore_limit)
        self._question_gen_instructions_template: str | None = None
        self._critic_instructions_template: str | None = None
        self._X_column = "data"
        self._stop_training: bool = False
        self._task_description: str | None = None

        self._frontier: List[BuildTask] = []  # Frontier for resumable training

    def view_node(self, node_id: int, format: Literal["png", "svg"] = "png") -> bytes:
        """Render the subtree rooted at node_id as a PNG/SVG image and return bytes.

        Parameters
        ----------
        node_id: int
            The id of the node to visualize as the root of the rendered subtree.
        format: Literal["png", "svg"], default="png"
            Output image format.
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found")

        try:
            from graphviz import Digraph  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'graphviz' Python package is required. Install it with `pip install graphviz`."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Graphviz failed to load. Ensure both the Python package (`pip install graphviz`) "
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

        def _truncate(text: str, max_len: int = 140) -> str:
            if len(text) <= max_len:
                return text
            return text[: max_len - 3] + "..."

        dot = Digraph(name=f"GPTree_{self.name}_{node_id}", format=format, graph_attr={"rankdir": "TB"})  # type: ignore

        visited: set[int] = set()
        queue: List[Node] = []
        root = self._nodes[node_id]
        queue.append(root)

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
            if current.question is not None:
                q = current.question
                label_lines.append(_truncate(f"Q: {q.value}"))
                label_lines.append(f"choices: {', '.join(q.choices)}")
            if current.split_ratios is not None:
                label_lines.append(
                    f"split: {', '.join(str(x) for x in current.split_ratios)}"
                )
            if current.class_distribution:
                dist_str = ", ".join(
                    f"{k}:{v}" for k, v in current.class_distribution.items()
                )
                label_lines.append(_truncate(f"dist: {dist_str}", 160))

            node_label = _escape("\n".join(label_lines))
            dot.node(str(current.id), node_label, shape="box", style="rounded,filled", fillcolor="lightgrey", fontsize="10")  # type: ignore

            for child in current.children or []:
                edge_label = _escape(str(child.label))
                dot.edge(str(current.id), str(child.id), label=edge_label)  # type: ignore
                if child.id not in visited:
                    queue.append(child)

        return dot.pipe(format=format)  # type: ignore

    def stop(self) -> None:
        """Stop the training of the tree."""
        self._stop_training = True

    def advice(self, advice: str) -> None:
        """Set context/advice for question generations."""
        if not isinstance(advice, str):  # type: ignore
            raise ValueError("Expert advice must be a string")
        self._expert_advice = advice

    @property
    def expert_advice(self) -> str | None:
        """Get the expert advice."""
        return self._expert_advice

    def _get_name(self, name: str | None) -> str:
        if name is None:
            name = str(uuid4()).replace("-", "_")
            logger.info(f"No name provided. Assigned name: {name}")

        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError("Name must be only alphanumeric and underscores")
        return name

    def _set_save_path(self, save_path: str | Path | None) -> Path:
        """Set the save path of the tree."""

        lib_dir = Path(__file__).resolve().parent.parent

        if save_path is None:
            return (lib_dir / "gptrees").resolve()
        else:
            save_path = Path(save_path).resolve()
            if lib_dir in save_path.parents:
                raise ValueError("Save path must be outside of the library directory")
            return save_path

    @classmethod
    def load(cls, path: str | Path) -> "GPTree":
        """Load a GPTree from a saved directory or JSON file.

        Parameters
        ----------
        path: str | Path
            Either the directory containing a "*_tree.json" file or the JSON file itself.

        Returns
        -------
        GPTree
            A reconstructed GPTree instance with nodes, frontier, and optional data.
        """
        path = Path(path).resolve()

        if path.is_dir():
            candidates = sorted(path.glob("*_tree.json"))
            if not candidates:
                raise FileNotFoundError(f"No '*_tree.json' found in directory: {path}")
            tree_json_path = candidates[0]
            save_dir = path
        else:
            tree_json_path = path
            save_dir = path.parent

        payload = orjson.loads(tree_json_path.read_bytes())

        name: str = payload.get("tree_name")
        params = payload.get("params", {})
        llm_priorities = payload.get("llm_priorities", {})
        templates = payload.get("templates", {})

        instance = cls(
            question_gen_llmc=llm_priorities.get("question_gen_llmc", []),
            critic_llmc=llm_priorities.get("critic_llmc", []),
            question_gen_instructions_llmc=llm_priorities.get(
                "question_gen_instructions_llmc", []
            ),
            question_gen_temperature=params.get("question_gen_temperature", 0.0),
            critic_temperature=params.get("critic_temperature", 0.0),
            question_gen_instructions_gen_temperature=params.get(
                "question_gen_instructions_gen_temperature", 0.0
            ),
            criterion=params.get("criterion", "gini"),
            max_depth=params.get("max_depth"),
            max_node_width=params.get("max_node_width", 3),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_clusters=params.get("max_clusters"),
            llm_semaphore_limit=params.get("llm_semaphore_limit", 3),
            min_question_candidates=params.get("min_question_candidates", 3),
            max_question_candidates=params.get("max_question_candidates", 10),
            expert_advice=payload.get("expert_advice"),
            context_samples_n=params.get("context_samples_n", 30),
            class_ratio=params.get("class_ratio", "balanced"),
            use_critic=params.get("use_critic", False),
            save_path=save_dir,
            save_training_data=payload.get("save_training_data", False),
            name=name,
        )

        instance._X_column = payload.get("x_column", instance._X_column)
        instance._classes = payload.get("classes")
        instance._task_description = payload.get("task_description")
        instance._question_gen_instructions_template = templates.get(
            "question_gen_instructions_template"
        )
        instance._critic_instructions_template = templates.get(
            "critic_instructions_template"
        )
        instance._node_counter = int(payload.get("node_counter", 0))

        def _to_node_question(qd: Dict[str, object] | None) -> NodeQuestion | None:
            if qd is None:
                return None
            return NodeQuestion(
                value=cast(str, qd.get("value")),
                choices=list(cast(List[str], qd.get("choices", []))),
                question_type=cast(QuestionType, qd.get("question_type")),
                df_column=cast(str, qd.get("df_column")),
                score=cast(float | None, qd.get("score")),
            )

        nodes_payload = payload.get("nodes", [])
        id_to_node: Dict[int, Node] = {}

        for nd in nodes_payload:
            node_id = int(nd.get("id"))
            parent_id_val = nd.get("parent_id")
            parent_id = int(parent_id_val) if parent_id_val is not None else None
            split_ratios_list = nd.get("split_ratios")
            split_ratios = (
                tuple(int(x) for x in split_ratios_list) if split_ratios_list else None
            )
            question = _to_node_question(nd.get("question"))
            questions_list = [
                q
                for q in [_to_node_question(qd) for qd in nd.get("questions", [])]
                if q is not None
            ]

            node = Node(
                id=node_id,
                parent_id=parent_id,
                label=cast(str, nd.get("label")),
                gini=float(nd.get("gini", 0.0)),
                class_distribution=cast(
                    Dict[str, int], nd.get("class_distribution", {})
                ),
                split_ratios=split_ratios,
                question=question,
                questions=questions_list,
                memory_context=cast(str | None, nd.get("memory_context")),
                children=[],
            )
            id_to_node[node_id] = node

        # Link children by parent_id
        for node in id_to_node.values():
            if node.parent_id is not None and node.parent_id in id_to_node:
                parent = id_to_node[node.parent_id]
                parent.children = parent.children or []
                parent.children.append(node)

        instance._nodes = id_to_node
        root_id = payload.get("root_id")
        instance._root = id_to_node.get(int(root_id)) if root_id is not None else None

        # frontier
        frontier_payload = payload.get("frontier", [])
        instance._frontier = []
        for ft in frontier_payload:
            try:
                frontier_task = BuildTask(
                    node_id=int(ft.get("node_id")),
                    parent_id=(
                        int(ft.get("parent_id"))
                        if ft.get("parent_id") is not None
                        else None
                    ),
                    depth=int(ft.get("depth", 0)),
                    label=cast(str, ft.get("label")),
                    sample_indices=np.array(
                        ft.get("sample_indices", []), dtype=np.intp
                    ),
                )
            except Exception as e:
                raise DataError(f"Invalid frontier entry insaved data: {ft}") from e
            instance._frontier.append(frontier_task)

        # token usage
        instance.token_usage = []
        for tu in payload.get("token_usage", []):
            try:
                token_count = TokenCount(
                    provider=tu.get("provider"),  # type: ignore
                    model=tu.get("model"),  # type: ignore
                    value=tu.get("value"),
                )
            except Exception as e:
                raise DataError(f"Invalid token usage entry in saved data: {tu}") from e
            instance.token_usage.append(token_count)

        # load training data if present
        data_csv_default = save_dir / f"{name}_data.csv"
        fallback_csv = save_dir / "data.csv"
        data_csv_path = data_csv_default if data_csv_default.exists() else fallback_csv
        if data_csv_path.exists():
            df = pd.read_csv(data_csv_path)  # type: ignore
            if "y" in df.columns:
                instance._y = df["y"].to_numpy(dtype=np.str_)  # type: ignore
                instance._X = df.drop(columns=["y"])  # type: ignore
            else:
                instance._X = df  # type: ignore
                instance._y = None

        # Ensure save_path and name align with loaded data
        instance.save_path = save_dir
        instance.name = name

        return instance

    def _save(self):
        """Save the tree state to JSON (and optionally training data to CSV)"""
        self.save_path.mkdir(parents=True, exist_ok=True)

        tree_json_path = self.save_path / f"{self.name}_tree.json"

        nodes_serialized = [asdict(node) for node in self._nodes.values()]
        frontier_serialized = [asdict(t) for t in self._frontier]
        token_usage_serialized = [asdict(tc) for tc in self.token_usage]

        def _serialize_llm_priorities(prios: List[LLMChoice]) -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            for p in prios:
                if isinstance(p, BaseModel):
                    out.append(p.model_dump())
                else:
                    out.append(p)  # type: ignore
            return out

        payload: Dict[str, object] = {
            "tree_name": self.name,
            "created_at": datetime.datetime.now().isoformat(),
            "root_id": self._root.id if self._root is not None else None,
            "classes": list(self._classes) if self._classes is not None else None,
            "x_column": self._X_column,
            "save_training_data": self.save_training_data,
            "save_path": str(self.save_path),
            "params": {
                "criterion": self.criterion,
                "max_depth": self.max_depth,
                "max_node_width": self.max_node_width,
                "min_samples_leaf": self.min_samples_leaf,
                "max_clusters": self.max_clusters,
                "llm_semaphore_limit": self.llm_semaphore_limit,
                "min_question_candidates": self.min_question_candidates,
                "max_question_candidates": self.max_question_candidates,
                "context_samples_n": self.context_samples_n,
                "class_ratio": self.class_ratio,
                "use_critic": self.use_critic,
                "question_gen_temperature": self.question_gen_temperature,
                "critic_temperature": self.critic_temperature,
                "question_gen_instructions_gen_temperature": self.question_gen_instructions_gen_temperature,
            },
            "llm_priorities": {
                "question_gen_llmc": _serialize_llm_priorities(self.question_gen_llmc),
                "critic_llmc": _serialize_llm_priorities(self.critic_llmc),
                "question_gen_instructions_llmc": _serialize_llm_priorities(
                    self.question_gen_instructions_llmc
                ),
            },
            "templates": {
                "question_gen_instructions_template": self._question_gen_instructions_template,
                "critic_instructions_template": self._critic_instructions_template,
            },
            "expert_advice": self._expert_advice,
            "task_description": self._task_description,
            "node_counter": self._node_counter,
            "nodes": nodes_serialized,
            "frontier": frontier_serialized,
            "token_usage": token_usage_serialized,
        }
        payload_json = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

        with tree_json_path.open("w", encoding="utf-8") as f:
            f.write(payload_json.decode("utf-8"))

        if self.save_training_data and self._X is not None and self._y is not None:
            df_to_save: pd.DataFrame = self._X.copy()
            df_to_save["y"] = self._y
            data_csv_path = self.save_path / f"{self.name}_data.csv"
            df_to_save.to_csv(str(data_csv_path), index=True)  # type: ignore

    @property
    def classes(self) -> List[str] | None:
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
                    f"Instructions_template must contain the tag '{num_questions_tag}' "
                    "This tag will be replaced with the number of questions to generate at each generation run"
                )
            else:
                self._question_gen_instructions_template = instructions_template
                return

        async with self._llm_semaphore:
            response = await llm.respond(
                query=f"The task is to build a decision tree model for this: {task_description}",
                llm_priority=self.question_gen_instructions_llmc,
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

        self._task_description = task_description
        self._question_gen_instructions_template = response.response

    def _get_question_gen_instructions(self, num_questions: int) -> str:

        if not self._question_gen_instructions_template:
            raise ValueError(
                "Question generation instructions template is not set"
                "Set the template using `init_question_gen_instructions_template`"
            )

        instructions = self._question_gen_instructions_template.replace(
            num_questions_tag, str(num_questions)
        )
        instructions += f"\n\nIMPORTANT: Limit the number of choices per question to {self.max_node_width} and more than 1."
        return instructions

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

        if self._expert_advice is not None:
            query += f"Consider this expert advice: {self._expert_advice}\n"
        if context_memory is not None:
            query += f"Context from previous nodes: {context_memory}"

        async with self._llm_semaphore:
            response = await llm.respond(
                query=query,
                llm_priority=self.question_gen_llmc,
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
                llm_priority=self.question_gen_llmc,
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
    ) -> AsyncGenerator[Node, None]:
        X = self._X
        y = self._y

        if self._stop_training:
            self._frontier.append(
                BuildTask(
                    node_id=id,
                    parent_id=parent_id,
                    depth=depth,
                    label=label,
                    sample_indices=sample_indices,
                )
            )
            return

        if X is None or y is None:
            raise ValueError("X, y,  must be set")

        sample_y = y[sample_indices]

        uniq, counts = np.unique(sample_y, return_counts=True)
        class_distribution: Dict[str, int] = {
            str(u): int(c) for u, c in zip(uniq, counts)
        }

        if self._classes is None:
            self._classes = list(uniq)

        # NOTE: Could just calculate, but we want to be consistent
        gini = self._gini(sample_indices)

        if (
            depth >= (self.max_depth or float("inf"))
            or sample_y.shape[0] < len(self._classes) * self.min_samples_leaf
            or uniq.shape[0] == 1
        ):
            logger.info(f"Terminal node at depth {depth} with id {id}")
            node = Node(
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
            self._nodes[id] = node
            if id == 0:
                self._root = node
            yield node
            return

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
            node = Node(
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
            self._nodes[id] = node
            if id == 0:
                self._root = node
            yield node
            return

        choice_dfs = {
            choice: cast(pd.DataFrame, X[X[chosen_question.df_column] == choice])
            for choice in chosen_question.choices
        }
        split_ratios = tuple([df.shape[0] for df in choice_dfs.values()])

        node = Node(
            id=id,
            parent_id=parent_id,
            label=label,
            gini=gini,
            question=chosen_question,
            questions=[NodeQuestion(**q.model_dump()) for q in questions.questions],
            memory_context=questions.memory_context,
            split_ratios=split_ratios,
            class_distribution=class_distribution,
            children=[],
        )
        self._nodes[id] = node
        if id == 0:
            self._root = node

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

        for idx, (choice, indices) in enumerate(choice_indices_list):

            if self._stop_training:
                for c, inds in choice_indices_list[idx:]:
                    self._frontier.append(
                        BuildTask(
                            node_id=new_id_map[c],
                            parent_id=id,
                            depth=depth + 1,
                            label=c,
                            sample_indices=inds,
                        )
                    )
                yield node
                return

            async for child_node in self._build_tree(
                id=new_id_map[choice],
                parent_id=id,
                depth=depth + 1,
                label=choice,
                sample_indices=indices,
                verbose=verbose,
            ):
                node.children = node.children or []
                node.children.append(child_node)
                yield node

        return

    def _set_data(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[np.str_],
        copy_data: bool,
    ) -> None:
        if X.columns.tolist() != [self._X_column]:
            raise DataError(
                f"X must be a pandas DataFrame with a single column named {self._X_column}"
            )
        if not isinstance(y, np.ndarray) or y.dtype != np.str_ or y.ndim != 1:  # type: ignore
            raise DataError(f"y must be a numpy array of strings with one dimension")

        if y.shape[0] != X.shape[0]:
            raise DataError(f"y and X must have the same number of rows")

        if copy_data:
            self._X = deepcopy(X)  # type: ignore
            self._y = deepcopy(y)
        else:
            self._X = X  # type: ignore
            self._y = y

        self._nodes = {}
        self._root = None
        self._node_counter = 0
        self._frontier = []
        self._stop_training = False

    async def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[np.str_],
        copy_data: bool = True,
        verbose: bool = True,
    ) -> AsyncGenerator[Node, None]:
        """
        Train the tree. Run as an async generator. Can be stopped by calling `stop`.

        Parameters
        ----------
        X: pd.DataFrame
            The data to train the tree on.

        y: npt.NDArray[np.str_]
            The target labels to train the tree on.

        copy_data: bool, default=True
            Whether to copy the data.

        verbose: bool, default=True
            Whether to print verbose output.

        Returns
        -------
        AsyncGenerator[Node, None]
            The root node of the tree.
        """
        self._set_data(X, y, copy_data)

        async for updated_root in self._build_tree(
            id=0,
            parent_id=None,
            depth=0,
            label="root",
            sample_indices=X.index.to_numpy(dtype=np.intp),  # type: ignore
            verbose=verbose,
        ):
            if self._stop_training:
                break
            yield updated_root

    async def continue_fit(self, verbose: bool = True) -> AsyncGenerator[Node, None]:
        """
        Continue training the tree. Run as an async generator. Can be stopped by calling `stop`.

        Parameters
        ----------
        verbose: bool, default=True
            Whether to print verbose output.

        Returns
        -------
        AsyncGenerator[Node, None]
            The root node of the tree.
        """
        if self._X is None or self._y is None:
            raise ValueError(
                "No data found on tree. Please fit the tree first. "
                "If tree was loaded, then `save_training_data` must have been set to False hence no data was saved."
            )

        self._stop_training = False

        if len(self._nodes) == 0:  # Just starting. Kick off from root.
            indices: IndexArray = self._X.index.to_numpy(dtype=np.intp)  # type: ignore
            async for updated_root in self._build_tree(
                id=0,
                parent_id=None,
                depth=0,
                label="root",
                sample_indices=indices,
                verbose=verbose,
            ):
                if self._stop_training:
                    break
                yield updated_root
            return

        while self._frontier:  # Resume from queued frontier
            task = self._frontier.pop(0)
            async for updated_node in self._build_tree(
                id=task.node_id,
                parent_id=task.parent_id,
                depth=task.depth,
                label=task.label,
                sample_indices=task.sample_indices,
                verbose=verbose,
            ):
                if self._stop_training:
                    return
                yield updated_node
