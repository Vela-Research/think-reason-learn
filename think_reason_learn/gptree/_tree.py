"""The fitted tree: nodes with their questions, branches and class counts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from think_reason_learn.llm import Question


@dataclass(slots=True)
class Node:
    """One node. A leaf has no question; an inner node routes by its answer.

    Attributes:
        id: Index of the node in ``Tree.nodes``.
        depth: Root is 0.
        value: Weighted class counts of the training samples that reached it.
        n_samples: Training samples that reached it.
        question: The question that splits it; ``None`` for a leaf.
        children: Child id per answer label.
        default: The answer followed by a sample left unanswered.
        parent: Parent id; ``None`` for the root.
        branch: The parent's answer that leads here.
        candidates: Questions considered here.
    """

    id: int
    depth: int
    value: tuple[float, ...]
    n_samples: int
    question: Question | None = None
    children: dict[str, int] = field(default_factory=dict)
    default: str | None = None
    parent: int | None = None
    branch: str | None = None
    candidates: tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool:
        """Whether the node has no question."""
        return self.question is None


@dataclass(slots=True)
class Tree:
    """The nodes of a fitted GPTree; the root is ``nodes[0]``."""

    nodes: list[Node] = field(default_factory=list)

    def add(
        self,
        *,
        depth: int,
        value: tuple[float, ...],
        n_samples: int,
        parent: int | None = None,
        branch: str | None = None,
    ) -> Node:
        """Append a leaf with the next id."""
        node = Node(
            len(self.nodes), depth, value, n_samples, parent=parent, branch=branch
        )
        self.nodes.append(node)
        return node

    def proba(self, node_id: int) -> npt.NDArray[np.float64]:
        """Class probabilities at a node: its training class shares."""
        value = np.asarray(self.nodes[node_id].value, dtype=float)
        return (
            value / value.sum()
            if value.sum() > 0
            else np.full(len(value), 1 / len(value))
        )

    def path(self, node_id: int) -> list[tuple[str, str]]:
        """(question, answer) from the root down to ``node_id``."""
        steps: list[tuple[str, str]] = []
        node = self.nodes[node_id]
        while node.parent is not None:
            parent = self.nodes[node.parent]
            if parent.question is not None and node.branch is not None:
                steps.append((parent.question.text, node.branch))
            node = parent
        return steps[::-1]

    def prune(self, node_id: int) -> None:
        """Make ``node_id`` a leaf, drop the nodes below it and renumber the rest."""
        node = self.nodes[node_id]
        node.question, node.default, node.children = None, None, {}
        reachable, queue = [], [0]
        while queue:
            current = self.nodes[queue.pop(0)]
            reachable.append(current)
            queue.extend(current.children.values())
        reachable.sort(key=lambda n: n.id)  # keep surviving nodes in their order
        new_id = {n.id: i for i, n in enumerate(reachable)}
        for n in reachable:
            n.id = new_id[n.id]
            n.parent = new_id[n.parent] if n.parent is not None else None
            n.children = {answer: new_id[child] for answer, child in n.children.items()}
        self.nodes = reachable

    @property
    def n_leaves(self) -> int:
        """Number of leaves."""
        return sum(node.is_leaf for node in self.nodes)

    def to_frame(self, classes: Sequence[object]) -> pd.DataFrame:
        """One row per node: its question, branches and class counts."""
        rows = [
            {
                "node": n.id,
                "depth": n.depth,
                "parent": n.parent,
                "branch": n.branch,
                "question": n.question.text if n.question else None,
                "children": dict(n.children) or None,
                "n_samples": n.n_samples,
                **{str(c): v for c, v in zip(classes, n.value)},
            }
            for n in self.nodes
        ]
        return pd.DataFrame(rows).set_index("node")

    def export_graphviz(self, classes: Sequence[object]) -> str:
        """The tree in Graphviz DOT, for ``graphviz.Source`` or ``dot``."""
        lines = ["digraph GPTree {", '  node [shape=box, fontname="Helvetica"];']
        for n in self.nodes:
            shares = ", ".join(
                f"{c}: {p:.0%}" for c, p in zip(classes, self.proba(n.id))
            )
            head = n.question.text if n.question else "leaf"
            label = f"{head}\\nsamples: {n.n_samples}\\n{shares}".replace('"', "'")
            lines.append(f'  {n.id} [label="{label}"];')
            lines.extend(
                f'  {n.id} -> {child} [label="{answer}"];'
                for answer, child in n.children.items()
            )
        return "\n".join([*lines, "}"])
