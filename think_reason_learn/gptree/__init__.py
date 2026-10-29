"""GPTree: a decision tree whose splits are questions written by a language model."""

from ._gptree import GPTree
from ._split import Criterion
from ._tree import Node, Tree

__all__ = ["Criterion", "GPTree", "Node", "Tree"]
