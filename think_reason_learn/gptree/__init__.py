"""A decision tree classifier employing LLMs for dynamic feature generation.

Each node uses language models to generate contextual questions and evaluate
answers, enabling adaptive tree construction for classification tasks.
"""

from ._gptree import GPTree, Node, NodeQuestion
from ._types import QuestionType, Criterion

__all__ = ["GPTree", "Node", "NodeQuestion", "QuestionType", "Criterion"]
