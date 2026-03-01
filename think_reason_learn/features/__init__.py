"""Lambda-based feature generation.

Generate binary classification rules as executable Python lambda expressions
via an LLM, then evaluate them deterministically on structured data to produce
a binary feature matrix — at zero marginal cost per evaluation.
"""

from ._evaluator import FeatureEvaluator
from ._generator import FeatureGenerator
from ._types import DataSchema, GeneratedRule, GeneratedRules, HelperFunction, Rule

__all__ = [
    "DataSchema",
    "FeatureEvaluator",
    "FeatureGenerator",
    "GeneratedRule",
    "GeneratedRules",
    "HelperFunction",
    "Rule",
]
