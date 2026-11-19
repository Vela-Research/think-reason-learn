"""Policy decisions: 2 to 4 ordered outcomes, from a written policy, labels or both.

``PolicyClassifier`` builds yes/no checks from a policy's criteria, has a cheap
answerer answer them, and weighs the answers with an ordinal model, so every
prediction comes with points by criterion. The cases it learns from are labelled
by you, by a teacher that applies the policy, or both. ``label_cases``,
``write_checks`` and ``answer_checks`` are its steps, usable on their own.
"""

from ._answers import aanswer_checks, answer_checks
from ._checks import Form, Signs
from ._classifier import Decision, PolicyClassifier
from ._labels import alabel_cases, label_cases
from ._types import (
    Criterion,
    Explanation,
    Kind,
    Mistake,
    Outcome,
    Policy,
    Teacher,
    TeacherVerdict,
)
from ._writer import awrite_checks, write_checks

__all__ = [
    "Criterion",
    "Decision",
    "Explanation",
    "Form",
    "Kind",
    "Mistake",
    "Outcome",
    "Policy",
    "PolicyClassifier",
    "Signs",
    "Teacher",
    "TeacherVerdict",
    "aanswer_checks",
    "alabel_cases",
    "answer_checks",
    "awrite_checks",
    "label_cases",
    "write_checks",
]
