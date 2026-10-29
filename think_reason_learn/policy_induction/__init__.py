"""Policy Induction: a policy of rules, refined batch by batch, weighed per rule."""

from ._policy_induction import PolicyInduction, Predictor

__all__ = ["PolicyInduction", "Predictor"]
