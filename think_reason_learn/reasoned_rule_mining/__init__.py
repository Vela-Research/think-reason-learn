"""Reasoned Rule Mining.

An interpretable binary classifier that mines natural-language IF-THEN rules
from LLM reasoning over labelled data, compiles them into a decision policy, and
predicts via a calibrated, confidence-weighted ensemble vote with an optional
harsh re-evaluation stream.
"""

from ._reasoned_rule_mining import ReasonedRuleMining
from ._types import ExtractedRule, RRMConfig, Vote

__all__ = ["ReasonedRuleMining", "RRMConfig", "ExtractedRule", "Vote"]
