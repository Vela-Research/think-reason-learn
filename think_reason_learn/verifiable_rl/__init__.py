"""Verifiable RL.

An adaptive information-gathering binary classifier: a learned policy network
decides, slot by slot, which information to reveal next (or to STOP), guided by
Monte-Carlo tree-search targets and optional weak LLM supervision; a classifier
then predicts the label from the accumulated partial state.
"""

from ._supervisor import ActionSupervisor, LLMActionSupervisor
from ._types import NextActionPreference, QueryResult, VerifiableRLConfig
from ._verifiable_rl import VerifiableRL

__all__ = [
    "VerifiableRL",
    "VerifiableRLConfig",
    "QueryResult",
    "NextActionPreference",
    "LLMActionSupervisor",
    "ActionSupervisor",
]
