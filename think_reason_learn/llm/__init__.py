"""One interface to every model, reasoning or decision.

Reasoning models write text and answer questions; decision models answer.

A model is a string, ``"provider:model"``, or an ``LLM`` with settings. Every
call goes through an ``LLMSession``, which handles fallbacks, retries, speed
limits, the reply cache, budgets and usage the same way for every provider.
"""

from ._adapters import Answers
from ._config import Config, Limits, config_context, get_config, set_config, set_limits
from ._loop import run_async, run_sync
from ._questions import NO, YES, Answer, ChoiceQuestion, Question, YesNoQuestion
from ._session import LLMSession
from ._spec import (
    LLM,
    Endpoint,
    Model,
    ModelKind,
    ModelName,
    Models,
    ReasoningEffort,
    resolve,
    resolve_models,
)
from ._usage import Budget, Price, TokenCounts, Usage, UsageRecord, format_usd

__all__ = [
    "LLM",
    "NO",
    "YES",
    "Answer",
    "Answers",
    "Budget",
    "ChoiceQuestion",
    "Config",
    "Endpoint",
    "LLMSession",
    "Limits",
    "Model",
    "ModelKind",
    "ModelName",
    "Models",
    "Price",
    "Question",
    "ReasoningEffort",
    "TokenCounts",
    "Usage",
    "UsageRecord",
    "YesNoQuestion",
    "config_context",
    "format_usd",
    "get_config",
    "resolve",
    "resolve_models",
    "run_async",
    "run_sync",
    "set_config",
    "set_limits",
]
