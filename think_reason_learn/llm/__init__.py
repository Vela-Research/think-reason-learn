"""One interface to every model, reasoning or decision.

Reasoning models write text and answer questions; decision models answer.

Using models: a model is a string, ``"provider:model"``, or an ``LLM`` with
settings. ``config_context``, ``set_config`` and ``set_limits`` set the cache,
cost cap, retries and speed limits; a fitted model's ``usage_`` is a ``Usage``.

Writing a model (a ``TRLClassifier`` subclass): every call goes through the
``LLMSession`` it is given, which handles fallbacks, retries, speed limits, the
reply cache, budgets and usage the same way for every provider. Questions are
``YesNoQuestion`` or ``ChoiceQuestion``; ``estimate_answers`` and
``estimate_generations`` plan the calls that ``estimate_cost`` reports. The
built-in models use only these names.
"""

from ._adapters import Answers
from ._config import (
    Config,
    ConfigChanges,
    config_context,
    get_config,
    set_config,
    set_limits,
)
from ._estimate import (
    GENERATE_OUTPUT_TOKENS,
    QUESTION_TOKENS,
    estimate_answers,
    estimate_generations,
    mean_tokens,
    text_tokens,
)
from ._loop import run_async, run_sync
from ._questions import NO, YES, YES_AT, Answer, ChoiceQuestion, Question, YesNoQuestion
from ._session import LLMSession
from ._spec import (
    LLM,
    Endpoint,
    Model,
    ModelKind,
    ModelName,
    Models,
    ReasoningEffort,
    resolve_models,
)
from ._usage import Price, Usage, UsageRecord

__all__ = [
    # Using models
    "LLM",
    "Config",
    "ConfigChanges",
    "Model",
    "ModelKind",
    "ModelName",
    "Models",
    "Price",
    "ReasoningEffort",
    "Usage",
    "UsageRecord",
    "config_context",
    "get_config",
    "set_config",
    "set_limits",
    # Writing a model
    "GENERATE_OUTPUT_TOKENS",
    "NO",
    "QUESTION_TOKENS",
    "YES",
    "YES_AT",
    "Answer",
    "Answers",
    "ChoiceQuestion",
    "Endpoint",
    "LLMSession",
    "Question",
    "YesNoQuestion",
    "estimate_answers",
    "estimate_generations",
    "mean_tokens",
    "resolve_models",
    "run_async",
    "run_sync",
    "text_tokens",
]
