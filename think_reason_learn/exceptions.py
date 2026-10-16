"""Exceptions raised by Think Reason Learn.

Every one derives from ``TRLError``. Of the model-call errors (``LLMError``),
``NoOutputError`` is the one a model can recover from by skipping a step;
``ProviderAuthError`` and ``BudgetExceededError`` stop the run.
"""

__all__ = [
    "BudgetExceededError",
    "DataError",
    "LLMError",
    "MissingAPIKeyError",
    "NoOutputError",
    "ProviderAuthError",
    "TRLError",
]


class TRLError(Exception):
    """Base of the library's exceptions."""


class DataError(TRLError):
    """Data not in the expected format."""


class LLMError(TRLError):
    """A model call failed."""


class NoOutputError(LLMError):
    """No model gave usable output: every one refused, failed or was unreadable."""


class MissingAPIKeyError(TRLError, ValueError):
    """A model was chosen whose API key is not set."""


class ProviderAuthError(LLMError):
    """A provider rejected the API key, or the account has no quota left."""


class BudgetExceededError(LLMError):
    """A run would pass, or has passed, its cost cap (``max_cost_usd``)."""
