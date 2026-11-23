"""Exceptions raised by Think Reason Learn.

Every one derives from ``TRLError``. Of the model-call errors (``LLMError``),
``NoOutputError`` is the one a model can recover from by skipping a step;
``ProviderAuthError``, ``ProviderDownError`` and ``BudgetExceededError`` stop the
run.
"""

__all__ = [
    "BudgetExceededError",
    "DataError",
    "LLMError",
    "MissingAPIKeyError",
    "NoOutputError",
    "ProviderAuthError",
    "ProviderDownError",
    "TRLError",
]


class TRLError(Exception):
    """Base of the library's exceptions."""


class DataError(TRLError):
    """Data not in the expected format."""


class LLMError(TRLError):
    """A model call failed."""


class NoOutputError(LLMError):
    """No model gave usable output: every one refused, failed or was unreadable.

    Args:
        message: What failed.
        reasons: Why, one per model tried, e.g. ``"openai:gpt-5-mini: no usable
            output (the reply was cut off at the output limit)"``.
    """

    def __init__(self, message: str, reasons: tuple[str, ...] = ()) -> None:
        """An error for ``message``, with each model's reason."""
        super().__init__(message)
        self.reasons = reasons


class MissingAPIKeyError(TRLError, ValueError):
    """A model was chosen whose API key is not set."""


class ProviderAuthError(LLMError):
    """A provider rejected the API key, or the account has no quota left."""


class ProviderDownError(LLMError):
    """The last model left kept failing, so the run stopped instead of waiting."""


class BudgetExceededError(LLMError):
    """A run would pass, or has passed, its cost cap (``max_cost_usd``)."""
