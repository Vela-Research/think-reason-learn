"""Exceptions raised by Think Reason Learn."""


class DataError(Exception):
    """Data not in the expected format."""


class LLMError(Exception):
    """LLM failed to respond."""


class MissingAPIKeyError(ValueError):
    """A model was chosen whose API key is not set."""


class ProviderAuthError(LLMError):
    """A provider rejected the API key, or the account has no quota left."""


class BudgetExceededError(LLMError):
    """A run would pass, or has passed, its cost cap (``max_cost_usd``)."""
