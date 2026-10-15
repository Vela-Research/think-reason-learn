"""Exceptions raised by Think Reason Learn."""


class DataError(Exception):
    """Data not in the expected format."""


class LLMError(Exception):
    """LLM failed to respond."""


class CorruptionError(Exception):
    """Internal state corruption detected. A model's state is corrupted."""


class JevAuthError(LLMError):
    """Typesafe rejected the TYPESAFE_API_KEY (HTTP 401 or 403)."""


class JevCostCapError(LLMError):
    """A Jev run would pass, or has passed, its cost cap (``max_cost_usd``)."""


class MissingAPIKeyError(ValueError):
    """A model was chosen whose API key is not set."""


class ProviderAuthError(LLMError):
    """A provider rejected the API key, or the account has no quota left."""


class BudgetExceededError(LLMError):
    """A run would pass, or has passed, its cost cap (``max_cost_usd``)."""
