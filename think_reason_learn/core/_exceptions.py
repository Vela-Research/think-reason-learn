class DataError(Exception):
    """Data not in the expected format."""


class LLMError(Exception):
    """LLM failed to respond."""


class CorruptionError(Exception):
    """Raised when a condition is not supposed to be met. Signifies a bug in model state."""
