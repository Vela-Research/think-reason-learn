class DataError(Exception):
    """Data not in the expected format."""


class LLMError(Exception):
    """LLM failed to respond."""


class CorruptionError(Exception):
    """Internal state corruption detected. A model's state is corrupted."""
