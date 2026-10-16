"""Think Reason Learn.

An open-source Python library for building interpretable, tree/forest-style
decision-making systems powered by large language models (LLMs).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


def _detect_version() -> str:
    """The installed version; a source checkout that is not installed has none."""
    try:
        return _pkg_version("think-reason-learn")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__: str = _detect_version()

from think_reason_learn.llm import (  # noqa: E402
    LLM,
    Price,
    config_context,
    get_config,
    set_config,
    set_limits,
)

__all__ = [
    "LLM",
    "Price",
    "__version__",
    "config_context",
    "get_config",
    "set_config",
    "set_limits",
]
