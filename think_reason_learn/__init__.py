"""Think Reason Learn.

An open-source Python library for building interpretable, tree/forest-style
decision-making systems powered by large language models (LLMs).
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _detect_version() -> str:
    """Return the installed distribution version, with a dev fallback.

    First tries importlib.metadata for the installed wheel/sdist. If that fails
    (e.g., running directly from a source checkout without installation), it
    attempts to read ``pyproject.toml`` at the repository root to obtain the
    static PEP 621 ``[project].version`` value. As a last resort, returns a
    sentinel version string.
    """
    distribution_name = "think-reason-learn"

    try:
        return _pkg_version(distribution_name)
    except PackageNotFoundError:
        pass

    try:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                import tomllib
            except Exception:  # pragma: no cover - ultra-rare in py313+
                tomllib = None

            if tomllib is not None:
                data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
                project_tbl = data.get("project", {})
                project_version = project_tbl.get("version")
                if isinstance(project_version, str) and project_version:
                    return project_version
    except Exception:
        pass

    return "0.0.0+unknown"


__version__: str = _detect_version()

__all__ = ["__version__"]
