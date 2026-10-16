"""VerifiableRL: decide which information to reveal, one slot at a time.

Needs PyTorch: ``pip install 'think-reason-learn[rl]'``.
"""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "VerifiableRL needs PyTorch: pip install 'think-reason-learn[rl]'."
    ) from exc

from ._verifiable_rl import SlotFeatures, SlotTexts, VerifiableRL

__all__ = ["SlotFeatures", "SlotTexts", "VerifiableRL"]
