"""VerifiableRL: decide which information to reveal, one slot at a time.

Needs PyTorch: ``pip install 'think-reason-learn[rl]'``.
"""

import importlib.util

if importlib.util.find_spec("torch") is None:  # pragma: no cover - only without it
    raise ImportError(
        "VerifiableRL needs PyTorch: pip install 'think-reason-learn[rl]'."
    )

from ._verifiable_rl import SlotFeatures, SlotTexts, VerifiableRL

__all__ = ["SlotFeatures", "SlotTexts", "VerifiableRL"]
