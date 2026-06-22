"""Policy and classifier networks for Verifiable RL.

Both are simple MLPs, ported verbatim (architecture-wise) from the standalone
reference so the shipped ``runs/model_*`` checkpoints load with the default
hidden widths (``policy_hidden=512``, ``clf_hidden=256``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Maps a (partially observed) state vector to an action distribution.

    The output is a softmax over ``action_dim`` actions: one per information
    slot plus a final STOP action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512) -> None:
        """Build the policy MLP.

        Args:
            state_dim: Length of the concatenated state vector.
            action_dim: Number of actions (``n_slots + 1``).
            hidden: Width of the first hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action probabilities for ``x`` (1-D or 2-D)."""
        return self.net(x)


class Classifier(nn.Module):
    """Predicts a success logit from a (partially observed) state vector."""

    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        """Build the classifier MLP.

        Args:
            state_dim: Length of the concatenated state vector.
            hidden: Width of the first hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return success logits (scalar for 1-D input, ``(B,)`` for 2-D)."""
        return self.net(x).squeeze(-1)
