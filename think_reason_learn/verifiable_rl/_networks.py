"""The policy and classifier networks.

The policy has the paper's two-part head: a softmax over the information actions
and a sigmoid for STOP. Both networks are small MLPs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, outputs),
    )


class PolicyNet(nn.Module):
    """Logits for each information action, then one for STOP."""

    def __init__(self, state_dim: int, n_slots: int, hidden: int) -> None:
        """Build the MLP for ``n_slots`` information actions plus STOP."""
        super().__init__()
        self.net = _mlp(state_dim, hidden, n_slots + 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Logits, shape ``(batch, n_slots + 1)``."""
        return self.net(states)


def action_probs(logits: torch.Tensor) -> torch.Tensor:
    """Action probabilities: STOP by its sigmoid; the slots share the rest."""
    go_on = torch.sigmoid(-logits[:, -1:])  # 1 - P(STOP), exact when STOP saturates
    stop = torch.sigmoid(logits[:, -1:])
    return torch.cat([torch.softmax(logits[:, :-1], dim=1) * go_on, stop], dim=1)


class Classifier(nn.Module):
    """The logit of the positive class from a partly observed state."""

    def __init__(self, state_dim: int, hidden: int) -> None:
        """Build the MLP."""
        super().__init__()
        self.net = _mlp(state_dim, hidden, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Logits, shape ``(batch,)``."""
        return self.net(states).squeeze(-1)
