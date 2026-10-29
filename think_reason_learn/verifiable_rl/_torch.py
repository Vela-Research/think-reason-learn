"""PyTorch helpers: tensors from arrays, shuffled batches, seeded networks, weights.

Some wrap PyTorch functions whose annotations are partial, with the ``pyright:
ignore`` comments they need.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


def tensor(
    array: npt.NDArray[np.generic], like: torch.Tensor | torch.device
) -> torch.Tensor:
    """``array`` as a tensor on ``like``'s device (and dtype, for a tensor)."""
    converted = torch.from_numpy(array)  # pyright: ignore[reportUnknownMemberType]  # partial annotation
    return converted.to(like)


def shuffled_batches(
    n: int, size: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """Row indices ``0..n-1`` shuffled by ``generator``, in batches of ``size``."""
    order = torch.randperm(n, generator=generator)
    return tuple(order[start : start + size] for start in range(0, n, size))


def seeded[M: nn.Module](build: Callable[[], M], seed: int, device: str) -> M:
    """A network ``build`` makes, its weights drawn from its own seeded generator.

    It is built on the meta device, so PyTorch's process-wide generator is
    neither read nor changed: fits running at once cannot disturb each other.
    Linear layers get PyTorch's default initialisation.
    """
    with torch.device("meta"):
        network = build()
    network = network.to_empty(device=device)
    generator = torch.Generator().manual_seed(seed)
    for layer in network.modules():
        if isinstance(layer, nn.Linear):
            weight = torch.empty(layer.weight.shape)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
            bound = 1 / math.sqrt(layer.in_features)
            bias = torch.empty(layer.out_features).uniform_(
                -bound, bound, generator=generator
            )
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
    return network


type Weights = dict[str, dict[str, torch.Tensor]]
"""Each network's ``state_dict``, by network."""


def save_weights(path: Path, weights: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    """Write networks' weights to ``path``."""
    torch.save({name: dict(state) for name, state in weights.items()}, path)


def load_weights(path: Path, device: str) -> Weights:
    """Read weights written by ``save_weights``: tensors only, never code.

    Raises:
        ValueError: The file holds something else.
    """
    loaded: object = torch.load(path, map_location=device, weights_only=True)
    return {name: _tensors(state, path) for name, state in _pairs(loaded, path)}


def _pairs(value: object, path: Path) -> list[tuple[str, object]]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not hold network weights.")
    pairs: list[tuple[object, object]] = list(value.items())  # pyright: ignore[reportUnknownArgumentType]  # checked to be a dict
    return [(str(key), item) for key, item in pairs]


def _tensors(state: object, path: Path) -> dict[str, torch.Tensor]:
    return {k: v for k, v in _pairs(state, path) if isinstance(v, torch.Tensor)}
