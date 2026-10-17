"""PyTorch helpers: tensors from arrays, shuffled batches, a private seed, weights.

Some wrap PyTorch functions whose annotations are partial, with the ``pyright:
ignore`` comments they need.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


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


@contextmanager
def private_cpu_rng(seed: int) -> Generator[None]:
    """Seed torch's CPU generator inside the block, restoring it afterwards."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]  # partial annotation
        torch.default_generator.manual_seed(seed)
        yield


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
