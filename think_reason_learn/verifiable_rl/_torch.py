"""PyTorch helpers: tensors from arrays, shuffled batches, a private seed.

Two wrap PyTorch functions whose annotations are partial, with the ``pyright:
ignore`` comments they need.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

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
