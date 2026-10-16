"""Typed doors into the few PyTorch functions whose annotations are partial.

The ``pyright: ignore`` comments they need sit here, so the rest of VerifiableRL
is checked strictly.
"""

from __future__ import annotations

from collections.abc import Iterator
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


def backward(loss: torch.Tensor) -> None:
    """Back-propagate ``loss``."""
    loss.backward()  # pyright: ignore[reportUnknownMemberType]  # partial annotation


@contextmanager
def private_cpu_rng(seed: int) -> Iterator[None]:
    """Seed torch's CPU generator inside the block, restoring it afterwards."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]  # partial annotation
        torch.default_generator.manual_seed(seed)
        yield
