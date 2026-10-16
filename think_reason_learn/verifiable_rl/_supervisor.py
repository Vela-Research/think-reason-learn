"""The optional LLM supervisor: a nudge when the policy cannot choose.

When the policy's top two information actions are nearly tied, a reasoning model
is shown only what has been revealed so far and asked which slot to reveal next.
Its pick gets a multiplicative boost; STOP is never changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from think_reason_learn.llm import Endpoint, LLMSession, run_sync

INSTRUCTIONS: Final = """\
You advise an agent that gathers information about a sample one slot at a time
before making a decision. Given what it has seen so far, pick the one remaining
slot most likely to change the decision. If none stands out, pick none."""


class _Pick(BaseModel):
    slot: str | None = None


def is_uncertain(probs: npt.NDArray[np.float64], n_slots: int, delta: float) -> bool:
    """Whether the top two information actions differ by less than ``delta``."""
    if n_slots < 2:
        return False
    top = np.sort(probs[:n_slots])[-2:]
    return bool(top[1] - top[0] < delta)


def boost(
    probs: npt.NDArray[np.float64], n_slots: int, slot: int, factor: float
) -> npt.NDArray[np.float64]:
    """Multiply one information action by ``1 + factor``, keeping STOP's share."""
    boosted = probs.copy()
    info_mass = boosted[:n_slots].sum()
    boosted[slot] *= 1 + factor
    boosted[:n_slots] *= info_mass / boosted[:n_slots].sum()
    return boosted


class Supervisor:
    """Asks a reasoning model which unrevealed slot to reveal next."""

    def __init__(
        self,
        models: tuple[Endpoint, ...],
        session: LLMSession,
        slot_names: Sequence[str],
    ) -> None:
        """Advise with ``models``, recording calls in ``session``."""
        self.models = models
        self.session = session
        self.slot_names = list(slot_names)

    def pick(self, seen: Mapping[str, str], available: Sequence[str]) -> int | None:
        """The index of the suggested slot, or ``None``.

        Args:
            seen: Text of each revealed slot; nothing unrevealed is shown.
            available: Slots that may be revealed next.
        """
        shown = (
            "\n".join(f"{slot}: {text}" for slot, text in seen.items())
            or "(nothing yet)"
        )
        options = ", ".join(available)
        prompt = f"Revealed so far:\n{shown}\n\nSlots you may reveal next: {options}"
        reply = run_sync(
            self.session.generate(
                self.models,
                prompt,
                role="supervisor",
                schema=_Pick,
                instructions=INSTRUCTIONS,
            )
        )
        slot = reply.slot
        return (
            self.slot_names.index(slot)
            if slot is not None and slot in available
            else None
        )
