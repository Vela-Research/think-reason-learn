"""The optional LLM supervisor: a nudge when the policy cannot choose.

When the policy's top two information actions are nearly tied, a reasoning model
is shown only what has been revealed so far and asked which slot to reveal next.
Its pick gets a small added probability, as in 0.2; STOP is never changed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from think_reason_learn.exceptions import NoOutputError
from think_reason_learn.llm import Endpoint, LLMSession, run_sync

logger = logging.getLogger(__name__)

INSTRUCTIONS: Final = """\
You advise an agent that gathers information about a sample one slot at a time
before making a decision. Given what it has seen so far, pick the one remaining
slot most likely to change the decision. If none stands out, pick none."""


def is_uncertain(probs: npt.NDArray[np.float64], n_slots: int, delta: float) -> bool:
    """Whether the top two information actions differ by less than ``delta``."""
    if n_slots < 2:
        return False
    top = np.sort(probs[:n_slots])[-2:]
    return bool(top[1] - top[0] < delta)


def boost(
    probs: npt.NDArray[np.float64], n_slots: int, slot: int, bias: float
) -> npt.NDArray[np.float64]:
    """Add ``bias`` to one information action, keeping STOP's share."""
    boosted = probs.copy()
    info_mass = boosted[:n_slots].sum()
    boosted[slot] += bias
    boosted[:n_slots] *= info_mass / boosted[:n_slots].sum()
    return boosted


class Supervisor:
    """Asks a reasoning model which unrevealed slot to reveal next.

    Args:
        models: Reasoning models, in fallback order.
        session: Where calls are made and recorded.
        slot_names: Slot names, in column order.
        slot_texts: Per slot, one text per sample.
        delta: Gap between the top two slots below which the model is asked.
        bias: Probability added to the pick before the slots are rescaled.
    """

    def __init__(
        self,
        models: tuple[Endpoint, ...],
        session: LLMSession,
        slot_names: Sequence[str],
        slot_texts: Mapping[str, Sequence[str]],
        delta: float,
        bias: float,
    ) -> None:
        """Advise with ``models``, recording calls in ``session``."""
        self.models, self.session = models, session
        self.slot_names, self.slot_texts = list(slot_names), slot_texts
        self.delta, self.bias = delta, bias

    def nudge(
        self,
        probs: npt.NDArray[np.float64],
        allowed: npt.NDArray[np.bool_],
        masks: npt.NDArray[np.bool_],
        samples: Sequence[int] | npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        """Boost the model's pick in rows where the policy is unsure.

        Rows are skipped when STOP outweighs every slot together, since no boost
        could change the action. The rows' questions are asked at once.

        Args:
            probs: Action probabilities, one row per state.
            allowed: The actions allowed in each row.
            masks: The slots revealed in each row.
            samples: The sample of each row, to look up its slot texts.
        """
        n_slots = len(self.slot_names)
        probs = np.where(allowed, probs, 0.0)
        unsure = [
            k
            for k, row in enumerate(probs)
            if row[n_slots] < row[:n_slots].sum()
            and is_uncertain(row, n_slots, self.delta)
        ]
        picks = run_sync(_gather(self._pick(masks[k], samples[k]) for k in unsure))
        for k, pick in zip(unsure, picks, strict=True):
            if pick is not None:
                probs[k] = boost(probs[k], n_slots, pick, self.bias)
        return probs

    async def _pick(self, mask: npt.NDArray[np.bool_], sample: int) -> int | None:
        """The suggested slot; only revealed slots' texts are shown."""
        names = self.slot_names
        shown = (
            "\n".join(
                f"{name}: {self.slot_texts[name][sample]}"
                for name, seen in zip(names, mask, strict=True)
                if seen
            )
            or "(nothing yet)"
        )
        available = [name for name, seen in zip(names, mask, strict=True) if not seen]
        prompt = (
            f"Revealed so far:\n{shown}\n\n"
            f"Slots you may reveal next: {', '.join(available)}"
        )
        try:
            reply = await self.session.generate(
                self.models,
                prompt,
                role="supervisor",
                schema=_Pick,
                instructions=INSTRUCTIONS,
            )
        except NoOutputError as error:
            logger.warning(f"The supervisor gave no advice: {error}")
            return None
        return names.index(reply.slot) if reply.slot in available else None


async def _gather[T](calls: Iterable[Coroutine[object, object, T]]) -> list[T]:
    return list(await asyncio.gather(*calls))


class _Pick(BaseModel):
    slot: str | None = None
