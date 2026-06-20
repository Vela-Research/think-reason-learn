"""Optional LLM action supervisor for Verifiable RL.

The supervisor provides *weak* action preferences: when the policy is unsure
which slot to query next (its top-two info probabilities are nearly tied), it is
asked which remaining slot is most informative, and a small probability bias is
applied. The STOP action is never touched by the supervisor.

The core RL algorithm is fully offline; this module is only used when a
supervisor is attached to :class:`VerifiableRL`.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from think_reason_learn.core.llms import LLMChoice, TokenCounter, llm as _default_llm

from ._types import NextActionPreference

DEFAULT_SUPERVISOR_INSTRUCTIONS = (
    "You are a decision-support module for an information-gathering agent. "
    "Choose the ONE remaining information slot most likely to change the final "
    "success/failure decision. Do not suggest already-observed slots. Prefer the "
    "slot with the highest marginal information value. Return at most one slot as "
    "a weak reference; if nothing stands out, return an empty list. Return "
    "structured output only."
)


@runtime_checkable
class ActionSupervisor(Protocol):
    """Anything that can suggest the next slot to query.

    Implementations must be synchronous from the policy's point of view.
    """

    def prefer(
        self, observed: set[str], available: Sequence[str], profile: str
    ) -> str | None:
        """Return a preferred slot from ``available`` (or ``None``)."""
        ...


def is_uncertain(pi_info: np.ndarray, delta: float) -> bool:
    """Return True if the top-two info probabilities differ by less than ``delta``."""
    if len(pi_info) < 2:
        return False
    order = np.argsort(-pi_info)
    return bool((pi_info[order[0]] - pi_info[order[1]]) < delta)


def apply_action_bias(
    pi: np.ndarray,
    n_info: int,
    preferred: str | None,
    slot_names: Sequence[str],
    bias: float,
) -> np.ndarray:
    """Add ``bias`` to the preferred slot, preserving info-vs-STOP mass.

    The STOP probability (``pi[n_info]``) is left unchanged; the info portion is
    renormalised to its original total mass so the supervisor only changes *what*
    to ask next, not *when* to stop.
    """
    out = np.asarray(pi, dtype=np.float64).copy()
    if preferred is None or preferred not in slot_names:
        return out
    k = list(slot_names).index(preferred)
    if k >= n_info:
        return out
    pi_info = out[:n_info].copy()
    info_mass = float(pi_info.sum())
    pi_info[k] += bias
    pi_info = np.clip(pi_info, 1e-8, None)
    pi_info = pi_info / pi_info.sum() * info_mass
    out[:n_info] = pi_info
    return out


class LLMActionSupervisor:
    """An :class:`ActionSupervisor` backed by the TRL unified LLM interface.

    Args:
        llmc: LLM choices in priority order.
        instructions: System instructions for the preference call.
        temperature: Sampling temperature (default deterministic).
        llm_semaphore_limit: Max concurrent LLM calls.
        _llm: LLM instance for dependency injection (testing). If None, uses the
            global ``llm`` singleton.
    """

    def __init__(
        self,
        llmc: List[LLMChoice],
        *,
        instructions: str = DEFAULT_SUPERVISOR_INSTRUCTIONS,
        temperature: float = 0.0,
        llm_semaphore_limit: int = 3,
        _llm: Any = None,
    ) -> None:
        if not llmc:
            raise ValueError("llmc must be a non-empty list of LLM choices")
        if llm_semaphore_limit < 1:
            raise ValueError("llm_semaphore_limit must be >= 1")
        self.llmc = llmc
        self.instructions = instructions
        self.temperature = temperature
        self._llm = _llm if _llm is not None else _default_llm
        self._semaphore = asyncio.Semaphore(llm_semaphore_limit)
        self._token_counter = TokenCounter()
        self._cache: Dict[Tuple[frozenset, Tuple[str, ...], str], str | None] = {}

    @property
    def token_usage(self) -> TokenCounter:
        """Accumulated token usage across supervisor calls."""
        return self._token_counter

    def _build_query(
        self, observed: set[str], available: Sequence[str], profile: str
    ) -> str:
        avail_lines = "\n".join(f"- {s}: information slot '{s}'" for s in available)
        return (
            f"Founder profile (observed information only):\n{profile}\n\n"
            f"Already observed slots: {sorted(observed)}\n\n"
            f"Available slots to choose from:\n{avail_lines}\n"
        )

    async def aprefer(
        self, observed: set[str], available: Sequence[str], profile: str
    ) -> str | None:
        """Async: ask the LLM for the most informative remaining slot."""
        candidates = [s for s in available if s not in observed]
        if not candidates:
            return None

        query = self._build_query(observed, candidates, profile)
        async with self._semaphore:
            response = await self._llm.respond(
                query=query,
                llm_priority=self.llmc,
                response_format=NextActionPreference,
                instructions=self.instructions,
                temperature=self.temperature,
            )
        await self._token_counter.append(
            provider=response.provider_model.provider,
            model=response.provider_model.model,
            value=response.total_tokens,
            caller="LLMActionSupervisor.aprefer",
        )

        prefer = list(response.response.prefer) if response.response else []
        for slot in prefer:
            if slot in candidates:
                return slot
        return None

    def prefer(
        self, observed: set[str], available: Sequence[str], profile: str
    ) -> str | None:
        """Sync wrapper around :meth:`aprefer`, with per-state caching.

        The coroutine runs in an isolated worker thread so the caller's event
        loop (and the process-wide default loop) is never touched; this keeps
        the synchronous training/prediction path safe to call from anywhere.
        """
        key = (frozenset(observed), tuple(available), profile)
        if key in self._cache:
            return self._cache[key]

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                lambda: asyncio.run(self.aprefer(observed, available, profile))
            ).result()
        self._cache[key] = result
        return result
