"""Deterministic fake LLM for offline PolicyInduction testing."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Literal, Set, Type

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.core.llms._schemas import (
    NOT_GIVEN,
    LLMChoice,
    LLMResponse,
    NotGiven,
    T,
)
from think_reason_learn.policy_induction._policy_induction import (
    Answer,
    BatchedAnswers,
    Policies,
    PolicyAnswer,
)
from think_reason_learn.policy_induction._prompts import max_policy_num_tag


_FAKE_PROVIDER = OpenAIChoice(model="gpt-4.1-nano")

# Pulls the policy text and sample out of an unbatched scoring query.
_SINGLE_RE = re.compile(r"Policy:\n(.*?)\n\nSample:\n(.*)", re.DOTALL)
# Pulls the policy block and sample out of a batched scoring query.
_BATCH_RE = re.compile(r"Policies:\n(.*?)\n\nSample:\n(.*)", re.DOTALL)
_ID_RE = re.compile(r"^id=(\d+): (.*)$", re.MULTILINE)


class FakePolicyLLM:
    """Drop-in replacement for ``LLM`` covering PolicyInduction's call sites.

    Dispatches on ``response_format``:

    1. ``str``             -- ``set_task`` template (carries the max-policy tag)
    2. ``Policies``        -- ``_generate_policies``
    3. ``Answer``          -- ``_score_policy_single`` (unbatched + re-query)
    4. ``BatchedAnswers``  -- ``_score_policy_batch``

    Verdicts are a pure function of (policy text, sample text), so the batched
    and unbatched paths must agree cell-for-cell. That equality is what the
    ``policy_batch_size=1`` test asserts.

    Args:
        n_policies: How many policies ``_generate_policies`` returns.
        drop_ids: Local ids omitted from every batched response.
        shift_ids: Added to every returned ``policy_id``, breaking alignment.
        shift_first_only: Apply ``shift_ids`` to the first batch call only.
        fail_after: Raise on batch call N and every one after it (1-indexed).
        fail_single: Make every individual ``Answer`` call raise.
    """

    def __init__(
        self,
        n_policies: int = 6,
        drop_ids: Set[int] | None = None,
        shift_ids: int = 0,
        shift_first_only: bool = False,
        fail_after: int | None = None,
        fail_single: bool = False,
    ) -> None:
        self.n_policies = n_policies
        self.drop_ids = drop_ids or set()
        self.shift_ids = shift_ids
        self.shift_first_only = shift_first_only
        self.fail_after = fail_after
        self.fail_single = fail_single
        self._call_count = 0
        self.calls: List[Dict[str, Any]] = []

    # ── Call accounting ────────────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        return self._call_count

    def count_of(self, response_format: Any) -> int:
        """Number of calls made with the given response_format."""
        return sum(1 for c in self.calls if c["response_format"] is response_format)

    @property
    def batch_calls(self) -> int:
        return self.count_of(BatchedAnswers)

    @property
    def single_calls(self) -> int:
        return self.count_of(Answer)

    def reset(self) -> None:
        self._call_count = 0
        self.calls = []

    # ── Verdict ────────────────────────────────────────────────────────────────

    @staticmethod
    def verdict(policy_text: str, sample_str: str) -> Literal["YES", "NO"]:
        """Stable YES/NO for a (policy, sample) pair, independent of batching."""
        digest = hashlib.md5(f"{policy_text}||{sample_str}".encode()).digest()
        return "YES" if digest[0] % 2 == 0 else "NO"

    def policy_texts(self) -> List[str]:
        return [
            f"Policy {i}: the founder shows trait {i}." for i in range(self.n_policies)
        ]

    # ── Dispatch ───────────────────────────────────────────────────────────────

    async def respond(
        self,
        query: str,
        llm_priority: List[LLMChoice],
        response_format: Type[T],
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        **kwargs: Dict[str, Any],
    ) -> LLMResponse[Any]:
        self._call_count += 1
        self.calls.append(
            {
                "query": query,
                "response_format": response_format,
                "instructions": instructions,
                "n": self._call_count,
            }
        )

        if response_format is Policies:
            return self._policies_response()
        if response_format is Answer:
            return self._single_response(query)
        if response_format is BatchedAnswers:
            return self._batch_response(query)
        if response_format is str:
            return self._set_task_response()
        raise TypeError(
            f"FakePolicyLLM: unknown response_format {response_format!r}. "
            "Add a handler for this new call site."
        )

    # ── Canned responses ───────────────────────────────────────────────────────

    def _set_task_response(self) -> LLMResponse[str]:
        return LLMResponse(
            response=(
                f"Enrich the existing policies, at most {max_policy_num_tag} "
                "in total, using the labelled samples below."
            ),
            logprobs=[],
            total_tokens=50,
            provider_model=_FAKE_PROVIDER,
        )

    def _policies_response(self) -> LLMResponse[Policies]:
        return LLMResponse(
            response=Policies(policies=self.policy_texts()),
            logprobs=[],
            total_tokens=100,
            provider_model=_FAKE_PROVIDER,
        )

    def _single_response(self, query: str) -> LLMResponse[Answer]:
        if self.fail_single:
            raise RuntimeError("FakePolicyLLM: injected single-call failure")
        m = _SINGLE_RE.search(query)
        if m is None:
            raise AssertionError(f"Unparseable single-scoring query:\n{query}")
        policy_text, sample_str = m.group(1).strip(), m.group(2).strip()
        return LLMResponse(
            response=Answer(answer=self.verdict(policy_text, sample_str)),
            logprobs=[],
            total_tokens=15,
            provider_model=_FAKE_PROVIDER,
        )

    def _batch_response(self, query: str) -> LLMResponse[BatchedAnswers]:
        batch_n = self.batch_calls  # 1-indexed: this call is already recorded
        if self.fail_after is not None and batch_n >= self.fail_after:
            raise RuntimeError("FakePolicyLLM: injected batch failure")

        m = _BATCH_RE.search(query)
        if m is None:
            raise AssertionError(f"Unparseable batch-scoring query:\n{query}")
        block, sample_str = m.group(1), m.group(2).strip()

        shift = self.shift_ids
        if self.shift_first_only and batch_n != 1:
            shift = 0

        answers = [
            PolicyAnswer(
                policy_id=int(local_id) + shift,
                answer=self.verdict(text.strip(), sample_str),
            )
            for local_id, text in _ID_RE.findall(block)
            if int(local_id) not in self.drop_ids
        ]
        return LLMResponse(
            response=BatchedAnswers(answers=answers),
            logprobs=[],
            total_tokens=20 * max(len(answers), 1),
            provider_model=_FAKE_PROVIDER,
        )
