"""Offline tests for PolicyInduction scoring and prediction with Jev."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from think_reason_learn.core.llms import (
    JevBudget,
    JevChoice,
    NoulQuestion,
    OpenAIChoice,
)
from think_reason_learn.core.llms._schemas import LLMResponse
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.policy_induction import _policy_induction as pi_module
from think_reason_learn.policy_induction._policy_induction import (
    Answer as PIAnswer,
    Policies,
)
from tests.fake_jev import FakeJevLLM

POLICIES = [
    "Strong technical background predicts success.",
    "Prior exits predict success.",
    "Large seed rounds predict success.",
]
TASK = "Predict whether the founder succeeds."
ROWS = [f"strong founder {i}" for i in range(5)] + [
    f"weak founder {i}" for i in range(5)
]
LABELS = ["YES"] * 5 + ["NO"] * 5


def _prob(state: str, instructions: str) -> float:
    if POLICIES[0] in instructions:
        return 0.9 if "strong" in state else 0.1
    return 0.5


class FakePILLM(FakeJevLLM):
    """Generation for PolicyInduction plus Jev-like answering."""

    async def respond(
        self,
        query: str,
        llm_priority: List[Any],
        response_format: Any,
        instructions: Any = None,
        temperature: Any = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        self.calls.append({"query": query, "response_format": response_format})
        provider = OpenAIChoice(model="gpt-4.1-nano")
        if response_format is Policies:
            response: Any = Policies(policies=POLICIES)
        elif response_format is PIAnswer:
            response = PIAnswer(answer="YES")
        else:
            raise TypeError(f"unexpected response_format {response_format!r}")
        return LLMResponse(
            response=response, logprobs=[], total_tokens=10, provider_model=provider
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePILLM:
    fake = FakePILLM(prob=_prob)
    monkeypatch.setattr(pi_module, "llm", fake)
    return fake


def _pi(tmp_path: Path, **kwargs: Any) -> PolicyInduction:
    return PolicyInduction(
        gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")],
        predict_llmc=kwargs.pop("predict_llmc", [JevChoice(max_cost_usd=2.0)]),
        config={"cv_folds": 2},
        max_policy_length=3,
        max_samples_as_context=4,
        save_path=tmp_path,
        name="jev_pi",
        confirm_requests=False,
        **kwargs,
    )


async def _fit(pi: PolicyInduction) -> PolicyInduction:
    await pi.set_task(TASK, instructions_template="Write <max_policy_length> rules.")
    return await pi.fit(pd.DataFrame({"text": ROWS}), LABELS)


@pytest.mark.asyncio
async def test_fit_scores_all_policies_in_one_request_per_sample(
    tmp_path: Path, fake: FakePILLM
) -> None:
    await _fit(_pi(tmp_path))

    assert sorted(r.state for r in fake.answer_requests) == sorted(
        f"text: {row}" for row in ROWS
    )
    for request in fake.answer_requests:
        assert len(request.questions) == len(POLICIES)
        for question, policy in zip(request.questions.values(), POLICIES):
            assert isinstance(question, NoulQuestion)
            assert policy in question.instructions
            assert TASK in question.instructions
    assert not any(c["response_format"] is PIAnswer for c in fake.calls)


@pytest.mark.asyncio
async def test_scores_are_yes_at_one_half(tmp_path: Path, fake: FakePILLM) -> None:
    pi = await _fit(_pi(tmp_path))

    memory = pi.get_memory()
    assert list(memory.at[0, "predictions"]) == ["YES"] * 5 + ["NO"] * 5
    assert list(memory.at[1, "predictions"]) == ["YES"] * 10


@pytest.mark.asyncio
async def test_fit_uses_the_choice_cap(tmp_path: Path, fake: FakePILLM) -> None:
    await _fit(_pi(tmp_path))

    [call] = fake.answer_calls
    assert isinstance(call["budget"], JevBudget)
    assert call["budget"].max_cost_usd == 2.0


@pytest.mark.asyncio
async def test_predict_asks_weighted_policies_once_per_sample(
    tmp_path: Path, fake: FakePILLM
) -> None:
    pi = await _fit(_pi(tmp_path))
    fake.answer_requests.clear()
    fake.answer_calls.clear()

    X_new = pd.DataFrame({"text": ["strong founder x", "weak founder y"]})
    out = [(i, vec, pred) async for i, vec, pred, _ in pi.predict(X_new)]

    weighted = [pos for pos, w in enumerate(pi.lr.coef_[0]) if w != 0]
    assert len(fake.answer_calls) == 1
    assert len(fake.answer_requests) == 2
    assert all(len(r.questions) == len(weighted) for r in fake.answer_requests)
    by_index = {i: (vec, pred) for i, vec, pred in out}
    assert by_index[0][1] == "YES" and by_index[1][1] == "NO"
    assert np.array_equal(by_index[0][0][weighted], np.ones(len(weighted)))


@pytest.mark.asyncio
async def test_request_estimate_counts_one_jev_request_per_sample(
    tmp_path: Path, fake: FakePILLM
) -> None:
    pi = _pi(tmp_path)
    pi._set_data(pd.DataFrame({"text": ROWS}), LABELS)

    estimates = pi._estimate_fit_requests()

    assert estimates["jev-latest (scoring)"] == len(ROWS)


@pytest.mark.asyncio
async def test_chat_scoring_keeps_the_chat_path(
    tmp_path: Path, fake: FakePILLM
) -> None:
    await _fit(_pi(tmp_path, predict_llmc=[OpenAIChoice(model="gpt-4.1-nano")]))

    assert fake.answer_requests == []
    assert any(c["response_format"] is PIAnswer for c in fake.calls)
