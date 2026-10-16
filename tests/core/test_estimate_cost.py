"""estimate_cost: the calls a fit or predict would make, before any is sent."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
from sklearn.exceptions import NotFittedError

from think_reason_learn import LLM, config_context
from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.exceptions import DataError
from think_reason_learn.gptree import GPTree
from think_reason_learn.llm import Price, Question
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining
from think_reason_learn.rrf import RRF
from think_reason_learn.testing import FakeLLM, GenerateRequest, fake_llms

WRITER, JUDGE = LLM("fake:writer", price=Price(1, 4)), LLM("fake:judge", price=Price(1))
X = [
    f"Founder {i} is {'good' if i % 2 else 'bad'} at building companies."
    for i in range(80)
]
Y = ["good" if i % 2 else "bad" for i in range(80)]
NEW = [f"Another founder {i} is {'good' if i % 3 else 'bad'}." for i in range(30)]

MODELS: dict[str, Callable[[], TRLClassifier]] = {
    "rrf": lambda: RRF(task="t", generator=WRITER, answerer=JUDGE, n_prompts=3, cv=2),
    "rrf-screened": lambda: RRF(
        task="t",
        generator=WRITER,
        answerer=JUDGE,
        n_prompts=3,
        cv=2,
        screening_size=20,
        screened_questions=5,
    ),
    "policy": lambda: PolicyInduction(
        task="t",
        generator=WRITER,
        answerer=JUDGE,
        n_samples=20,
        n_rounds=2,
        batch_size=4,
        pos_label="good",
    ),
    "policy-rules": lambda: PolicyInduction(
        task="t",
        generator=WRITER,
        answerer=JUDGE,
        n_samples=20,
        n_rounds=1,
        batch_size=4,
        predictor="rules",
        cv=2,
        pos_label="good",
    ),
    "rule-mining": lambda: ReasonedRuleMining(
        task="t",
        generator=WRITER,
        answerer=JUDGE,
        n_samples=20,
        cv=2,
        pos_label="good",
    ),
    "gptree": lambda: GPTree(task="t", generator=WRITER, answerer=JUDGE, critic=True),
}


def _writer() -> FakeLLM:
    ids = itertools.count()

    def write(request: GenerateRequest) -> dict[str, object] | str:
        fields = request.schema.model_fields if request.schema is not None else {}
        if "questions" in fields:
            texts = [f"Is fact {next(ids)} true?" for _ in range(10)]
            plain = fields["questions"].annotation == list[str]
            return {"questions": texts if plain else [{"text": t} for t in texts]}
        if "rules" in fields:
            return {"rules": [f"Rule {next(ids)}." for _ in range(5)]}
        if "rule" in fields:
            return {"reasoning": "r", "rule": f"IF fact {next(ids)} THEN good"}
        return "Instructions."

    return FakeLLM(generate=write)


def _judge(sample: str, question: Question) -> float:
    return 0.97 if "good" in sample else 0.1


@pytest.mark.parametrize("name", MODELS)
def test_estimates_match_the_calls_made(name: str) -> None:
    writer, judge = _writer(), FakeLLM(answer=_judge, kind="decision")
    with fake_llms(writer=writer, judge=judge), config_context(cache=False):
        model = MODELS[name]()
        planned = model.estimate_cost(X, Y)
        assert writer.n_calls == judge.n_calls == 0  # nothing is sent
        assert planned.cost_usd > 0 and not planned.unpriced
        model.fit(X, Y)
        made = model.usage_.calls
        predicted = model.estimate_cost(NEW).calls
        model.predict(NEW)
    if name == "gptree":  # every level is counted; this toy tree stops at one
        assert made <= planned.calls
    else:
        assert planned.calls == made
    assert predicted == pytest.approx(model.usage_.calls - made, rel=0.2)


def test_predict_needs_a_fitted_model_and_fit_matching_labels() -> None:
    with fake_llms(writer=_writer(), judge=FakeLLM(answer=_judge, kind="decision")):
        with pytest.raises(NotFittedError):
            MODELS["rrf"]().estimate_cost(X)
        with pytest.raises(DataError, match="labels"):
            MODELS["rrf"]().estimate_cost(X, Y[:-1])
