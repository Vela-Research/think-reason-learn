"""LLMSession: fallbacks, the reply cache, retries, budgets and usage, offline."""

from __future__ import annotations

from pathlib import Path


import pytest
from pydantic import BaseModel

from think_reason_learn.core.exceptions import (
    BudgetExceededError,
    LLMError,
    ProviderAuthError,
)
from think_reason_learn.llm import (
    LLM,
    ChoiceQuestion,
    LLMSession,
    Price,
    YesNoQuestion,
    config_context,
    resolve_models,
    run_sync,
)
from think_reason_learn.llm import Endpoint, get_config
from think_reason_learn.llm._cache import open_cache
from think_reason_learn.testing import (
    Failure,
    FakeLLM,
    GenerateRequest,
    fake_llms,
    scripted,
)

QUESTIONS = {
    "tech": YesNoQuestion("Is the founder technical?"),
    "stage": ChoiceQuestion("What stage is the company?", ("seed", "series_a")),
}


class Topics(BaseModel):
    topics: list[str]


def _answer(sample: str, question: YesNoQuestion | ChoiceQuestion) -> str:
    if isinstance(question, ChoiceQuestion):
        return "seed"
    return "YES" if "engineer" in sample else "NO"


def _writers(*names: str) -> tuple[LLM, ...]:
    return tuple(LLM(f"fake:{name}") for name in names)


def test_generate_text_and_schema() -> None:
    writer = FakeLLM(generate=lambda r: Topics(topics=["a"]) if r.schema else "hello")
    with fake_llms(writer=writer):
        session = LLMSession()
        models = resolve_models("fake:writer", param="generator", task="generate")
        assert run_sync(session.generate(models, "Say hi", role="generator")) == "hello"
        topics = run_sync(
            session.generate(models, "List", role="generator", schema=Topics)
        )
    assert topics == Topics(topics=["a"])
    assert [r.role for r in session.usage] == ["generator", "generator"]


def test_generate_falls_back_when_a_model_gives_nothing() -> None:
    empty, good = FakeLLM(generate=lambda r: "  "), FakeLLM(generate=lambda r: "ok")
    with fake_llms(empty=empty, good=good):
        output = run_sync(
            LLMSession().generate(_resolved("empty", "good"), "p", role="generator")
        )
    assert output == "ok"
    assert (empty.n_calls, good.n_calls) == (1, 1)


def test_generate_raises_when_every_model_fails() -> None:
    with fake_llms(bad=FakeLLM(failures=["bad_request"])):
        with pytest.raises(LLMError, match="fake bad request"):
            run_sync(LLMSession().generate(_resolved("bad"), "p", role="generator"))


def test_generation_is_cached_per_draw() -> None:
    writer = FakeLLM(generate=scripted(["first", "second"]))
    with fake_llms(writer=writer):
        models = _resolved("writer")
        again = [
            run_sync(LLMSession().generate(models, "p", role="g")) for _ in range(2)
        ]
        other = run_sync(LLMSession().generate(models, "p", role="g", draw=1))
    assert again == ["first", "first"] and other == "second"
    assert writer.n_calls == 2


def test_answers_from_a_reasoning_model_are_certain() -> None:
    judge = FakeLLM(answer=_answer)
    with fake_llms(judge=judge):
        results = run_sync(
            LLMSession().answer(
                _resolved("judge"),
                ["an engineer", "a lawyer"],
                QUESTIONS,
                role="answerer",
            )
        )
    assert [r["tech"].label for r in results if r["tech"]] == ["YES", "NO"]
    first = results[0]["stage"]
    assert first is not None and first.probabilities == {"seed": 1.0, "series_a": 0.0}
    assert judge.n_calls == 4  # one question per call by default


def test_decision_models_give_probabilities_in_one_call_per_sample() -> None:
    jev = FakeLLM(
        answer=lambda s, q: 0.8 if isinstance(q, YesNoQuestion) else "seed",
        kind="decision",
    )
    with fake_llms(jev=jev):
        [result] = run_sync(
            LLMSession().answer(
                _resolved("jev"), ["an engineer"], QUESTIONS, role="answerer"
            )
        )
    tech = result["tech"]
    assert tech is not None and (tech.label, tech.probability()) == ("YES", 0.8)
    assert jev.n_calls == 1


def test_questions_per_call_groups_questions() -> None:
    judge = FakeLLM(answer=_answer)
    with fake_llms(judge=judge):
        models = resolve_models(
            LLM("fake:judge", questions_per_call=2), param="answerer", task="answer"
        )
        run_sync(LLMSession().answer(models, ["a", "b"], QUESTIONS, role="answerer"))
    assert judge.n_calls == 2


def test_unanswered_questions_fall_back_to_the_next_model() -> None:
    first = FakeLLM(answer=lambda s, q: "YES" if isinstance(q, YesNoQuestion) else None)
    second = FakeLLM(answer=_answer)
    with fake_llms(first=first, second=second):
        [result] = run_sync(
            LLMSession().answer(
                _resolved("first", "second"), ["x"], QUESTIONS, role="a"
            )
        )
    assert result["tech"] is not None and result["stage"] is not None
    assert [c.questions for c in second.calls] == [(QUESTIONS["stage"].text,)]


def test_nothing_answered_raises() -> None:
    with fake_llms(mute=FakeLLM(answer=lambda s, q: None)):
        with pytest.raises(LLMError, match="No answers"):
            run_sync(LLMSession().answer(_resolved("mute"), ["x"], QUESTIONS, role="a"))


def test_answers_are_never_paid_for_twice() -> None:
    judge = FakeLLM(answer=_answer)
    with fake_llms(judge=judge):
        models = _resolved("judge")
        run_sync(LLMSession().answer(models, ["x", "x"], QUESTIONS, role="a"))
        assert judge.n_calls == 2  # identical samples are asked once
        session = LLMSession()
        run_sync(session.answer(models, ["x", "y"], QUESTIONS, role="a"))
    assert judge.n_calls == 4  # only "y" is new
    [hits] = [r for r in session.usage if r.cached]
    assert (hits.calls, hits.cost_usd) == (2, 0.0)


def test_unreadable_cache_entries_are_asked_again(tmp_path: Path) -> None:
    judge = FakeLLM(answer=_answer)
    with fake_llms(judge=judge):
        models = _resolved("judge")
        run_sync(LLMSession().answer(models, ["x"], QUESTIONS, role="a"))
        cache = open_cache(get_config().cache_path)
        rows = cache._db.execute("SELECT key FROM replies").fetchall()
        cache.put_many({key: {"label": "corrupt"} for (key,) in rows})
        [result] = run_sync(LLMSession().answer(models, ["x"], QUESTIONS, role="a"))
    assert all(answer is not None for answer in result.values())
    assert judge.n_calls == 4


def test_no_cache_asks_again() -> None:
    judge = FakeLLM(answer=_answer)
    with fake_llms(judge=judge), config_context(cache=False):
        for _ in range(2):
            run_sync(
                LLMSession().answer(_resolved("judge"), ["x"], QUESTIONS, role="a")
            )
    assert judge.n_calls == 4


@pytest.mark.parametrize("failure", ["rate_limit", "server_error", "timeout"])
def test_temporary_failures_are_retried(failure: Failure) -> None:
    judge = FakeLLM(answer=_answer, failures=[failure])
    with fake_llms(judge=judge):
        [result] = run_sync(
            LLMSession().answer(
                _resolved("judge"), ["x"], {"tech": QUESTIONS["tech"]}, role="a"
            )
        )
    assert result["tech"] is not None and judge.n_calls == 2


@pytest.mark.parametrize("failure", ["auth", "quota"])
def test_a_rejected_key_stops_the_run(failure: Failure) -> None:
    judge = FakeLLM(answer=_answer, failures=[failure])
    with fake_llms(judge=judge):
        with pytest.raises(ProviderAuthError):
            run_sync(
                LLMSession().answer(_resolved("judge"), ["x"], QUESTIONS, role="a")
            )


def test_budget_refuses_work_estimated_over_the_cap() -> None:
    judge = FakeLLM(answer=_answer)
    with fake_llms(judge=judge), config_context(max_cost_usd=0.0001):
        models = resolve_models(
            LLM("fake:judge", price=Price(input=10.0)), param="a", task="answer"
        )
        with pytest.raises(BudgetExceededError, match="Nothing was sent"):
            run_sync(LLMSession().answer(models, ["x" * 4000], QUESTIONS, role="a"))
    assert judge.n_calls == 0


def test_budget_stops_once_spending_passes_the_cap() -> None:
    writer = FakeLLM(generate=lambda r: "ok", tokens=(1_000_000, 0))
    with fake_llms(writer=writer), config_context(max_cost_usd=0.5):
        models = resolve_models(
            LLM("fake:writer", price=Price(input=1.0)), param="g", task="generate"
        )
        with pytest.raises(BudgetExceededError, match="cap"):
            run_sync(LLMSession().generate(models, "p", role="g"))


def test_usage_records_cost_and_roles() -> None:
    judge = FakeLLM(answer=_answer, tokens=(1_000, 100))
    with fake_llms(judge=judge):
        models = resolve_models(
            LLM("fake:judge", price=Price(input=1.0, output=2.0)),
            param="a",
            task="answer",
        )
        session = LLMSession()
        run_sync(session.answer(models, ["x"], QUESTIONS, role="answerer"))
    assert session.usage.calls == 2
    assert session.usage.cost_usd == pytest.approx(2 * (1_000 * 1.0 + 100 * 2.0) / 1e6)
    summary = session.usage.by("role", "model")
    assert summary.to_dict("records")[0]["calls"] == 2


def test_unknown_price_is_reported_not_zero() -> None:
    with fake_llms(judge=FakeLLM(answer=_answer)):
        session = LLMSession()
        run_sync(session.answer(_resolved("judge"), ["x"], QUESTIONS, role="a"))
    assert session.usage.unpriced == {"fake:judge"}


def test_progress_counts_answers() -> None:
    seen: list[int] = []
    with fake_llms(judge=FakeLLM(answer=_answer)):
        run_sync(
            LLMSession().answer(
                _resolved("judge"),
                ["x", "y"],
                QUESTIONS,
                role="a",
                progress=seen.append,
            )
        )
    assert sum(seen) == 4


@pytest.mark.asyncio
async def test_works_inside_a_running_event_loop() -> None:
    """A notebook cell runs inside an event loop; sync calls still work there."""
    with fake_llms(writer=FakeLLM(generate=lambda r: "ok")):

        async def cell() -> str:
            return run_sync(LLMSession().generate(_resolved("writer"), "p", role="g"))

        assert await cell() == "ok"


def test_generate_request_reaches_the_model() -> None:
    seen: list[GenerateRequest] = []

    def reply(request: GenerateRequest) -> str:
        seen.append(request)
        return "ok"

    with fake_llms(writer=FakeLLM(generate=reply)):
        run_sync(
            LLMSession().generate(
                _resolved("writer"),
                "p",
                role="g",
                instructions="be brief",
                prefix="Sample: x",
            )
        )
    assert (seen[0].instructions, seen[0].prefix) == ("be brief", "Sample: x")


def _resolved(*names: str) -> tuple[Endpoint, ...]:
    return resolve_models(_writers(*names), param="models", task="answer")
