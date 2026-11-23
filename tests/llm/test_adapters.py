"""Each protocol on the wire: request bodies, replies, usage and failures."""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Callable, Mapping

import httpx2
import pytest
from pydantic import BaseModel

from think_reason_learn.exceptions import LLMError, NoOutputError, ProviderAuthError
from think_reason_learn.llm import (
    LLM,
    Answers,
    ChoiceQuestion,
    Endpoint,
    LLMSession,
    Model,
    ReasoningEffort,
    YesNoQuestion,
    resolve_models,
    run_sync,
)
from think_reason_learn.llm._adapters._base import Failure, ProviderError, classify
from think_reason_learn.llm._calls import _drop, _dropped
from think_reason_learn.llm._spec import resolve

type Handler = Callable[[httpx2.Request], httpx2.Response]
type Wire = Callable[[Handler], list[httpx2.Request]]

NEEDS_ANTHROPIC = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="needs the [anthropic] extra",
)

TECH = {"tech": YesNoQuestion("Is the founder technical?")}
DESCRIBED = ChoiceQuestion(
    "Stage?", ("seed", "series_a"), {"seed": "before any priced round"}
)


def _answer(
    model: Model, questions: Mapping[str, YesNoQuestion | ChoiceQuestion] = TECH
) -> tuple[LLMSession, list[Answers]]:
    session = LLMSession()
    models = resolve_models(model, param="answerer", task="answer")
    return session, run_sync(
        session.answer(models, ["An engineer."], questions, role="answerer")
    )


def _chat(
    content: str, *, usage: Mapping[str, object] | None = None
) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "served-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": usage
            or {
                "prompt_tokens": 120,
                "completion_tokens": 5,
                "total_tokens": 125,
                "prompt_tokens_details": {"cached_tokens": 100},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    )


def _label(answers: Answers, qid: str) -> str | None:
    answer = answers[qid]
    return answer.label if answer is not None else None


def _yes(answers: Answers, qid: str) -> float | None:
    answer = answers[qid]
    return answer.probability() if answer is not None else None


def _error(
    status: int, message: str, headers: dict[str, str] | None = None
) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"error": {"message": message, "type": "error"}},
        headers=headers or {},
    )


class TestOpenAICompatible:
    def test_sample_first_question_last(self, wire: Wire) -> None:
        sent = wire(lambda r: _chat('{"a1": "YES"}'))
        session, [result] = _answer(LLM("openai:gpt-5-mini", api_key="k"))

        body = json.loads(sent[0].content)
        _, user = body["messages"]
        assert user["content"].startswith("Sample:\nAn engineer.")
        assert user["content"].endswith("Answer with exactly one of: YES, NO.")
        assert body["temperature"] == 0 and body["prompt_cache_key"]  # answers repeat
        assert body["response_format"]["type"] == "json_schema"
        assert _label(result, "tech") == "YES"
        [record] = list(session.usage)
        assert (
            record.input_tokens,
            record.cache_read_tokens,
            record.reasoning_tokens,
        ) == (120, 100, 2)

    def test_choice_descriptions_are_in_the_prompt(self, wire: Wire) -> None:
        sent = wire(lambda r: _chat('{"a1": "seed"}'))
        _, [result] = _answer(
            LLM("openai:gpt-5-mini", api_key="k"), {"stage": DESCRIBED}
        )
        _, user = json.loads(sent[0].content)["messages"]
        assert user["content"].endswith(
            "Answer with exactly one of: seed, series_a.\n"
            "- seed: before any priced round"
        )
        assert _label(result, "stage") == "seed"

    def test_choice_descriptions_follow_their_question(self, wire: Wire) -> None:
        sent = wire(lambda r: _chat('{"a1": "seed", "a2": "YES"}'))
        model = LLM("openai:gpt-5-mini", api_key="k", questions_per_call=2)
        _answer(model, {"stage": DESCRIBED, **TECH})
        _, user = json.loads(sent[0].content)["messages"]
        assert (
            "1. Stage? (one of: seed, series_a)\n   - seed: before any priced round\n"
            f"2. {TECH['tech'].text} (one of: YES, NO)\n"
        ) in user["content"]

    def test_rejected_temperature_is_dropped_for_good(self, wire: Wire) -> None:
        replies = iter(
            [
                _error(
                    400,
                    "Unsupported parameter: 'temperature' is not supported.",
                )
            ]
        )
        sent = wire(lambda r: next(replies, None) or _chat('{"a1": "NO"}'))
        model = LLM("openai:gpt-5-mini", api_key="k", temperature=0.0)
        _answer(model)
        _answer(model, {"other": YesNoQuestion("Other?")})

        assert [("temperature" in json.loads(r.content)) for r in sent] == [
            True,
            False,
            False,
        ]

    def test_writing_and_votes_keep_the_models_temperature(self, wire: Wire) -> None:
        sent = wire(lambda r: _chat('{"a1": "YES"}'))
        models = resolve_models(
            LLM("openai:gpt-5-mini", api_key="k"), param="answerer", task="answer"
        )
        session = LLMSession()
        run_sync(
            session.answer(models, ["An engineer."], TECH, role="a", ensemble=True)
        )
        run_sync(session.generate(models, "Write.", role="generator"))
        _answer(LLM("openai:gpt-5-mini", api_key="k", temperature=0.7))
        sent_temperatures = [json.loads(r.content).get("temperature") for r in sent]
        assert sent_temperatures == [None, None, 0.7]

    def test_a_custom_server_never_gets_the_providers_key(
        self, wire: Wire, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        sent = wire(lambda r: _error(401, "Incorrect API key provided"))
        proxy = LLM("openai:gpt-5-mini", base_url="https://proxy.example/v1")
        with pytest.raises(ProviderAuthError, match="No key was sent"):
            _answer(proxy)
        assert "sk-from-env" not in sent[0].headers["authorization"]

    def test_a_key_a_server_echoes_is_never_kept(self, wire: Wire) -> None:
        key = "sk-test-0123456789abcdef"
        wire(lambda r: _error(401, f"Invalid API key {key}: rejected " + "x" * 9000))
        session = LLMSession()
        models = resolve_models(
            LLM("openai:gpt-5-mini", api_key=key), param="a", task="answer"
        )
        with pytest.raises(ProviderAuthError) as raised:
            run_sync(session.answer(models, ["An engineer."], TECH, role="a"))
        kept = [record.error or "" for record in session.usage]
        assert key not in str(raised.value) and raised.value.__cause__ is None
        assert kept and all(key not in e and len(e) < 1000 for e in kept)

    def test_rate_limit_waits_and_retries(self, wire: Wire) -> None:
        replies = iter([_error(429, "Rate limit reached", {"retry-after": "0"})])
        sent = wire(lambda r: next(replies, None) or _chat('{"a1": "YES"}'))
        _, [result] = _answer(LLM("openai:gpt-5-mini", api_key="k"))
        assert len(sent) == 2 and result["tech"] is not None

    def test_rejected_key_stops_the_run(self, wire: Wire) -> None:
        wire(lambda r: _error(401, "Incorrect API key provided"))
        with pytest.raises(ProviderAuthError, match="Incorrect API key"):
            _answer(LLM("openai:gpt-5-mini", api_key="bad"))

    def test_a_bad_key_reported_as_400_stops_the_run(self, wire: Wire) -> None:
        wire(lambda r: _error(400, "Please pass a valid API key"))
        with pytest.raises(ProviderAuthError):
            _answer(LLM("google:gemini-2.5-flash", api_key="bad"))

    def test_openrouter_reports_its_cost(self, wire: Wire) -> None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "cost": 0.0012,
        }
        sent = wire(lambda r: _chat('{"a1": "YES"}', usage=usage))
        session, _ = _answer(
            LLM("openrouter:meta-llama/llama-3.3-70b-instruct", api_key="k")
        )

        assert "prompt_cache_key" not in json.loads(sent[0].content)
        [record] = list(session.usage)
        assert (record.cost_usd, record.cost_source) == (0.0012, "provider")

    def test_flex_tier_and_effort_are_sent(self, wire: Wire) -> None:
        sent = wire(lambda r: _chat('{"a1": "YES"}'))
        _answer(
            LLM(
                "openai:gpt-5-mini",
                api_key="k",
                service_tier="flex",
                reasoning_effort="low",
            )
        )
        body = json.loads(sent[0].content)
        assert (body["service_tier"], body["reasoning_effort"]) == ("flex", "low")

    def test_json_object_providers_are_validated(self, wire: Wire) -> None:
        sent = wire(lambda r: _chat('{"a1": "YES"}'))
        _, [result] = _answer(LLM("deepseek:deepseek-chat", api_key="k"))
        body = json.loads(sent[0].content)
        assert body["response_format"] == {"type": "json_object"}
        assert '"a1"' in body["messages"][0]["content"]  # the schema is in the prompt
        assert _label(result, "tech") == "YES"

    @pytest.mark.parametrize(
        "reply",
        [
            _chat('{"a1": "YE'),  # cut-off JSON
            _chat('{"a1": "MAYBE"}'),  # not a label
            _chat("null"),
        ],
    )
    def test_unusable_replies_are_unanswered_not_crashes(
        self, wire: Wire, reply: httpx2.Response
    ) -> None:
        wire(lambda r: reply)
        with pytest.raises(LLMError, match="No answers"):
            _answer(LLM("openai:gpt-5-mini", api_key="k"))

    def test_concurrent_drops_keep_each_other(self) -> None:
        endpoint: Endpoint = resolve(LLM("openai:gpt-5-mini", api_key="k"))
        for param in ("temperature", "reasoning_effort"):  # both saw nothing dropped
            error = ProviderError("rejected_param", "unsupported", param=param)
            assert _drop(endpoint, error, frozenset())
        assert _dropped[endpoint.key] == {"temperature", "reasoning_effort"}


@pytest.mark.parametrize(
    ("status", "message", "failure"),
    [
        (None, "connection reset", "transient"),
        (529, "overloaded", "transient"),
        (401, "nope", "auth"),
        (400, "API key not valid. Please pass a valid API key.", "auth"),
        (402, "Payment required", "quota"),
        (429, "You exceeded your current quota: insufficient_quota", "quota"),
        (400, "Your credit balance is too low", "quota"),
        (429, "Rate limit reached for requests", "rate_limit"),
        (429, "Quota exceeded; check your plan and billing details", "rate_limit"),
        (404, "The model does not exist", "unavailable"),
        (
            400,
            "Unsupported value: 'temperature' does not support 0.0",
            "rejected_param",
        ),
        (400, "max_tokens is too large", "bad_request"),  # no rejection words
        (403, "Input was flagged by moderation", "bad_request"),  # one sample
    ],
)
def test_classify(status: int | None, message: str, failure: Failure) -> None:
    assert classify(status, message, {}).failure == failure


def test_classify_names_the_rejected_setting() -> None:
    error = classify(400, "Unrecognized request argument: reasoning.effort", {})
    assert error.param == "reasoning_effort"


def _message(text: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 20,
            },
        },
    )


@NEEDS_ANTHROPIC
class TestAnthropic:
    def test_sample_is_cached_and_question_follows(self, wire: Wire) -> None:
        sent = wire(lambda r: _message('{"a1": "YES"}'))
        session, [result] = _answer(LLM("anthropic:claude-haiku-4-5", api_key="k"))

        body = json.loads(sent[0].content)
        first, second = body["messages"][0]["content"]
        assert first["cache_control"] == {"type": "ephemeral"}
        assert first["text"].startswith("Sample:") and second["text"].startswith(
            "Question:"
        )
        assert body["temperature"] == 0 and body["output_config"]["format"]
        assert _label(result, "tech") == "YES"
        [record] = list(session.usage)
        assert (
            record.input_tokens,
            record.cache_read_tokens,
            record.cache_write_tokens,
        ) == (130, 100, 20)

    def test_set_temperature_and_effort_are_sent(self, wire: Wire) -> None:
        sent = wire(lambda r: _message('{"a1": "NO"}'))
        _answer(
            LLM(
                "anthropic:claude-haiku-4-5",
                api_key="k",
                temperature=0.0,
                reasoning_effort="high",
            )
        )
        body = json.loads(sent[0].content)
        assert (body["temperature"], body["output_config"]["effort"]) == (0.0, "high")

    def test_no_effort_asks_for_the_lowest(self, wire: Wire) -> None:
        sent = wire(lambda r: _message('{"a1": "NO"}'))
        _answer(LLM("anthropic:claude-haiku-4-5", api_key="k", reasoning_effort="none"))
        assert json.loads(sent[0].content)["output_config"]["effort"] == "low"


class TestSystemOne:
    def test_jev_wire_format_and_price(self, wire: Wire) -> None:
        def reply(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    "model": "jev-1.13.0",
                    "answers": {"tech": {"noul": 0.7}, "stage": {"choice": "seed"}},
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 3},
                },
            )

        sent = wire(reply)
        questions = {**TECH, "stage": ChoiceQuestion("Stage?", ("seed", "series_a"))}
        session, [result] = _answer(LLM("jev", api_key="k"), questions)

        body = json.loads(sent[0].content)
        assert sent[0].headers["authorization"] == "Bearer k"
        assert body["state"] == "An engineer."
        assert body["questions"]["tech"] == {
            "type": "noul",
            "instructions": TECH["tech"].text,
        }
        assert body["questions"]["stage"]["criteria"] == {
            "seed": None,
            "series_a": None,
        }
        assert _yes(result, "tech") == 0.7 and _label(result, "stage") == "seed"
        [record] = list(session.usage)
        assert record.served_model == "jev-1.13.0"
        assert record.cost_usd is not None and math.isclose(record.cost_usd, 0.042)

    def test_choice_descriptions_are_sent_as_criteria(self, wire: Wire) -> None:
        sent = wire(
            lambda r: httpx2.Response(
                200, json={"answers": {"stage": {"choice": "seed"}}, "usage": {}}
            )
        )
        _answer(LLM("jev", api_key="k"), {"stage": DESCRIBED})
        assert json.loads(sent[0].content)["questions"]["stage"]["criteria"] == {
            "seed": "before any priced round",
            "series_a": None,
        }

    def test_invalid_answers_are_unanswered(self, wire: Wire) -> None:
        wire(
            lambda r: httpx2.Response(
                200, json={"answers": {"tech": {"noul": 3}}, "usage": {}}
            )
        )
        with pytest.raises(LLMError, match="No answers"):
            _answer(LLM("jev", api_key="k"))

    def test_openrouter_decision_models_and_their_cost(self, wire: Wire) -> None:
        sent = wire(
            lambda r: httpx2.Response(
                200,
                json={
                    "answers": {"tech": {"type": "noul", "noul": 0.8}},
                    "model": "openai/gpt-6-luna-decisions-20261006",
                    "usage": {"input_tokens": 400, "output_tokens": 0, "cost": 4e-5},
                },
            )
        )
        session, [result] = _answer(
            LLM("openrouter:openai/gpt-6-luna-decisions", api_key="k")
        )
        assert sent[0].url == "https://openrouter.ai/api/alpha/decisions"
        assert json.loads(sent[0].content)["model"] == "openai/gpt-6-luna-decisions"
        assert math.isclose(_yes(result, "tech") or 0.0, 0.8)
        [record] = list(session.usage)
        assert (record.cost_usd, record.cost_source) == (4e-5, "provider")

    def test_azure_uses_the_api_key_header(self, wire: Wire) -> None:
        sent = wire(
            lambda r: httpx2.Response(200, json={"answers": {"tech": {"noul": 0.1}}})
        )
        _answer(
            LLM(
                "azure:Microsoft-Decision-1",
                api_key="k",
                base_url="https://r.example/systemone",
                kind="decision",
            )
        )
        assert (
            sent[0].headers["api-key"] == "k" and "authorization" not in sent[0].headers
        )

    @pytest.mark.parametrize(
        "model",
        [
            LLM("kev", base_url="http://127.0.0.1:8009/v1/systemone", kind="decision"),
            LLM(
                "azure:Microsoft-Decision-1",
                base_url="https://r.example/systemone",
                kind="decision",
            ),
        ],
        ids=["bearer", "api-key-header"],
    )
    def test_a_keyless_server_gets_no_key_header(self, wire: Wire, model: LLM) -> None:
        # A base_url never gets the provider's key, so a decision model on your own
        # server, or a saved model loaded again, is called with no key at all.
        sent = wire(
            lambda r: httpx2.Response(200, json={"answers": {"tech": {"noul": 0.4}}})
        )
        _, [result] = _answer(model)
        assert "authorization" not in sent[0].headers
        assert "api-key" not in sent[0].headers
        assert math.isclose(_yes(result, "tech") or 0.0, 0.4)


class TestOpenAIDecisions:
    def test_predicate_choice_and_refusal(self, wire: Wire) -> None:
        def reply(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    "model": "gpt-6-luna",
                    "answers": [
                        {"type": "predicate", "name": "tech", "probability": 0.3},
                        {"type": "refusal", "name": "stage"},
                    ],
                    "usage": {
                        "input_tokens": 50,
                        "input_tokens_details": {
                            "cached_tokens": 0,
                            "cache_write_tokens": 0,
                        },
                        "output_tokens": 2,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 52,
                    },
                },
            )

        sent = wire(reply)
        questions = {**TECH, "stage": ChoiceQuestion("Stage?", ("seed", "series_a"))}
        _, [result] = _answer(
            LLM("openai:gpt-6-luna", api_key="k", kind="decision"), questions
        )

        assert sent[0].url.path.endswith("/decisions")
        wired = json.loads(sent[0].content)["questions"]
        assert wired[0] == {
            "type": "predicate",
            "name": "tech",
            "instructions": TECH["tech"].text,
        }
        assert wired[1]["choices"] == [{"value": "seed"}, {"value": "series_a"}]
        assert _label(result, "tech") == "NO" and result["stage"] is None

    def test_choice_descriptions_go_on_their_choices(self, wire: Wire) -> None:
        sent = wire(
            lambda r: httpx2.Response(
                200,
                json={
                    "model": "gpt-6-luna",
                    "answers": [
                        {
                            "type": "choice",
                            "name": "stage",
                            "choice": "seed",
                            "confidence": 0.9,
                            "probabilities": [
                                {"value": "seed", "probability": 0.9},
                                {"value": "series_a", "probability": 0.1},
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 50,
                        "input_tokens_details": {
                            "cached_tokens": 0,
                            "cache_write_tokens": 0,
                        },
                        "output_tokens": 2,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 52,
                    },
                },
            )
        )
        _, [result] = _answer(
            LLM("openai:gpt-6-luna", api_key="k", kind="decision"), {"stage": DESCRIBED}
        )
        [wired] = json.loads(sent[0].content)["questions"]
        assert wired["choices"] == [
            {"value": "seed", "description": "before any priced round"},
            {"value": "series_a"},
        ]
        assert _label(result, "stage") == "seed"


@pytest.mark.parametrize(
    "broken",
    [
        httpx2.Response(200, text="<html>proxy error</html>"),
        httpx2.Response(200, json={"choices": "zzz"}),
        _chat('{"a1": "YES"}', usage={"prompt_tokens": None, "total_tokens": 1}),
    ],
)
def test_an_unreadable_reply_is_retried_not_a_crash(
    wire: Wire, broken: httpx2.Response
) -> None:
    replies = iter([broken])
    sent = wire(lambda r: next(replies, None) or _chat('{"a1": "YES"}'))
    _, [result] = _answer(LLM("openai:gpt-5-mini", api_key="k"))
    assert _label(result, "tech") == "YES" and len(sent) == 2


def test_an_unreadable_decision_is_retried_not_a_crash(wire: Wire) -> None:
    decision = {
        "model": "gpt-6-luna",
        "answers": [{"type": "predicate", "name": "tech", "probability": 0.9}],
    }
    usage = {
        "input_tokens": 50,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 52,
    }
    replies = iter(
        [
            httpx2.Response(200, text="<html>proxy error</html>"),
            httpx2.Response(200, json=decision),  # no usage
        ]
    )
    sent = wire(
        lambda r: next(replies, None)
        or httpx2.Response(200, json={**decision, "usage": usage})
    )
    _, [result] = _answer(LLM("openai:gpt-6-luna", api_key="k", kind="decision"))
    assert _label(result, "tech") == "YES" and len(sent) == 3


def test_reasoning_tokens_left_out_of_completion_tokens_are_counted(
    wire: Wire,
) -> None:
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 610}
    wire(lambda r: _chat('{"a1": "YES"}', usage=usage))
    session, _ = _answer(LLM("xai:grok-4", api_key="k"))
    [record] = list(session.usage)
    assert record.output_tokens == 510


def test_a_model_the_account_cannot_use_tries_the_next() -> None:
    error = classify(
        403, "Project does not have access to model gpt-6 (model_not_found)", {}
    )
    assert error.failure == "unavailable"


@NEEDS_ANTHROPIC
def test_claude_gets_the_service_tier(wire: Wire) -> None:
    sent = wire(lambda r: _message('{"a1": "YES"}'))
    _answer(LLM("anthropic:claude-haiku-4-5", api_key="k", service_tier="auto"))
    assert json.loads(sent[0].content)["service_tier"] == "auto"


class _Gift(BaseModel):
    outcome: str


@pytest.mark.parametrize(
    ("content", "finish", "reason"),
    [
        ('```json\n{"outcome": "YES"}\n```', "stop", "is not a valid _Gift"),
        ('{"verdict": "YES"}', "stop", "outcome: Field required"),
        ('{"outcome": "YE', "length", "cut off at the output limit"),
        ("", "stop", "the reply was empty (finish_reason=stop)"),
    ],
)
def test_an_unusable_reply_says_why(
    wire: Wire, content: str, finish: str, reason: str
) -> None:
    def reply(request: httpx2.Request) -> httpx2.Response:
        response = _chat(content)
        body = json.loads(response.content)
        body["choices"][0]["finish_reason"] = finish
        return httpx2.Response(200, json=body)

    wire(reply)
    models = resolve_models(
        LLM("openrouter:anthropic/a-model", api_key="k"), param="g", task="generate"
    )
    with pytest.raises(NoOutputError) as raised:
        run_sync(LLMSession().generate(models, "p", role="g", schema=_Gift))
    [why] = raised.value.reasons
    assert why.startswith("openrouter:anthropic/a-model: no usable output (")
    assert reason in why and reason in str(raised.value)


def _verdict_reply(request: httpx2.Request) -> httpx2.Response:
    return _chat('{"outcome": "YES"}')


@pytest.mark.parametrize(
    ("model", "effort", "sent"),
    [
        ("openrouter:anthropic/claude-opus-5.5", "none", "low"),
        ("openrouter:anthropic/claude-opus-5.5", "minimal", "low"),
        ("openrouter:anthropic/claude-opus-5.5", "high", "high"),
        ("openrouter:openai/gpt-5-mini", "none", "none"),
    ],
)
def test_claude_through_openrouter_gets_its_lowest_effort(
    wire: Wire, model: str, effort: ReasoningEffort, sent: str
) -> None:
    requests = wire(_verdict_reply)
    llm = LLM(model, api_key="k", reasoning_effort=effort, kind="reasoning")
    models = resolve_models(llm, param="g", task="generate")
    run_sync(LLMSession().generate(models, "p", role="g", schema=_Gift))
    assert json.loads(requests[0].content)["reasoning_effort"] == sent


def test_a_model_that_must_reason_is_called_without_an_effort(wire: Wire) -> None:
    def must_reason(request: httpx2.Request) -> httpx2.Response:
        if "reasoning_effort" in json.loads(request.content):
            return _error(
                400, "Reasoning is mandatory for this endpoint and cannot be disabled."
            )
        return _verdict_reply(request)

    requests = wire(must_reason)
    llm = LLM("openrouter:some/thinker", api_key="k", reasoning_effort="none")
    models = resolve_models(llm, param="g", task="generate")
    reply = run_sync(LLMSession().generate(models, "p", role="g", schema=_Gift))
    assert reply.outcome == "YES" and len(requests) == 2
    assert "reasoning_effort" not in json.loads(requests[1].content)
    assert classify(400, "Reasoning is mandatory for this endpoint.", {}).param == (
        "reasoning_effort"
    )


def test_claude_through_openrouter_caches_the_prefix(wire: Wire) -> None:
    """Claude through OpenRouter caches only up to a breakpoint.

    The prefix goes in its own part, marked as the breakpoint.
    """
    requests = wire(_verdict_reply)
    llm = LLM("openrouter:anthropic/claude-opus-5.5", api_key="k", kind="reasoning")
    models = resolve_models(llm, param="g", task="generate")
    run_sync(
        LLMSession().generate(
            models, "the case", role="g", schema=_Gift, prefix="the policy"
        )
    )
    [user] = [
        m for m in json.loads(requests[0].content)["messages"] if m["role"] == "user"
    ]
    assert user["content"] == [
        {"type": "text", "text": "the policy", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "the case"},
    ]


@pytest.mark.parametrize(
    ("model", "prefix"),
    [
        ("openrouter:openai/gpt-5-mini", "the policy"),
        ("openrouter:anthropic/claude-opus-5.5", None),
    ],
)
def test_other_models_and_no_prefix_get_one_plain_string(
    wire: Wire, model: str, prefix: str | None
) -> None:
    requests = wire(_verdict_reply)
    llm = LLM(model, api_key="k", kind="reasoning")
    models = resolve_models(llm, param="g", task="generate")
    run_sync(
        LLMSession().generate(models, "the case", role="g", schema=_Gift, prefix=prefix)
    )
    [user] = [
        m for m in json.loads(requests[0].content)["messages"] if m["role"] == "user"
    ]
    assert user["content"] == ("the policy\n\nthe case" if prefix else "the case")
