"""Each protocol on the wire: request bodies, replies, usage and failures."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Mapping

import httpx2
import pytest

from think_reason_learn.exceptions import LLMError, ProviderAuthError
from think_reason_learn.llm import (
    LLM,
    Answers,
    ChoiceQuestion,
    Endpoint,
    LLMSession,
    Model,
    YesNoQuestion,
    resolve,
    resolve_models,
    run_sync,
)
from think_reason_learn.llm._adapters._base import Failure, ProviderError, classify
from think_reason_learn.llm._calls import _drop, _dropped

type Handler = Callable[[httpx2.Request], httpx2.Response]
type Wire = Callable[[Handler], list[httpx2.Request]]

NEEDS_ANTHROPIC = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="needs the [anthropic] extra",
)

TECH = {"tech": YesNoQuestion("Is the founder technical?")}


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
        system, user = body["messages"]
        assert user["content"].startswith("Sample:\nAn engineer.")
        assert user["content"].endswith("Answer with exactly one of: YES, NO.")
        assert "temperature" not in body and body["prompt_cache_key"]
        assert body["response_format"]["type"] == "json_schema"
        assert _label(result, "tech") == "YES"
        [record] = list(session.usage)
        assert (
            record.input_tokens,
            record.cache_read_tokens,
            record.reasoning_tokens,
        ) == (120, 100, 2)

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
        assert "temperature" not in body and body["output_config"]["format"]
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
        assert (record.served_model, record.cost_usd) == (
            "jev-1.13.0",
            pytest.approx(0.042),
        )

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
        assert _yes(result, "tech") == pytest.approx(0.8)
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


def test_an_unreadable_reply_is_retried_not_a_crash(wire: Wire) -> None:
    replies = iter([httpx2.Response(200, text="<html>proxy error</html>")])
    sent = wire(lambda r: next(replies, None) or _chat('{"a1": "YES"}'))
    _, [result] = _answer(LLM("openai:gpt-5-mini", api_key="k"))
    assert _label(result, "tech") == "YES" and len(sent) == 2


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
