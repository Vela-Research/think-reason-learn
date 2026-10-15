"""Each protocol on the wire: request bodies, replies, usage and failures."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import httpx2
import pytest

from think_reason_learn.core.exceptions import LLMError, ProviderAuthError
from think_reason_learn.llm import (
    LLM,
    Answers,
    ChoiceQuestion,
    LLMSession,
    Model,
    YesNoQuestion,
    resolve_models,
    run_sync,
)

type Handler = Callable[[httpx2.Request], httpx2.Response]
type Wire = Callable[[Handler], list[httpx2.Request]]

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
        assert json.loads(sent[0].content)["response_format"] == {"type": "json_object"}
        assert _label(result, "tech") == "YES"


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

    def test_azure_uses_the_api_key_header(self, wire: Wire) -> None:
        sent = wire(
            lambda r: httpx2.Response(200, json={"answers": {"tech": {"noul": 0.1}}})
        )
        _answer(
            LLM(
                "azure:Microsoft-Decision-1",
                api_key="k",
                base_url="https://r.example/systemone",
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
        _, [result] = _answer(LLM("openai:gpt-6-luna", api_key="k"), questions)

        assert sent[0].url.path.endswith("/decisions")
        wired = json.loads(sent[0].content)["questions"]
        assert wired[0] == {
            "type": "predicate",
            "name": "tech",
            "instructions": TECH["tech"].text,
        }
        assert wired[1]["choices"] == [{"value": "seed"}, {"value": "series_a"}]
        assert _label(result, "tech") == "NO" and result["stage"] is None
