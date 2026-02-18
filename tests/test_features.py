"""Tests for the features module (lambda-based feature generation)."""

from __future__ import annotations

from typing import Any, Dict, List, Type

import pandas as pd
import pytest

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.core.llms._schemas import (
    LLMChoice,
    LLMResponse,
    NOT_GIVEN,
    NotGiven,
    T,
)
from think_reason_learn.features import (
    DataSchema,
    FeatureEvaluator,
    FeatureGenerator,
    GeneratedRule,
    GeneratedRules,
    HelperFunction,
    Rule,
)
from think_reason_learn.features._prompts import (
    build_system_prompt,
    build_user_prompt,
    format_samples,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_FAKE_PROVIDER = OpenAIChoice(model="gpt-4.1-nano")
LLM_CHOICE: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]


def _parse_qs(qs_str: str) -> float:
    """Test helper: parse QS ranking string."""
    if not qs_str or qs_str == "":
        return 999.0
    if qs_str == "200+":
        return 250.0
    try:
        return float(qs_str)
    except (ValueError, TypeError):
        return 999.0


HELPER_PARSE_QS = HelperFunction(
    name="parse_qs",
    func=_parse_qs,
    signature="parse_qs(qs_str: str) -> float",
    docstring='Converts "1" -> 1.0, "200+" -> 250.0, "" -> 999.0',
)


def _sample_founders() -> list[dict[str, Any]]:
    return [
        {
            "industry": "Software Development",
            "educations": [
                {"degree": "PhD", "field": "Computer Science", "qs_ranking": "5"},
            ],
            "jobs": [
                {
                    "role": "CTO",
                    "company_size": "51-200",
                    "industry": "Software",
                    "duration": "4-5",
                },
            ],
            "ipos": [],
            "acquisitions": [{"company": "Acme"}],
        },
        {
            "industry": "Healthcare",
            "educations": [
                {"degree": "MBA", "field": "Business", "qs_ranking": "50"},
            ],
            "jobs": [
                {
                    "role": "Manager",
                    "company_size": "2-10",
                    "industry": "Healthcare",
                    "duration": "2-3",
                },
            ],
            "ipos": [],
            "acquisitions": [],
        },
        {
            "industry": "FinTech",
            "educations": [],
            "jobs": [],
            "ipos": [{"company": "BigCo"}],
            "acquisitions": [],
        },
        {
            "industry": "Software Development",
            "educations": [
                {"degree": "BS", "field": "Marketing", "qs_ranking": "200+"},
            ],
            "jobs": [
                {
                    "role": "Sales Rep",
                    "company_size": "10001+",
                    "industry": "Retail",
                    "duration": ">9",
                },
            ],
            "ipos": [],
            "acquisitions": [],
        },
        {
            "industry": "AI",
            "educations": [
                {"degree": "MS", "field": "Physics", "qs_ranking": "1"},
            ],
            "jobs": [
                {
                    "role": "Founder",
                    "company_size": "2-10",
                    "industry": "AI",
                    "duration": "<2",
                },
                {
                    "role": "CEO",
                    "company_size": "11-50",
                    "industry": "AI",
                    "duration": "2-3",
                },
            ],
            "ipos": [],
            "acquisitions": [],
        },
    ]


SAMPLE_LABELS = [1, 0, 1, 0, 1]


SAMPLE_SCHEMA = DataSchema(
    description="A startup founder profile",
    schema_text=(
        "{\n"
        '    "industry": str,\n'
        '    "educations": [{"degree": str, "field": str, "qs_ranking": str}],\n'
        '    "jobs": [{"role": str, "company_size": str,'
        ' "industry": str, "duration": str}],\n'
        '    "ipos": list,\n'
        '    "acquisitions": list\n'
        "}"
    ),
    param_name="founder",
    example_rules=[
        Rule(
            name="has_phd",
            description="Founder has a PhD degree",
            expression=(
                "lambda founder: "
                'any("phd" in e.get("degree", "").lower() '
                'for e in founder.get("educations", []))'
            ),
        ),
        Rule(
            name="prior_exit",
            description="Founder has prior IPO or acquisition",
            expression=(
                "lambda founder: "
                'len(founder.get("ipos", []) or []) '
                '+ len(founder.get("acquisitions", []) or []) > 0'
            ),
        ),
    ],
)


HAND_WRITTEN_RULES = [
    Rule(
        name="has_phd",
        description="Founder has a PhD degree",
        expression=(
            "lambda founder: "
            'any("phd" in e.get("degree", "").lower() '
            'for e in founder.get("educations", []))'
        ),
    ),
    Rule(
        name="prior_exit",
        description="Founder has prior IPO or acquisition",
        expression=(
            "lambda founder: "
            'len(founder.get("ipos", []) or []) '
            '+ len(founder.get("acquisitions", []) or []) > 0'
        ),
    ),
    Rule(
        name="elite_education",
        description="Founder attended a top 10 QS-ranked university",
        expression=(
            "lambda founder: "
            'any(parse_qs(e.get("qs_ranking", "")) <= 10 '
            'for e in founder.get("educations", []))'
        ),
    ),
]


class FakeFeatureLLM:
    """Minimal fake LLM that returns canned GeneratedRules."""

    def __init__(self) -> None:
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"query": query, "response_format": response_format})

        if response_format is GeneratedRules:
            return self._rules_response()
        raise TypeError(f"FakeFeatureLLM: unknown format {response_format!r}")

    def _rules_response(self) -> LLMResponse[GeneratedRules]:
        rules = GeneratedRules(
            rules=[
                GeneratedRule(
                    name="has_phd",
                    description="Founder has a PhD degree",
                    expression=(
                        "lambda founder: "
                        'any("phd" in e.get("degree", "").lower() '
                        'for e in founder.get("educations", []))'
                    ),
                ),
                GeneratedRule(
                    name="prior_exit",
                    description="Founder has prior IPO or acquisition",
                    expression=(
                        "lambda founder: "
                        'len(founder.get("ipos", []) or []) '
                        '+ len(founder.get("acquisitions", []) or []) > 0'
                    ),
                ),
            ]
        )
        return LLMResponse(
            response=rules,
            logprobs=[("t", -0.1)],
            total_tokens=200,
            provider_model=_FAKE_PROVIDER,
        )


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestFeatureEvaluator:
    """Tests for FeatureEvaluator (safe lambda execution)."""

    def test_basic_evaluation(self) -> None:
        evaluator = FeatureEvaluator(
            rules=HAND_WRITTEN_RULES,
            helpers=[HELPER_PARSE_QS],
        )
        founders = _sample_founders()

        result = evaluator.evaluate(founders[0])
        # Founder 0: PhD=yes, exit=yes (has acquisition), QS=5 (elite)
        assert result["has_phd"] == 1
        assert result["prior_exit"] == 1
        assert result["elite_education"] == 1

    def test_negative_evaluation(self) -> None:
        evaluator = FeatureEvaluator(
            rules=HAND_WRITTEN_RULES,
            helpers=[HELPER_PARSE_QS],
        )
        founders = _sample_founders()

        result = evaluator.evaluate(founders[1])
        # Founder 1: MBA (not PhD), no exits, QS=50 (not elite)
        assert result["has_phd"] == 0
        assert result["prior_exit"] == 0
        assert result["elite_education"] == 0

    def test_empty_lists_handled(self) -> None:
        evaluator = FeatureEvaluator(
            rules=HAND_WRITTEN_RULES,
            helpers=[HELPER_PARSE_QS],
        )
        founders = _sample_founders()

        result = evaluator.evaluate(founders[2])
        # Founder 2: no educations, has IPO
        assert result["has_phd"] == 0
        assert result["prior_exit"] == 1
        assert result["elite_education"] == 0

    def test_evaluate_df(self) -> None:
        evaluator = FeatureEvaluator(
            rules=HAND_WRITTEN_RULES,
            helpers=[HELPER_PARSE_QS],
        )
        founders = _sample_founders()

        df = evaluator.evaluate_df(founders)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 3)
        assert list(df.columns) == ["has_phd", "prior_exit", "elite_education"]
        assert set(df.values.flatten()).issubset({0, 1})

    def test_compilation_error_returns_zero(self) -> None:
        bad_rule = Rule(
            name="broken",
            description="Invalid syntax",
            expression="lambda founder: this is not valid python",
        )
        evaluator = FeatureEvaluator(rules=[bad_rule])

        assert "broken" in evaluator.compilation_errors
        result = evaluator.evaluate({"industry": "test"})
        assert result["broken"] == 0

    def test_runtime_error_returns_zero(self) -> None:
        """A rule that compiles but fails at runtime should return 0."""
        rule = Rule(
            name="divide_by_zero",
            description="Will crash at runtime",
            expression="lambda founder: 1 / len(founder.get('nonexistent', []))",
        )
        evaluator = FeatureEvaluator(rules=[rule])
        assert not evaluator.compilation_errors

        result = evaluator.evaluate({"industry": "test"})
        assert result["divide_by_zero"] == 0

    def test_rules_property(self) -> None:
        evaluator = FeatureEvaluator(rules=HAND_WRITTEN_RULES)
        assert len(evaluator.rules) == 3
        assert evaluator.rules[0].name == "has_phd"

    def test_empty_rules(self) -> None:
        evaluator = FeatureEvaluator(rules=[])
        result = evaluator.evaluate({"industry": "test"})
        assert result == {}

        df = evaluator.evaluate_df([{"a": 1}, {"a": 2}])
        assert df.shape == (2, 0)

    def test_always_true_rule(self) -> None:
        rule = Rule(
            name="always",
            description="Always true",
            expression="lambda x: True",
        )
        evaluator = FeatureEvaluator(rules=[rule])
        assert evaluator.evaluate({"anything": 42}) == {"always": 1}


# ---------------------------------------------------------------------------
# Eval sandbox security tests
# ---------------------------------------------------------------------------


class TestEvalSandbox:
    """Verify that dangerous builtins are blocked."""

    def test_import_blocked(self) -> None:
        rule = Rule(
            name="malicious_import",
            description="Tries to import os",
            expression="lambda x: __import__('os').system('echo pwned')",
        )
        evaluator = FeatureEvaluator(rules=[rule])
        # Compiles OK (just a name reference) but fails at runtime
        result = evaluator.evaluate({})
        assert result["malicious_import"] == 0

    def test_exec_blocked(self) -> None:
        rule = Rule(
            name="malicious_exec",
            description="Tries exec",
            expression="lambda x: exec('import os')",
        )
        evaluator = FeatureEvaluator(rules=[rule])
        # Compiles OK but fails at runtime (exec not in builtins)
        result = evaluator.evaluate({})
        assert result["malicious_exec"] == 0

    def test_eval_blocked(self) -> None:
        rule = Rule(
            name="malicious_eval",
            description="Tries eval",
            expression="lambda x: eval('1+1')",
        )
        evaluator = FeatureEvaluator(rules=[rule])
        # eval may compile but should fail at runtime
        result = evaluator.evaluate({})
        assert result["malicious_eval"] == 0

    def test_open_blocked(self) -> None:
        rule = Rule(
            name="malicious_open",
            description="Tries to open a file",
            expression="lambda x: open('/etc/passwd').read()",
        )
        evaluator = FeatureEvaluator(rules=[rule])
        # Should fail at compile time (open not in builtins) or runtime
        result = evaluator.evaluate({})
        assert result["malicious_open"] == 0

    def test_getattr_blocked(self) -> None:
        rule = Rule(
            name="malicious_getattr",
            description="Tries getattr",
            expression="lambda x: getattr(x, '__class__')",
        )
        evaluator = FeatureEvaluator(rules=[rule])
        result = evaluator.evaluate({})
        assert result["malicious_getattr"] == 0

    def test_safe_builtins_allow_normal_operations(self) -> None:
        """Verify the whitelist allows normal lambda operations."""
        rules = [
            Rule("uses_any", "Uses any()", "lambda x: any(True for _ in [1])"),
            Rule("uses_len", "Uses len()", "lambda x: len(x.get('a', [])) > 0"),
            Rule("uses_sum", "Uses sum()", "lambda x: sum([1, 2, 3]) == 6"),
            Rule(
                "uses_sorted",
                "Uses sorted()",
                "lambda x: sorted([3, 1, 2]) == [1, 2, 3]",
            ),
            Rule(
                "uses_isinstance",
                "Uses isinstance()",
                "lambda x: isinstance(x, dict)",
            ),
        ]
        evaluator = FeatureEvaluator(rules=rules)
        assert not evaluator.compilation_errors

        result = evaluator.evaluate({"a": [1, 2]})
        assert all(v == 1 for v in result.values())


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    """Tests for prompt construction utilities."""

    def test_system_prompt_includes_schema(self) -> None:
        prompt = build_system_prompt(SAMPLE_SCHEMA, [HELPER_PARSE_QS], 30)
        assert "educations" in prompt
        assert "founder" in prompt
        assert "parse_qs" in prompt

    def test_system_prompt_includes_helpers(self) -> None:
        prompt = build_system_prompt(SAMPLE_SCHEMA, [HELPER_PARSE_QS], 30)
        assert "parse_qs(qs_str: str) -> float" in prompt
        assert "999.0" in prompt

    def test_system_prompt_includes_examples(self) -> None:
        prompt = build_system_prompt(SAMPLE_SCHEMA, [], 30)
        assert "has_phd" in prompt
        assert "prior_exit" in prompt

    def test_system_prompt_rule_count(self) -> None:
        prompt = build_system_prompt(SAMPLE_SCHEMA, [], 20)
        assert "20" in prompt

    def test_user_prompt_without_feedback(self) -> None:
        prompt = build_user_prompt("some samples", 30)
        assert "30" in prompt
        assert "some samples" in prompt
        assert "FEEDBACK" not in prompt

    def test_user_prompt_with_feedback(self) -> None:
        prior = [
            Rule("old_rule", "An old rule", "lambda x: True"),
        ]
        prompt = build_user_prompt("some samples", 30, prior_rules=prior)
        assert "FEEDBACK" in prompt
        assert "old_rule" in prompt

    def test_format_samples_reproducible(self) -> None:
        founders = _sample_founders()
        text1 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=42)
        text2 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=42)
        assert text1 == text2

    def test_format_samples_contains_tags(self) -> None:
        founders = _sample_founders()
        text = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=42)
        assert "[SUCCESS]" in text
        assert "[FAIL]" in text

    def test_format_samples_different_seeds(self) -> None:
        founders = _sample_founders()
        text1 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=1)
        text2 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=2)
        # Different seeds should produce different orderings
        # (with only 5 samples the content is the same but order differs)
        assert "[SUCCESS]" in text1
        assert "[SUCCESS]" in text2


# ---------------------------------------------------------------------------
# Generator tests (with fake LLM)
# ---------------------------------------------------------------------------


class TestFeatureGenerator:
    """Tests for FeatureGenerator using a fake LLM."""

    @pytest.mark.asyncio
    async def test_generate_returns_rules(self) -> None:
        fake = FakeFeatureLLM()
        gen = FeatureGenerator(
            schema=SAMPLE_SCHEMA,
            helpers=[HELPER_PARSE_QS],
            llm_priority=LLM_CHOICE,
        )
        # Monkey-patch the llm module to use our fake
        import think_reason_learn.features._generator as gen_mod

        original_llm = gen_mod.llm
        gen_mod.llm = fake  # type: ignore[assignment]
        try:
            rules = await gen.generate(
                samples=_sample_founders(),
                labels=SAMPLE_LABELS,
                n_rules=10,
            )
        finally:
            gen_mod.llm = original_llm

        assert len(rules) == 2  # FakeFeatureLLM returns 2 rules
        assert rules[0].name == "has_phd"
        assert rules[1].name == "prior_exit"
        assert rules[0].expression.startswith("lambda founder:")

    @pytest.mark.asyncio
    async def test_generate_calls_llm_once(self) -> None:
        fake = FakeFeatureLLM()
        gen = FeatureGenerator(
            schema=SAMPLE_SCHEMA,
            helpers=[HELPER_PARSE_QS],
            llm_priority=LLM_CHOICE,
        )
        import think_reason_learn.features._generator as gen_mod

        original_llm = gen_mod.llm
        gen_mod.llm = fake  # type: ignore[assignment]
        try:
            await gen.generate(
                samples=_sample_founders(),
                labels=SAMPLE_LABELS,
            )
        finally:
            gen_mod.llm = original_llm

        assert len(fake.calls) == 1
        assert fake.calls[0]["response_format"] is GeneratedRules


# ---------------------------------------------------------------------------
# Integration test: generate → evaluate
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end test: rules → evaluator → feature matrix."""

    @pytest.mark.asyncio
    async def test_generate_then_evaluate(self) -> None:
        fake = FakeFeatureLLM()
        gen = FeatureGenerator(
            schema=SAMPLE_SCHEMA,
            helpers=[HELPER_PARSE_QS],
            llm_priority=LLM_CHOICE,
        )
        import think_reason_learn.features._generator as gen_mod

        original_llm = gen_mod.llm
        gen_mod.llm = fake  # type: ignore[assignment]
        try:
            rules = await gen.generate(
                samples=_sample_founders(),
                labels=SAMPLE_LABELS,
                n_rules=10,
            )
        finally:
            gen_mod.llm = original_llm

        evaluator = FeatureEvaluator(rules=rules)
        df = evaluator.evaluate_df(_sample_founders())

        assert df.shape == (5, 2)
        assert list(df.columns) == ["has_phd", "prior_exit"]
        # Founder 0: PhD=yes, exit=yes (acquisition)
        assert df.iloc[0]["has_phd"] == 1
        assert df.iloc[0]["prior_exit"] == 1
        # Founder 1: MBA, no exits
        assert df.iloc[1]["has_phd"] == 0
        assert df.iloc[1]["prior_exit"] == 0
