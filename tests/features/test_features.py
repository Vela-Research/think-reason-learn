"""Tests for the features module (lambda-based feature generation)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pandas as pd
import pytest

from think_reason_learn.exceptions import DataError
from think_reason_learn.features import (
    CognitiveMode,
    DataSchema,
    FeatureEvaluator,
    FeatureGenerator,
    HelperFunction,
    Rule,
)
from think_reason_learn.features._prompts import (
    build_cognitive_section,
    build_system_prompt,
    build_user_prompt,
    format_samples,
)
from think_reason_learn.features._types import GeneratedRule, GeneratedRules, Record
from think_reason_learn.testing import FakeLLM, fake_llms

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


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


SAMPLE_FOUNDERS: Final[tuple[Record, ...]] = (
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
)


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
        founders = SAMPLE_FOUNDERS

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
        founders = SAMPLE_FOUNDERS

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
        founders = SAMPLE_FOUNDERS

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
        founders = SAMPLE_FOUNDERS

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
        founders = SAMPLE_FOUNDERS
        text1 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=42)
        text2 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=42)
        assert text1 == text2

    def test_format_samples_contains_tags(self) -> None:
        founders = SAMPLE_FOUNDERS
        text = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=42)
        assert "[positive: 1]" in text
        assert "[negative: 0]" in text

    def test_format_samples_different_seeds(self) -> None:
        founders = SAMPLE_FOUNDERS
        text1 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=1)
        text2 = format_samples(founders, SAMPLE_LABELS, n_samples=4, random_state=2)
        # Different seeds should produce different orderings
        # (with only 5 samples the content is the same but order differs)
        assert "[positive: 1]" in text1
        assert "[positive: 1]" in text2


# ---------------------------------------------------------------------------
# Generator tests (with fake LLM)
# ---------------------------------------------------------------------------


class TestFeatureGenerator:
    """FeatureGenerator with a fake model."""

    def test_generate_returns_rules_in_one_call(self, writer: FakeLLM) -> None:
        rules = _generator().generate(
            samples=SAMPLE_FOUNDERS, labels=SAMPLE_LABELS, n_rules=10
        )
        assert [r.name for r in rules] == ["has_phd", "prior_exit"]
        assert rules[0].expression.startswith("lambda founder:")
        assert writer.n_calls == 1 and writer.calls[0].schema is GeneratedRules

    @pytest.mark.asyncio
    async def test_async_generate(self, writer: FakeLLM) -> None:
        rules = await _generator().agenerate(
            samples=SAMPLE_FOUNDERS, labels=SAMPLE_LABELS
        )
        assert len(rules) == 2


# ---------------------------------------------------------------------------
# Integration test: generate → evaluate
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end test: rules → evaluator → feature matrix."""

    def test_generate_then_evaluate(self, writer: FakeLLM) -> None:
        rules = _generator().generate(
            samples=SAMPLE_FOUNDERS, labels=SAMPLE_LABELS, n_rules=10
        )
        df = FeatureEvaluator(rules=rules).evaluate_df(SAMPLE_FOUNDERS)
        assert df.shape == (5, 2)
        assert list(df.columns) == ["has_phd", "prior_exit"]
        assert df.iloc[0]["has_phd"] == 1 and df.iloc[0]["prior_exit"] == 1
        assert df.iloc[1]["has_phd"] == 0 and df.iloc[1]["prior_exit"] == 0


CANNED_RULES = GeneratedRules(
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


@pytest.fixture
def writer() -> Iterator[FakeLLM]:
    fake = FakeLLM(generate=lambda request: CANNED_RULES)
    with fake_llms(writer=fake):
        yield fake


def _generator() -> FeatureGenerator:
    return FeatureGenerator(
        schema=SAMPLE_SCHEMA, helpers=[HELPER_PARSE_QS], generator="fake:writer"
    )


# ---------------------------------------------------------------------------
# Cognitive prompt tests
# ---------------------------------------------------------------------------


class TestCognitiveMode:
    """Tests for CognitiveMode enum and cognitive prompt sections."""

    def test_cognitive_mode_values(self) -> None:
        assert CognitiveMode.BACKWARD_CHAINING == "backward_chaining"
        assert CognitiveMode.SUBGOAL_DECOMPOSITION == "subgoal_decomposition"
        assert CognitiveMode.VERIFICATION == "verification"
        assert CognitiveMode.BACKTRACKING == "backtracking"
        assert len(CognitiveMode) == 4

    def test_build_cognitive_section_empty(self) -> None:
        assert build_cognitive_section(set()) == ""

    def test_build_cognitive_section_single_mode(self) -> None:
        section = build_cognitive_section({CognitiveMode.BACKWARD_CHAINING})
        assert "BACKWARD CHAINING" in section
        assert "causal hypothesis" in section
        assert "SUBGOAL" not in section
        assert "VERIFICATION" not in section
        assert "BACKTRACKING" not in section

    def test_build_cognitive_section_all_modes(self) -> None:
        section = build_cognitive_section(set(CognitiveMode))
        assert "BACKWARD CHAINING" in section
        assert "SUBGOAL DECOMPOSITION" in section
        assert "VERIFICATION" in section
        assert "BACKTRACKING" in section
        assert "COGNITIVE REASONING" in section

    def test_build_cognitive_section_stable_ordering(self) -> None:
        """Sections appear in enum definition order regardless of set ordering."""
        section = build_cognitive_section(
            {CognitiveMode.BACKTRACKING, CognitiveMode.BACKWARD_CHAINING}
        )
        bc_pos = section.index("BACKWARD CHAINING")
        bt_pos = section.index("BACKTRACKING")
        assert bc_pos < bt_pos

    def test_system_prompt_without_cognitive_modes(self) -> None:
        prompt_default = build_system_prompt(SAMPLE_SCHEMA, [], 30)
        prompt_none = build_system_prompt(SAMPLE_SCHEMA, [], 30, cognitive_modes=None)
        prompt_empty = build_system_prompt(SAMPLE_SCHEMA, [], 30, cognitive_modes=set())
        assert "COGNITIVE REASONING" not in prompt_default
        assert prompt_default == prompt_none == prompt_empty

    def test_system_prompt_with_cognitive_modes(self) -> None:
        prompt = build_system_prompt(
            SAMPLE_SCHEMA,
            [],
            30,
            cognitive_modes={CognitiveMode.VERIFICATION},
        )
        assert "VERIFICATION" in prompt
        # Existing sections still present
        assert "DATA STRUCTURE" in prompt
        assert "IMPORTANT" in prompt
        assert "has_phd" in prompt

    def test_generator_passes_cognitive_modes(self, writer: FakeLLM) -> None:
        generator = FeatureGenerator(
            schema=SAMPLE_SCHEMA,
            helpers=[HELPER_PARSE_QS],
            generator="fake:writer",
            cognitive_modes={CognitiveMode.BACKWARD_CHAINING},
        )
        rules = generator.generate(
            samples=SAMPLE_FOUNDERS, labels=SAMPLE_LABELS, n_rules=10
        )
        assert len(rules) == 2
        assert "BACKWARD" in (writer.calls[0].instructions or "")


class TestReviewedSandbox:
    """Escapes found in review are refused before a rule runs."""

    @pytest.mark.parametrize(
        "expression",
        [
            "lambda r: [c for c in ().__class__.__base__.__subclasses__()][0]",
            "lambda r: r.get.__self__",
            "lambda r: '{0.__class__}'.format(r)",
            "lambda r: [g := (g.gi_frame.f_back for _ in [1]), list(g)]",
            "lambda r: __builtins__",
            "lambda a, b: a",
            "(lambda r: r)(1)",
        ],
    )
    def test_refused_at_compile_time(self, expression: str) -> None:
        evaluator = FeatureEvaluator(rules=[Rule("bad", "", expression)])
        assert "bad" in evaluator.compilation_errors
        assert evaluator.evaluate({}) == {"bad": 0}

    def test_a_rule_cannot_change_the_records_others_read(self) -> None:
        record: dict[str, object] = {"tags": ["ai"], "name": "Ada"}
        rules = [
            Rule("wipe", "", "lambda r: r.clear() or r['tags'].pop()"),
            Rule("has_tags", "", "lambda r: len(r['tags']) > 0"),
        ]
        evaluator = FeatureEvaluator(rules=rules)
        assert "wipe" in evaluator.compilation_errors
        assert evaluator.evaluate(record) == {"wipe": 0, "has_tags": 1}
        assert record == {"tags": ["ai"], "name": "Ada"}

    def test_a_rule_cannot_rebind_a_name_for_other_rules(self) -> None:
        """A lambda's default is evaluated when ``eval`` builds it, in the scope
        it compiles in, so this would otherwise make ``len`` return 0 for every
        later rule.
        """
        record: dict[str, object] = {"tags": ["ai"]}
        rules = [
            Rule("poison", "", "lambda r=(len := lambda x: 0): False"),
            Rule("has_tags", "", "lambda r: len(r['tags']) > 0"),
        ]
        evaluator = FeatureEvaluator(rules=rules)
        assert "poison" in evaluator.compilation_errors
        assert evaluator.evaluate(record) == {"poison": 0, "has_tags": 1}

    def test_each_rule_compiles_in_its_own_scope(self) -> None:
        """Even if a write into a rule's scope were ever possible, it reaches
        only that rule.
        """
        rules = [Rule("a", "", "lambda r: 1"), Rule("b", "", "lambda r: 1")]
        evaluator = FeatureEvaluator(rules=rules)
        scopes = [e.function.__globals__ for e in evaluator._compiled if e.function]
        assert scopes[0] is not scopes[1]

    def test_rules_that_fail_on_the_data_are_counted_and_warned_of(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A rule that raises on every record scores 0 everywhere, which no
        column tells apart from a rule that is never true.
        """
        rules = [
            Rule("needs_number", "", "lambda r: r['v'] > 1_000_000"),
            Rule("fine", "", "lambda r: len(r['v']) > 0"),
        ]
        records: list[dict[str, object]] = [{"v": "text"}, {"v": "more text"}]
        evaluator = FeatureEvaluator(rules=rules)
        with caplog.at_level("WARNING"):
            frame = evaluator.evaluate_df(records)
        assert list(frame["needs_number"]) == [0, 0]
        assert list(frame["fine"]) == [1, 1]
        assert evaluator.rule_failures == {"needs_number": 2}
        assert "needs_number" in caplog.text

    def test_a_rule_too_deep_to_parse_is_refused(self) -> None:
        deep = Rule("deep", "", "lambda r: " + "-" * 100_000 + "1")
        assert "deep" in FeatureEvaluator(rules=[deep]).compilation_errors

    def test_endless_recursion_gives_zero(self) -> None:
        rule = Rule("deep", "", "lambda r: (f := lambda x: f(x)) and f(1)")
        assert FeatureEvaluator(rules=[rule]).evaluate({}) == {"deep": 0}

    def test_rule_names_must_differ(self) -> None:
        with pytest.raises(ValueError, match="distinct names"):
            FeatureEvaluator(
                rules=[Rule("a", "", "lambda r: 1"), Rule("a", "", "lambda r: 0")]
            )


def test_the_positive_label_is_chosen(writer: FakeLLM) -> None:
    labels = ["YES" if label else "NO" for label in SAMPLE_LABELS]
    _generator().generate(samples=SAMPLE_FOUNDERS, labels=labels, pos_label="YES")
    assert "[positive: YES]" in (writer.calls[0].prompt or "")
    with pytest.raises(DataError, match="pos_label="):
        _generator().generate(samples=SAMPLE_FOUNDERS, labels=labels)


def test_repeated_rule_names_are_made_distinct() -> None:
    twice = GeneratedRules(rules=[*CANNED_RULES.rules, CANNED_RULES.rules[0]])
    with fake_llms(writer=FakeLLM(generate=lambda request: twice)):
        rules = _generator().generate(samples=SAMPLE_FOUNDERS, labels=SAMPLE_LABELS)
    assert [r.name for r in rules] == ["has_phd", "prior_exit", "has_phd_2"]
    FeatureEvaluator(rules=rules)  # accepts them


def test_each_call_draws_afresh(writer: FakeLLM) -> None:
    generator = _generator()
    for _ in range(2):
        generator.generate(samples=SAMPLE_FOUNDERS, labels=SAMPLE_LABELS)
    assert writer.n_calls == 2  # the second call is not the first's cached reply
