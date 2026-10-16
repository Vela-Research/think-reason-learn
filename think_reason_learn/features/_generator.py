"""LLM-based generation of lambda feature rules.

A reasoning model writes binary classification rules as Python lambda
expression strings, returned as Pydantic structured output.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from think_reason_learn.llm import (
    LLMSession,
    Models,
    Usage,
    resolve_models,
    run_async,
    run_sync,
)

from ._prompts import build_system_prompt, build_user_prompt, format_samples
from ._types import CognitiveMode, DataSchema, GeneratedRules, HelperFunction, Rule

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Generate binary classification rules with a reasoning model.

    Args:
        schema: The input data structure, the lambda parameter name, and
            example rules for the prompt.
        helpers: Helper functions to advertise in the prompt; generated
            lambdas may call them.
        generator: Model(s) that write the rules, e.g. ``"openai:gpt-5-mini"``;
            set a temperature with ``LLM(..., temperature=0.7)``.
        cognitive_modes: Cognitive reasoning constraints to add to the system
            prompt (CoFEE-style); none keeps the baseline prompt.

    Attributes:
        usage: Every call made, with tokens and cost.
    """

    def __init__(
        self,
        schema: DataSchema,
        helpers: Sequence[HelperFunction],
        generator: Models,
        cognitive_modes: set[CognitiveMode] | None = None,
    ) -> None:
        self._schema = schema
        self._helpers = list(helpers)
        resolve_models(generator, param="generator", task="generate")  # fail early
        self._generator = generator  # resolved per call, so no key is held
        self._cognitive_modes = cognitive_modes or set()
        self.usage = Usage()

    def generate(
        self,
        samples: Sequence[dict[str, Any]],
        labels: Sequence[Any],
        *,
        n_rules: int = 30,
        n_samples: int = 60,
        prior_rules: Sequence[Rule] | None = None,
    ) -> list[Rule]:
        """Generate rules from structured records and their labels.

        Args:
            samples: Structured records (dicts).
            labels: One label per record, e.g. ``1`` / ``0``.
            n_rules: Rules to ask for.
            n_samples: Records shown in the prompt.
            prior_rules: Rules from a previous round, given as feedback.

        Returns:
            The rules, ready for ``FeatureEvaluator``.
        """
        return run_sync(
            self._generate(samples, labels, n_rules, n_samples, prior_rules)
        )

    async def agenerate(
        self,
        samples: Sequence[dict[str, Any]],
        labels: Sequence[Any],
        *,
        n_rules: int = 30,
        n_samples: int = 60,
        prior_rules: Sequence[Rule] | None = None,
    ) -> list[Rule]:
        """Generate rules, from async code. See ``generate``."""
        return await run_async(
            self._generate(samples, labels, n_rules, n_samples, prior_rules)
        )

    async def _generate(
        self,
        samples: Sequence[dict[str, Any]],
        labels: Sequence[Any],
        n_rules: int,
        n_samples: int,
        prior_rules: Sequence[Rule] | None,
    ) -> list[Rule]:
        system_prompt = build_system_prompt(
            self._schema, self._helpers, n_rules, cognitive_modes=self._cognitive_modes
        )
        user_prompt = build_user_prompt(
            format_samples(samples, labels, n_samples), n_rules, prior_rules
        )
        written = await LLMSession(usage=self.usage).generate(
            resolve_models(self._generator, param="generator", task="generate"),
            user_prompt,
            role="generator",
            schema=GeneratedRules,
            instructions=system_prompt,
        )
        rules = [
            Rule(name=r.name, description=r.description, expression=r.expression)
            for r in written.rules
        ]
        logger.info("Generated %d rules", len(rules))
        return rules
