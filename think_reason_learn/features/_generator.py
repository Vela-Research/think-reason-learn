"""LLM-based generation of lambda feature rules.

Uses the TRL :mod:`~think_reason_learn.core.llms` singleton and Pydantic
structured output to generate binary classification rules as Python lambda
expression strings.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from think_reason_learn.core.llms import LLMChoice, llm

from ._prompts import build_system_prompt, build_user_prompt, format_samples
from ._types import DataSchema, GeneratedRules, HelperFunction, Rule

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Generate binary classification rules via an LLM.

    Parameters
    ----------
    schema:
        Describes the input data structure, the lambda parameter name, and
        includes example rules for the prompt.
    helpers:
        Helper functions to advertise in the prompt (the LLM will reference
        them inside the generated lambda expressions).
    llm_priority:
        LLM provider / model choices, tried in order (see
        :class:`~think_reason_learn.core.llms.LLMChoice`).
    temperature:
        Sampling temperature for the LLM call.
    """

    def __init__(
        self,
        schema: DataSchema,
        helpers: Sequence[HelperFunction],
        llm_priority: Sequence[LLMChoice],
        temperature: float = 0.7,
    ) -> None:
        self._schema = schema
        self._helpers = list(helpers)
        self._llm_priority = list(llm_priority)
        self._temperature = temperature

    async def generate(
        self,
        samples: Sequence[dict[str, Any]],
        labels: Sequence[Any],
        *,
        n_rules: int = 30,
        n_samples: int = 60,
        prior_rules: Sequence[Rule] | None = None,
    ) -> list[Rule]:
        """Generate binary classification rules from structured data samples.

        Parameters
        ----------
        samples:
            All available structured data records (dicts).
        labels:
            Parallel label sequence (one per record, e.g. ``1`` / ``0``).
        n_rules:
            Number of rules to request from the LLM.
        n_samples:
            Number of sample records to include in the prompt.
        prior_rules:
            Optional rules from a previous iteration, injected as feedback.

        Returns:
        -------
        list[Rule]
            The generated rules, ready for :class:`FeatureEvaluator`.
        """
        system_prompt = build_system_prompt(
            self._schema,
            self._helpers,
            n_rules,
        )
        samples_text = format_samples(samples, labels, n_samples)
        user_prompt = build_user_prompt(
            samples_text,
            n_rules,
            prior_rules,
        )

        response = await llm.respond(
            query=user_prompt,
            llm_priority=self._llm_priority,
            response_format=GeneratedRules,
            instructions=system_prompt,
            temperature=self._temperature,
        )

        if response.response is None:
            logger.warning("LLM returned no response for rule generation")
            return []

        rules = [
            Rule(
                name=r.name,
                description=r.description,
                expression=r.expression,
            )
            for r in response.response.rules
        ]
        logger.info("Generated %d rules", len(rules))
        return rules
