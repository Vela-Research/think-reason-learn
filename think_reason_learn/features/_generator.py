"""LLM-based generation of lambda feature rules.

A reasoning model writes binary classification rules as Python lambda
expression strings, returned as Pydantic structured output.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from think_reason_learn.exceptions import DataError
from think_reason_learn.llm import (
    LLMSession,
    Models,
    Usage,
    resolve_models,
    run_async,
    run_sync,
)

from ._prompts import build_system_prompt, build_user_prompt, format_samples
from ._types import (
    CognitiveMode,
    DataSchema,
    GeneratedRules,
    HelperFunction,
    Record,
    Rule,
)

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
        random_state: Seed for the records shown; each call shows another
            draw, so repeated calls give new rules. ``None`` for a random seed.

    Attributes:
        usage: Every call made, with tokens and cost.
    """

    def __init__(
        self,
        schema: DataSchema,
        helpers: Sequence[HelperFunction],
        generator: Models,
        cognitive_modes: set[CognitiveMode] | None = None,
        random_state: int | None = 42,
    ) -> None:
        """Check the generator now; it is resolved again on each call."""
        self._schema = schema
        self._helpers = list(helpers)
        resolve_models(generator, param="generator", task="generate")  # fail early
        self._generator = generator  # resolved per call, so no key is held
        self._cognitive_modes = cognitive_modes or set()
        self._random_state = random_state
        self._calls = 0
        self.usage = Usage()

    def generate(
        self,
        samples: Sequence[Record],
        labels: Sequence[object],
        *,
        n_rules: int = 30,
        n_samples: int = 60,
        prior_rules: Sequence[Rule] | None = None,
        positive: object = 1,
    ) -> list[Rule]:
        """Generate rules from structured records and their labels.

        Args:
            samples: Structured records (dicts).
            labels: One label per record.
            n_rules: Rules to ask for.
            n_samples: Records shown in the prompt, half of them positive.
            prior_rules: Rules from a previous round, given as feedback.
            positive: The label the rules should find, e.g. ``"YES"``.

        Returns:
            The rules, ready for ``FeatureEvaluator``.

        Raises:
            DataError: No label equals ``positive``.
        """
        ask = _Ask(samples, labels, n_rules, n_samples, prior_rules, positive)
        return run_sync(self._generate(ask))

    async def agenerate(
        self,
        samples: Sequence[Record],
        labels: Sequence[object],
        *,
        n_rules: int = 30,
        n_samples: int = 60,
        prior_rules: Sequence[Rule] | None = None,
        positive: object = 1,
    ) -> list[Rule]:
        """Generate rules, from async code. See ``generate``."""
        ask = _Ask(samples, labels, n_rules, n_samples, prior_rules, positive)
        return await run_async(self._generate(ask))

    async def _generate(self, ask: _Ask) -> list[Rule]:
        if not any(label == ask.positive for label in ask.labels):
            raise DataError(
                f"No label equals positive={ask.positive!r}; pass positive= with "
                "the label the rules should find."
            )
        draw, self._calls = self._calls, self._calls + 1
        seed = None if self._random_state is None else self._random_state + draw
        shown = format_samples(
            ask.samples,
            ask.labels,
            ask.n_samples,
            positive=ask.positive,
            random_state=seed,
        )
        written = await LLMSession(usage=self.usage).generate(
            resolve_models(self._generator, param="generator", task="generate"),
            build_user_prompt(shown, ask.n_rules, ask.prior_rules),
            role="generator",
            schema=GeneratedRules,
            instructions=build_system_prompt(
                self._schema, self._helpers, ask.n_rules, self._cognitive_modes
            ),
            draw=draw,
        )
        rules = [Rule(r.name, r.description, r.expression) for r in written.rules]
        logger.info(f"Generated {len(rules)} rules")
        return rules


@dataclass(frozen=True, slots=True)
class _Ask:
    samples: Sequence[Record]
    labels: Sequence[object]
    n_rules: int
    n_samples: int
    prior_rules: Sequence[Rule] | None
    positive: object
