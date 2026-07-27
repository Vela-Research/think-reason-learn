"""VanillaClassifier.

A direct-prompting (zero-shot) founder-success classifier. Unlike RRF or
GPTree, it builds no intermediate structure: it sends each profile to the LLM
once with a fixed scoring prompt and thresholds the returned score. It is the
canonical *baseline* for VCBench-style founder screening, and ships with the
empirically best configurations from the research report (see
:mod:`think_reason_learn.vanilla._presets`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Sequence

import pandas as pd

from think_reason_learn.core.exceptions import DataError, LLMError
from think_reason_learn.core.llms import GoogleChoice, LLMChoice, TokenCounter, llm

from ._prompts import SCALES, USER_TEMPLATE, build_system_prompt
from ._presets import DEFAULT_PRESET, VanillaPreset
from ._types import (
    FounderScore,
    ScaleSpec,
    ScoringFormat,
    VanillaMetrics,
    compute_metrics,
)

logger = logging.getLogger(__name__)


class VanillaClassifier:
    """Direct-prompting founder-success classifier.

    The classifier is stateless with respect to training data: there is no
    ``fit`` step. Construct it with a scoring format, threshold, temperature and
    model, then call :meth:`score` (or :meth:`predict`) on a dataframe of
    profiles. :meth:`evaluate` and :meth:`sweep_thresholds` help validate a
    configuration against labelled data.

    Args:
        llmc: LLMs to use, in priority order. Defaults to a single
            :class:`GoogleChoice` for the resolved ``model``.
        scoring_format: ``"binary"``, ``"ternary"`` or ``"5point"``.
        threshold: Decision threshold; ``score >= threshold`` predicts success.
            Defaults to the scale's recommended threshold.
        temperature: Decoding temperature. ``0.0`` is reproducible; ``1.0`` is a
            higher-variance, higher-coverage regime.
        model: Convenience for the default ``llmc`` when ``llmc`` is omitted.
        name: Optional human-readable name for logging.
        llm_semaphore_limit: Max concurrent LLM calls.
        seed: Optional decoding seed (passed through to the provider; only
            affects sampling at ``temperature > 0``).
        extra_config: Extra provider config kwargs forwarded to every LLM call
            (e.g. a Gemini ``thinking_config``).
        _llm: LLM instance for testing (dependency injection). If None, uses the
            global ``llm`` singleton.

    Raises:
        ValueError: On an unknown ``scoring_format`` or non-positive semaphore.
    """

    def __init__(
        self,
        llmc: Sequence[LLMChoice] | None = None,
        scoring_format: ScoringFormat = "5point",
        threshold: int | None = None,
        temperature: float = 0.0,
        model: str = "gemini-3-flash",
        name: str = "vanilla",
        llm_semaphore_limit: int = 5,
        seed: int | None = None,
        extra_config: Dict[str, Any] | None = None,
        _llm: Any = None,
    ) -> None:
        if scoring_format not in SCALES:
            raise ValueError(
                f"Unknown scoring_format {scoring_format!r}. Supported: {list(SCALES)}."
            )
        if llm_semaphore_limit <= 0:
            raise ValueError("llm_semaphore_limit must be > 0")

        self.scoring_format: ScoringFormat = scoring_format
        self.scale: ScaleSpec = SCALES[scoring_format]
        self.threshold: int = (
            threshold if threshold is not None else self.scale.default_threshold
        )
        self.temperature = temperature
        self.model = model
        self.name = name
        self.seed = seed
        self.extra_config: Dict[str, Any] = dict(extra_config or {})

        self.llmc: List[LLMChoice] = list(llmc) if llmc else [GoogleChoice(model=model)]
        self._system_prompt = build_system_prompt(scoring_format)
        self._llm_instance: Any = _llm if _llm is not None else llm
        self._llm_semaphore = asyncio.Semaphore(llm_semaphore_limit)
        self._token_counter = TokenCounter()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_preset(
        cls,
        preset: VanillaPreset = DEFAULT_PRESET,
        *,
        llmc: Sequence[LLMChoice] | None = None,
        **overrides: Any,
    ) -> "VanillaClassifier":
        """Build a classifier from a recommended preset.

        Args:
            preset: A :class:`VanillaPreset` (e.g. ``FLASH_5POINT``).
            llmc: Optional explicit LLM priority list; defaults to the preset's
                model.
            **overrides: Keyword overrides for any constructor argument
                (e.g. ``temperature=1.0``, ``threshold=3``).

        Returns:
            A configured :class:`VanillaClassifier`.
        """
        params: Dict[str, Any] = {
            "scoring_format": preset.scoring_format,
            "threshold": preset.threshold,
            "temperature": preset.temperature,
            "model": preset.model,
            "name": preset.name,
        }
        params.update(overrides)
        return cls(llmc=llmc, **params)

    @property
    def system_prompt(self) -> str:
        """The rendered system prompt for the configured scoring format."""
        return self._system_prompt

    @property
    def token_usage(self) -> TokenCounter:
        """Cumulative token usage across all calls made by this instance."""
        return self._token_counter

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    async def _score_one(self, index: Any, profile: str) -> Dict[str, Any]:
        """Score a single profile; never raises, returns a result row."""
        base = {
            "index": index,
            "score": None,
            "predicted": 0,
            "confidence": None,
            "reasoning": "",
            "key_signals": [],
            "status": "FAIL",
            "error": None,
        }
        try:
            query = USER_TEMPLATE.format(profile=profile)
            kwargs: Dict[str, Any] = dict(self.extra_config)
            if self.seed is not None:
                kwargs["seed"] = self.seed

            async with self._llm_semaphore:
                response = await self._llm_instance.respond(
                    llm_priority=self.llmc,
                    query=query,
                    instructions=self._system_prompt,
                    response_format=FounderScore,
                    temperature=self.temperature,
                    **kwargs,
                )
            await self._token_counter.append(
                provider=response.provider_model.provider,
                model=response.provider_model.model,
                value=response.total_tokens,
                caller="VanillaClassifier._score_one",
            )
            result = response.response
            if result is None:
                raise LLMError("No response from LLM")

            predicted = 1 if result.score >= self.threshold else 0
            base.update(
                score=result.score,
                predicted=predicted,
                confidence=result.confidence,
                reasoning=result.reasoning,
                key_signals=list(result.key_signals),
                status="OK",
            )
        except Exception as exc:  # noqa: BLE001 - per-row isolation by design
            logger.warning("Vanilla scoring failed for sample %s", index, exc_info=True)
            base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    async def score(self, X: pd.DataFrame, column: str = "data") -> pd.DataFrame:
        """Score every profile in ``X`` concurrently.

        Args:
            X: Dataframe of profiles.
            column: Name of the column holding the profile text.

        Returns:
            A dataframe aligned to ``X.index`` with columns ``score``,
            ``predicted``, ``confidence``, ``reasoning``, ``key_signals``,
            ``status`` and ``error``. Failed rows have ``status == "FAIL"`` and
            ``predicted == 0``.

        Raises:
            DataError: If ``column`` is missing from ``X``.
        """
        if column not in X.columns:
            raise DataError(
                f"Column {column!r} not found in X. Available: {list(X.columns)}."
            )

        tasks = [
            self._score_one(idx, str(profile)) for idx, profile in X[column].items()
        ]
        rows = await asyncio.gather(*tasks)

        result = pd.DataFrame(rows).set_index("index")
        result.index = X.index
        return result

    async def predict(self, X: pd.DataFrame, column: str = "data") -> List[int]:
        """Return 0/1 predictions for every profile in ``X``.

        Args:
            X: Dataframe of profiles.
            column: Name of the column holding the profile text.

        Returns:
            A list of 0/1 predictions aligned to ``X``'s row order.
        """
        scored = await self.score(X, column=column)
        return scored["predicted"].fillna(0).astype(int).tolist()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    async def evaluate(
        self,
        X: pd.DataFrame,
        y: Sequence[int],
        column: str = "data",
        beta: float = 0.5,
    ) -> VanillaMetrics:
        """Score ``X`` and compute metrics against labels ``y``.

        Args:
            X: Dataframe of profiles.
            y: Ground-truth 0/1 labels, aligned to ``X``'s row order.
            column: Name of the profile-text column.
            beta: Beta for the F-beta metric (default ``0.5``).

        Returns:
            The :class:`VanillaMetrics` at the configured threshold.

        Raises:
            DataError: If ``len(y) != len(X)``.
        """
        if len(y) != len(X):
            raise DataError(f"len(y)={len(y)} does not match len(X)={len(X)}.")
        scored = await self.score(X, column=column)
        y_pred = scored["predicted"].fillna(0).astype(int).tolist()
        return compute_metrics(list(y), y_pred, threshold=self.threshold, beta=beta)

    async def sweep_thresholds(
        self,
        X: pd.DataFrame,
        y: Sequence[int],
        thresholds: Sequence[int] | None = None,
        column: str = "data",
        beta: float = 0.5,
    ) -> pd.DataFrame:
        """Score once, then report metrics across candidate thresholds.

        This makes a single set of LLM calls and re-thresholds the raw scores,
        so it is the cheap way to choose a threshold on a validation fold.

        Args:
            X: Dataframe of profiles.
            y: Ground-truth 0/1 labels.
            thresholds: Candidate thresholds. Defaults to every non-minimal
                value of the configured scale (e.g. ``[2, 3, 4, 5]`` for
                5-point).
            column: Name of the profile-text column.
            beta: Beta for the F-beta metric.

        Returns:
            A dataframe (one row per threshold) with columns ``threshold``,
            ``precision``, ``recall``, ``f_beta``, ``tp``, ``fp``, ``fn`` and
            ``n_predicted_positive``, sorted by ``f_beta`` descending.

        Raises:
            DataError: If ``len(y) != len(X)``.
        """
        if len(y) != len(X):
            raise DataError(f"len(y)={len(y)} does not match len(X)={len(X)}.")
        if thresholds is None:
            thresholds = [v for v in self.scale.values if v > min(self.scale.values)]

        scored = await self.score(X, column=column)
        # Coerce to numeric first so failed rows (score=None) become NaN without
        # triggering the object-dtype downcast FutureWarning, then floor them
        # below every valid score so they never cross any threshold.
        floor = min(self.scale.values) - 1
        raw = pd.Series(
            pd.to_numeric(scored["score"], errors="coerce"), index=scored.index
        ).fillna(floor)
        y_true = list(y)

        rows = []
        for thr in thresholds:
            y_pred = (raw >= thr).astype(int).tolist()
            m = compute_metrics(y_true, y_pred, threshold=thr, beta=beta)
            rows.append(
                {
                    "threshold": thr,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f_beta": m.f_beta,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                    "n_predicted_positive": m.n_predicted_positive,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("f_beta", ascending=False)
            .reset_index(drop=True)
        )


__all__ = ["VanillaClassifier"]
