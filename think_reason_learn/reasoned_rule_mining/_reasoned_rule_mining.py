"""Reasoned Rule Mining.

An interpretable binary classifier that mines natural-language IF-THEN rules
from LLM reasoning over labelled examples, compiles them into a single decision
policy, predicts via a confidence-weighted ensemble vote (with an optional harsh
re-evaluation second stream), and fuses the streams with a Platt-calibrated,
recall-floor F-beta-optimised linear combiner.

This is a library port of the standalone VCBench reference implementation. The
algorithm is preserved; run-script scaffolding (CLI, CSV I/O, cost estimation,
plotting, and VCBench-specific data munging) is intentionally omitted -- the
caller supplies a generic ``pandas.DataFrame`` and YES/NO labels.

One deliberate fidelity note: the standalone read explicit top-2 HIGH/LOW token
logprobs from each call. The unified TRL LLM interface exposes the chosen
answer's ``average_confidence`` (``exp`` of its mean token logprob) rather than
per-class alternatives, so each vote's probability is derived from that
confidence. The ensemble/harsh/calibration/combiner mechanics are otherwise
faithful.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from copy import deepcopy
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Self, Sequence, Tuple
from uuid import uuid4

import numpy as np
import orjson
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.linear_model import LogisticRegression

from think_reason_learn.core.exceptions import DataError, LLMError
from think_reason_learn.core.llms import LLMChoice, TokenCounter, llm

from ._prompts import (
    HARSH_INSTRUCTIONS,
    MEMORY_SUMMARY_INSTRUCTIONS,
    POLICY_COMPILATION_INSTRUCTIONS,
    REASONING_INSTRUCTIONS,
    RULE_EXTRACTION_INSTRUCTIONS,
    VOTE_INSTRUCTIONS,
)
from ._types import ExtractedRule, RRMConfig, Vote

logger = logging.getLogger(__name__)

_RULE_OUTCOME_RE = re.compile(r"label\s*=\s*(YES|NO)", re.IGNORECASE)


class ReasonedRuleMining:
    """Interpretable rule-mining binary classifier.

    Args:
        reason_llmc: LLMs for reasoning, policy compilation, and memory
            summarisation, in priority order.
        extract_llmc: LLMs for rule extraction. Defaults to ``reason_llmc``.
        predict_llmc: LLMs for ensemble voting and harsh re-evaluation. Defaults
            to ``reason_llmc``.
        config: Algorithm configuration. Defaults to ``RRMConfig()``.
        reason_temperature: Sampling temperature for reasoning/policy/memory.
        predict_temperature: Sampling temperature for voting/harsh stages.
        llm_semaphore_limit: Max concurrent LLM calls.
        save_path: Directory for checkpoints/models.
        name: Instance name (alphanumeric/underscore).
        _llm: LLM instance for dependency injection (testing). If None, uses the
            global ``llm`` singleton.
    """

    def __init__(
        self,
        reason_llmc: List[LLMChoice],
        extract_llmc: List[LLMChoice] | None = None,
        predict_llmc: List[LLMChoice] | None = None,
        config: RRMConfig | None = None,
        reason_temperature: float = 1.0,
        predict_temperature: float = 0.0,
        llm_semaphore_limit: int = 3,
        save_path: str | PathLike[str] | None = None,
        name: str | None = None,
        _llm: Any = None,
    ) -> None:
        self._verify_input_data(
            reason_llmc=reason_llmc,
            reason_temperature=reason_temperature,
            predict_temperature=predict_temperature,
            llm_semaphore_limit=llm_semaphore_limit,
            save_path=save_path,
            name=name,
        )

        self.reason_llmc = reason_llmc
        self.extract_llmc = extract_llmc or reason_llmc
        self.predict_llmc = predict_llmc or reason_llmc
        self.config = config if config is not None else RRMConfig()
        self.reason_temperature = reason_temperature
        self.predict_temperature = predict_temperature

        self.name = self._get_name(name)
        self.save_path = self._set_save_path(save_path)

        self._token_counter = TokenCounter()
        self._llm_semaphore_limit = llm_semaphore_limit
        self._llm_semaphore = asyncio.Semaphore(llm_semaphore_limit)
        self._llm_instance: Any = _llm if _llm is not None else llm

        self._task_description: str | None = None
        self._X: pd.DataFrame | None = None
        self._y: np.ndarray | None = None

        # Learned state.
        self._reasoning_logs: List[str] = []
        self._memory: str = ""
        self._rules: List[ExtractedRule] = []
        self._policy: str | None = None
        self._ens_platt: LogisticRegression | None = None
        self._mod_platt: LogisticRegression | None = None
        self._alpha: float | None = None
        self._beta: float | None = None
        self._bias: float | None = None
        self._threshold: float | None = None
        self._validation_result: dict | None = None

    # ------------------------------------------------------------------
    # Validation / setup
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_input_data(**kwargs: Any) -> None:
        if not kwargs["reason_llmc"]:
            raise ValueError("reason_llmc must be a non-empty list")
        for key in ("reason_temperature", "predict_temperature"):
            val = kwargs[key]
            if not (0 <= val <= 2):
                raise ValueError(f"{key} must be >= 0 and <= 2")
        if kwargs["llm_semaphore_limit"] <= 0:
            raise ValueError("llm_semaphore_limit must be > 0")
        sp = kwargs["save_path"]
        if not (sp is None or isinstance(sp, (str, Path))):
            raise ValueError("save_path must be None, a string, or a Path")
        nm = kwargs["name"]
        if not (nm is None or isinstance(nm, str)):
            raise ValueError("name must be None or a string")

    @staticmethod
    def _get_name(name: str | None) -> str:
        if name is None:
            name = str(uuid4()).replace("-", "_")
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError("Name must be only alphanumeric and underscores")
        return name

    def _set_save_path(self, save_path: str | PathLike[str] | None) -> Path:
        if save_path is None:
            return (Path(os.getcwd()) / "reasoned_rule_mining" / self.name).resolve()
        path = Path(save_path).resolve()
        if path.is_file():
            raise ValueError("Please provide a directory, not a file.")
        return path

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def token_usage(self) -> TokenCounter:
        """Token counter accumulating usage across all LLM calls."""
        return self._token_counter

    @property
    def task_description(self) -> str | None:
        """The configured task description."""
        return self._task_description

    @property
    def policy(self) -> str:
        """The compiled decision policy.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._policy is None:
            raise ValueError("No policy. Call fit() first.")
        return self._policy

    @property
    def rules(self) -> List[ExtractedRule]:
        """The extracted IF-THEN rules (all, before perplexity filtering)."""
        return self._rules

    @property
    def threshold(self) -> float:
        """The decision threshold on the combiner score.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._threshold is None:
            raise ValueError("No threshold. Call fit() first.")
        return self._threshold

    @property
    def weights(self) -> Tuple[float, float, float]:
        """The fitted combiner weights ``(alpha, beta, bias)``.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._alpha is None or self._beta is None or self._bias is None:
            raise ValueError("No weights. Call fit() first.")
        return (self._alpha, self._beta, self._bias)

    @property
    def validation_result(self) -> dict:
        """Diagnostics from weight optimisation.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._validation_result is None:
            raise ValueError("No validation result. Call fit() first.")
        return self._validation_result

    @property
    def llm_semaphore_limit(self) -> int:
        """Max concurrent LLM calls."""
        return self._llm_semaphore_limit

    @llm_semaphore_limit.setter
    def llm_semaphore_limit(self, value: int) -> None:
        if value <= 0:
            raise ValueError("llm_semaphore_limit must be > 0")
        self._llm_semaphore_limit = value
        self._llm_semaphore = asyncio.Semaphore(value)

    async def set_task(self, task_description: str) -> None:
        """Set the binary-classification task description.

        The description is interpolated into every prompt, so it should describe
        what YES and NO mean for the task.

        Args:
            task_description: A description of the binary classification task.

        Raises:
            ValueError: If ``task_description`` is empty.
        """
        if not task_description or not task_description.strip():
            raise ValueError("task_description must be a non-empty string")
        self._task_description = task_description

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------

    def _set_data(self, X: pd.DataFrame, y: Sequence[str], copy_data: bool) -> None:
        if not isinstance(X, pd.DataFrame):
            raise DataError("X must be a pandas DataFrame")
        if X.shape[0] == 0:
            raise DataError("X must have at least one row")
        if len(y) != X.shape[0]:
            raise DataError("y and X must have the same number of rows")
        if not all(isinstance(item, str) for item in y):
            raise DataError("y must be a sequence of strings")
        y_upper = np.array([str(yi).upper() for yi in y], dtype=np.str_)
        if set(np.unique(y_upper)) - {"YES", "NO"}:
            raise DataError("y must contain only 'YES' or 'NO' values")
        if set(np.unique(y_upper)) != {"YES", "NO"}:
            raise DataError("y must contain both 'YES' and 'NO' values to fit")

        if copy_data:
            self._X = deepcopy(X).reset_index(drop=True)
            self._y = y_upper.copy()
        else:
            self._X = X.reset_index(drop=True)
            self._y = y_upper

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        return "\n".join(f"{col}: {val}" for col, val in row.items())

    def _sample_texts(self, X: pd.DataFrame) -> List[str]:
        return [self._row_to_text(row) for _, row in X.iterrows()]

    # ------------------------------------------------------------------
    # Low-level LLM call
    # ------------------------------------------------------------------

    async def _respond(
        self,
        query: str,
        llm_priority: List[LLMChoice],
        response_format: Any,
        instructions: str,
        temperature: float,
        caller: str,
    ) -> Any:
        async with self._llm_semaphore:
            response = await self._llm_instance.respond(
                query=query,
                llm_priority=llm_priority,
                response_format=response_format,
                instructions=instructions,
                temperature=temperature,
            )
        await self._token_counter.append(
            provider=response.provider_model.provider,
            model=response.provider_model.model,
            value=response.total_tokens,
            caller=caller,
        )
        return response

    # ------------------------------------------------------------------
    # Stage 1: reasoning logs (+ rolling memory)
    # ------------------------------------------------------------------

    async def _reason_one(self, text: str, label: str, memory: str) -> str:
        mem_block = f"Notes so far:\n{memory}\n\n" if memory else ""
        query = (
            f"Task:\n{self._task_description}\n\n"
            f"{mem_block}"
            f"Sample:\n{text}\n\n"
            f"Known label: {label}"
        )
        response = await self._respond(
            query=query,
            llm_priority=self.reason_llmc,
            response_format=str,
            instructions=REASONING_INSTRUCTIONS,
            temperature=self.reason_temperature,
            caller="ReasonedRuleMining._reason_one",
        )
        return (response.response or "").strip()

    async def _update_memory(self, memory: str, new_logs: List[str]) -> str:
        query = (
            f"Task:\n{self._task_description}\n\n"
            f"Previous summary:\n{memory or '(none)'}\n\n"
            "New reasoning logs:\n" + "\n---\n".join(new_logs)
        )
        response = await self._respond(
            query=query,
            llm_priority=self.reason_llmc,
            response_format=str,
            instructions=MEMORY_SUMMARY_INSTRUCTIONS,
            temperature=self.reason_temperature,
            caller="ReasonedRuleMining._update_memory",
        )
        return (response.response or memory).strip()

    async def _generate_reasoning_logs(self) -> None:
        assert self._X is not None and self._y is not None
        texts = self._sample_texts(self._X)
        labels = list(self._y)
        n = len(texts)
        logs: List[str] = [""] * n
        memory = ""
        interval = self.config.memory_update_interval if self.config.use_memory else n

        for start in range(0, n, max(interval, 1)):
            stop = min(start + max(interval, 1), n)
            chunk = await asyncio.gather(
                *(
                    self._reason_one(texts[i], labels[i], memory)
                    for i in range(start, stop)
                )
            )
            for offset, log in enumerate(chunk):
                logs[start + offset] = log
            if self.config.use_memory and stop < n:
                memory = await self._update_memory(memory, list(chunk))

        self._reasoning_logs = logs
        self._memory = memory

    # ------------------------------------------------------------------
    # Stage 2: rule extraction (+ perplexity)
    # ------------------------------------------------------------------

    @staticmethod
    def _perplexity(logprobs: List[Tuple[str, float | None]]) -> float:
        vals = [lp for _, lp in logprobs if lp is not None]
        if not vals:
            return 1000.0
        return round(math.exp(-(sum(vals) / len(vals))), 4)

    @staticmethod
    def _align_outcome(rule_text: str, outcome: str) -> str:
        rule_text = rule_text.strip()
        if _RULE_OUTCOME_RE.search(rule_text):
            return _RULE_OUTCOME_RE.sub(f"label = {outcome}", rule_text, count=1)
        return f"{rule_text} THEN label = {outcome}"

    async def _extract_one(self, log: str, outcome: str) -> ExtractedRule:
        query = (
            f"Task:\n{self._task_description}\n\n"
            f"The sample's actual label is: {outcome}.\n\n"
            f"Reasoning Log:\n{log}"
        )
        response = await self._respond(
            query=query,
            llm_priority=self.extract_llmc,
            response_format=str,
            instructions=RULE_EXTRACTION_INSTRUCTIONS,
            temperature=self.predict_temperature,
            caller="ReasonedRuleMining._extract_one",
        )
        rule_text = (response.response or "").strip()
        if not rule_text:
            rule_text = "IF [conditions] THEN label = " + outcome
            perplexity = 1000.0
        else:
            perplexity = self._perplexity(response.logprobs)
        return ExtractedRule(
            rule=self._align_outcome(rule_text, outcome),
            outcome=outcome,  # type: ignore[arg-type]
            perplexity=perplexity,
        )

    async def _extract_rules(self) -> None:
        assert self._y is not None
        outcomes = [str(yi).upper() for yi in self._y]
        rules = await asyncio.gather(
            *(
                self._extract_one(log, outcome)
                for log, outcome in zip(self._reasoning_logs, outcomes)
            )
        )
        self._rules = list(rules)

    # ------------------------------------------------------------------
    # Stage 3: policy compilation
    # ------------------------------------------------------------------

    async def _compile_policy(self) -> None:
        kept = [
            r for r in self._rules if r.perplexity <= self.config.perplexity_threshold
        ]
        logger.info("Policy compilation: %d/%d rules kept", len(kept), len(self._rules))
        yes_rules = [r.rule for r in kept if r.outcome == "YES"]
        no_rules = [r.rule for r in kept if r.outcome == "NO"]
        query = (
            f"Task:\n{self._task_description}\n\n"
            "Rules for the YES class:\n" + ("\n".join(yes_rules) or "(none)") + "\n\n"
            "Rules for the NO class:\n" + ("\n".join(no_rules) or "(none)")
        )
        response = await self._respond(
            query=query,
            llm_priority=self.reason_llmc,
            response_format=str,
            instructions=POLICY_COMPILATION_INSTRUCTIONS,
            temperature=self.reason_temperature,
            caller="ReasonedRuleMining._compile_policy",
        )
        policy = (response.response or "").strip()
        if not policy:
            raise LLMError("Failed to compile a decision policy. Try another LLM.")
        self._policy = policy

    # ------------------------------------------------------------------
    # Stage 4 & 5: ensemble voting + harsh re-evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _vote_high_prob(response: Any) -> Tuple[str, float]:
        """Map a Vote response to (label, P(YES)) using answer confidence."""
        label = str(response.response.vote).strip().upper()
        conf = response.average_confidence
        if conf is None:
            conf = 0.5
        conf = float(min(max(conf, 0.0), 1.0))
        p_yes = conf if label == "YES" else 1.0 - conf
        return label, p_yes

    async def _single_vote(
        self, text: str, instructions: str, caller: str
    ) -> Tuple[str, float]:
        query = (
            f"Task:\n{self._task_description}\n\n"
            f"Decision Policy:\n{self._policy}\n\n"
            f"Sample:\n{text}"
        )
        response = await self._respond(
            query=query,
            llm_priority=self.predict_llmc,
            response_format=Vote,
            instructions=instructions,
            temperature=self.predict_temperature,
            caller=caller,
        )
        return self._vote_high_prob(response)

    async def _ensemble_vote(self, text: str) -> Tuple[float, str]:
        """Return (ensemble P(YES), tentative label) for one sample."""
        results = await asyncio.gather(
            *(
                self._single_vote(
                    text, VOTE_INSTRUCTIONS, "ReasonedRuleMining._ensemble_vote"
                )
                for _ in range(self.config.ensemble_size)
            )
        )
        cfg = self.config
        p_yes = float(np.mean([p for _, p in results]))
        yes_votes = sum(1 for label, _ in results if label == "YES")

        label = "YES" if p_yes > 0.5 else "NO"
        if label == "YES" and p_yes < cfg.ensemble_confidence_threshold:
            label = "NO"
        if yes_votes < cfg.ensemble_size * cfg.precision_voting_threshold:
            label = "NO"
        if cfg.enable_final_confidence_check and p_yes < cfg.final_confidence_threshold:
            label = "NO"
        return p_yes, label

    async def _harsh_vote(self, text: str) -> float:
        """Return harsh P(YES) for one sample (the second stream)."""
        instructions = HARSH_INSTRUCTIONS[self.config.harsh_level]
        results = await asyncio.gather(
            *(
                self._single_vote(text, instructions, "ReasonedRuleMining._harsh_vote")
                for _ in range(self.config.ensemble_size)
            )
        )
        return float(np.mean([p for _, p in results]))

    # ------------------------------------------------------------------
    # Calibration + combiner (numeric, pure)
    # ------------------------------------------------------------------

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p) - np.log(1 - p)

    def _platt_fit(self, p: np.ndarray, y: np.ndarray) -> LogisticRegression:
        feats = self._logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        lr = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=self.config.random_state
        )
        lr.fit(feats, y.astype(int))
        return lr

    def _platt_apply(self, lr: LogisticRegression, p: np.ndarray) -> np.ndarray:
        feats = self._logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        return np.clip(lr.predict_proba(feats)[:, 1], 1e-6, 1 - 1e-6)

    def _combiner_score(
        self,
        ens_cal: np.ndarray,
        mod_cal: np.ndarray,
        alpha: float,
        beta: float,
        bias: float,
    ) -> np.ndarray:
        s = np.full(len(ens_cal), float(bias), dtype=np.float64)
        if alpha != 0.0:
            l1 = self._logit(ens_cal).astype(np.float64, copy=False)
            s += alpha * np.where(np.isfinite(l1), l1, 0.0)
        if beta != 0.0 and not np.all(np.isnan(mod_cal)):
            l2 = self._logit(mod_cal).astype(np.float64, copy=False)
            s += beta * np.where(np.isfinite(l2), l2, 0.0)
        return s

    def _best_fbeta_at_recall(
        self, scores: np.ndarray, y: np.ndarray, min_recall: float
    ) -> Tuple[float, float, float, float] | None:
        """Return (fbeta, threshold, precision, recall) maximising F-beta.

        Subject to the constraint ``recall >= min_recall``. Returns ``None`` when
        no threshold satisfies the floor (or scores are degenerate).
        """
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        if s_min == s_max:
            return None
        pos = int(y.sum())
        if pos == 0:
            return None
        beta2 = self.config.beta**2
        min_tp = int(np.ceil(min_recall * pos))
        thresholds = np.linspace(s_min, s_max, self.config.weights_thresh_points)
        best: Tuple[float, float, float, float] | None = None
        best_f = -1.0
        for thr in thresholds:
            preds = scores >= thr
            if not preds.any():
                continue
            tp = int(np.sum(preds & (y == 1)))
            if tp < min_tp:
                continue
            fp = int(np.sum(preds & (y == 0)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / pos
            if recall < min_recall or precision <= 0:
                continue
            denom = beta2 * precision + recall
            if denom <= 0:
                continue
            fbeta = (1 + beta2) * (precision * recall) / denom
            if fbeta > best_f:
                best_f = fbeta
                best = (fbeta, float(thr), precision, recall)
        return best

    def _optimize_weights(
        self, ens_cal: np.ndarray, mod_cal: np.ndarray, y: np.ndarray, min_recall: float
    ) -> dict | None:
        cfg = self.config
        alphas = np.linspace(*cfg.weights_alpha_range, cfg.weights_n_points)
        betas = np.linspace(*cfg.weights_beta_range, cfg.weights_n_points)
        biases = np.linspace(*cfg.weights_bias_range, cfg.weights_n_points)
        has_mod = not np.all(np.isnan(mod_cal))
        best: dict | None = None
        for a in alphas:
            for b in betas if has_mod else [0.0]:
                for bi in biases:
                    scores = self._combiner_score(ens_cal, mod_cal, a, b, bi)
                    if not np.isfinite(scores).all():
                        continue
                    res = self._best_fbeta_at_recall(scores, y, min_recall)
                    if res is None:
                        continue
                    if best is None or res[0] > best["fbeta"]:
                        best = {
                            "fbeta": res[0],
                            "threshold": res[1],
                            "precision": res[2],
                            "recall": res[3],
                            "alpha": float(a),
                            "beta": float(b),
                            "bias": float(bi),
                            "min_recall": float(min_recall),
                        }
        return best

    def _calibrate_and_optimize(
        self,
        ensemble_prob: np.ndarray,
        moderate_prob: np.ndarray,
        reviewed: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self._ens_platt = self._platt_fit(ensemble_prob, y)
        ens_cal = self._platt_apply(self._ens_platt, ensemble_prob)

        mod_cal = np.full(len(ensemble_prob), np.nan, dtype=np.float64)
        mask = reviewed == 1
        self._mod_platt = None
        if mask.sum() > 0:
            if np.unique(y[mask]).size >= 2:
                self._mod_platt = self._platt_fit(moderate_prob[mask], y[mask])
                mod_cal[mask] = self._platt_apply(self._mod_platt, moderate_prob[mask])
            else:
                mod_cal[mask] = np.clip(moderate_prob[mask], 1e-6, 1 - 1e-6)

        best: dict | None = None
        per_floor: List[dict] = []
        for floor in self.config.recall_floors:
            res = self._optimize_weights(ens_cal, mod_cal, y, floor)
            if res is not None:
                per_floor.append(res)
                if best is None or res["fbeta"] > best["fbeta"]:
                    best = res
        if best is None:
            raise DataError(
                "Weight optimisation found no solution satisfying any recall "
                "floor. Check that training data has both classes and that the "
                "ensemble produced varied probabilities."
            )

        self._alpha = best["alpha"]
        self._beta = best["beta"]
        self._bias = best["bias"]
        self._threshold = best["threshold"]
        self._validation_result = {
            "selected": best,
            "per_floor": per_floor,
            "beta": self.config.beta,
            "n_reviewed": int(mask.sum()),
        }
        logger.info(
            "Combiner fitted: alpha=%.3f beta=%.3f bias=%.3f tau=%.3f (F%.2f=%.4f)",
            self._alpha,
            self._beta,
            self._bias,
            self._threshold,
            self.config.beta,
            best["fbeta"],
        )

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    async def _vote_all(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ens = await asyncio.gather(*(self._ensemble_vote(t) for t in texts))
        ensemble_prob = np.array([p for p, _ in ens], dtype=np.float64)
        reviewed = np.array([1 if label == "YES" else 0 for _, label in ens], dtype=int)

        moderate_prob = np.full(len(texts), np.nan, dtype=np.float64)
        if self.config.use_harsh:
            idxs = [i for i, r in enumerate(reviewed) if r == 1]
            if idxs:
                harsh = await asyncio.gather(
                    *(self._harsh_vote(texts[i]) for i in idxs)
                )
                for i, hp in zip(idxs, harsh):
                    moderate_prob[i] = hp
        return ensemble_prob, moderate_prob, reviewed

    async def fit(
        self,
        X: pd.DataFrame | None = None,
        y: Sequence[str] | None = None,
        *,
        copy_data: bool = True,
        reset: bool = False,
    ) -> Self:
        """Fit the model: mine rules, compile a policy, and tune the combiner.

        Args:
            X: Training features (a DataFrame; each row is stringified).
            y: Training labels, each "YES" or "NO" (case-insensitive).
            copy_data: Whether to copy input data.
            reset: Clear learned state and refit from scratch.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If ``set_task`` has not been called, or data is missing.
            DataError: If the data is invalid.
        """
        if self._task_description is None:
            raise ValueError("Call set_task(...) before fit().")

        if reset or X is not None or y is not None:
            if X is None or y is None:
                raise ValueError("Both X and y must be provided to (re)fit.")
            self._set_data(X, y, copy_data)
        if self._X is None or self._y is None:
            raise ValueError("No data found. Provide X and y.")

        logger.info("Stage 1/4: generating reasoning logs")
        await self._generate_reasoning_logs()
        logger.info("Stage 2/4: extracting rules")
        await self._extract_rules()
        logger.info("Stage 3/4: compiling decision policy")
        await self._compile_policy()
        logger.info("Stage 4/4: ensemble voting + calibration")
        texts = self._sample_texts(self._X)
        ensemble_prob, moderate_prob, reviewed = await self._vote_all(texts)
        y_bin = (self._y == "YES").astype(int)
        self._calibrate_and_optimize(ensemble_prob, moderate_prob, reviewed, y_bin)
        logger.info("ReasonedRuleMining fitted successfully")
        return self

    def _check_fitted(self) -> None:
        if (
            self._policy is None
            or self._alpha is None
            or self._beta is None
            or self._bias is None
            or self._threshold is None
        ):
            raise RuntimeError("Model not fitted. Call fit() before predict().")

    async def _predict_one(self, text: str) -> Tuple[float, str, Dict[str, float]]:
        assert (
            self._alpha is not None
            and self._beta is not None
            and self._bias is not None
            and self._threshold is not None
        )
        ensemble_prob, tentative = await self._ensemble_vote(text)
        moderate_prob = math.nan
        if self.config.use_harsh and tentative == "YES":
            moderate_prob = await self._harsh_vote(text)

        ens_cal = float(
            self._platt_apply(self._ens_platt, np.array([ensemble_prob]))[0]
            if self._ens_platt is not None
            else ensemble_prob
        )
        mod_cal = math.nan
        if not math.isnan(moderate_prob):
            if self._mod_platt is not None:
                mod_cal = float(
                    self._platt_apply(self._mod_platt, np.array([moderate_prob]))[0]
                )
            else:
                mod_cal = float(np.clip(moderate_prob, 1e-6, 1 - 1e-6))

        score = float(
            self._combiner_score(
                np.array([ens_cal]),
                np.array([mod_cal]),
                self._alpha,
                self._beta,
                self._bias,
            )[0]
        )
        label = "YES" if score >= self._threshold else "NO"
        scores = {
            "ensemble_prob": ensemble_prob,
            "moderate_prob": moderate_prob,
            "score": score,
        }
        return score, label, scores

    async def predict(
        self, samples: pd.DataFrame
    ) -> AsyncGenerator[
        Tuple[Any, Dict[str, float], Literal["YES", "NO"], TokenCounter], None
    ]:
        """Predict labels for samples.

        Args:
            samples: Samples to predict (a DataFrame; each row is stringified).

        Yields:
            Tuples of ``(sample_index, scores, label, token_counter)`` where
            ``scores`` holds the ensemble/harsh probabilities and the fused
            combiner ``score``, and ``label`` is "YES" or "NO".

        Raises:
            RuntimeError: If the model has not been fitted.
            DataError: If ``samples`` is not a non-empty DataFrame.
        """
        self._check_fitted()
        if not isinstance(samples, pd.DataFrame) or samples.shape[0] == 0:
            raise DataError("samples must be a non-empty DataFrame")

        indices = list(samples.index)
        texts = self._sample_texts(samples)
        results = await asyncio.gather(*(self._predict_one(t) for t in texts))
        for idx, (_, label, scores) in zip(indices, results):
            yield idx, scores, label, self._token_counter  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: str | PathLike[str] | None = None) -> None:
        """Persist the fitted model to a directory.

        Writes ``reasoned_rule_mining.json`` (config + learned state) plus
        ``ens_platt.joblib`` / ``mod_platt.joblib`` for the Platt scalers.

        Args:
            dir_path: Target directory. Defaults to ``self.save_path``.

        Raises:
            ValueError: If ``dir_path`` is an existing file.
        """
        base = Path(dir_path) if dir_path is not None else self.save_path
        if base.is_file():
            raise ValueError("Please provide a directory, not a file.")
        base.mkdir(parents=True, exist_ok=True)

        ens_file = mod_file = None
        if self._ens_platt is not None:
            ens_file = "ens_platt.joblib"
            joblib_dump(self._ens_platt, base / ens_file)
        if self._mod_platt is not None:
            mod_file = "mod_platt.joblib"
            joblib_dump(self._mod_platt, base / mod_file)

        manifest = {
            "name": self.name,
            "reason_llmc": self._dump_llmc(self.reason_llmc),
            "extract_llmc": self._dump_llmc(self.extract_llmc),
            "predict_llmc": self._dump_llmc(self.predict_llmc),
            "reason_temperature": self.reason_temperature,
            "predict_temperature": self.predict_temperature,
            "llm_semaphore_limit": self._llm_semaphore_limit,
            "config": asdict(self.config),
            "task_description": self._task_description,
            "policy": self._policy,
            "rules": [r.model_dump() for r in self._rules],
            "memory": self._memory,
            "alpha": self._alpha,
            "beta": self._beta,
            "bias": self._bias,
            "threshold": self._threshold,
            "validation_result": self._validation_result,
            "ens_platt_file": ens_file,
            "mod_platt_file": mod_file,
            "token_counter": self._token_counter.to_dict(),
            "version": 1,
        }
        with (base / "reasoned_rule_mining.json").open("wb") as f:
            f.write(orjson.dumps(manifest, option=orjson.OPT_SERIALIZE_NUMPY))

    @staticmethod
    def _dump_llmc(llmc: List[LLMChoice]) -> List[Any]:
        return [c if isinstance(c, dict) else c.model_dump() for c in llmc]

    @classmethod
    def load(cls, dir_path: str | PathLike[str]) -> "ReasonedRuleMining":
        """Load a model previously saved with :meth:`save`.

        Args:
            dir_path: Directory containing ``reasoned_rule_mining.json``.

        Returns:
            The restored instance.

        Raises:
            FileNotFoundError: If the manifest is missing.
        """
        base = Path(dir_path)
        manifest_path = base / "reasoned_rule_mining.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found in directory: {base}")
        with manifest_path.open("rb") as f:
            manifest = orjson.loads(f.read())

        inst = cls(
            reason_llmc=manifest["reason_llmc"],
            extract_llmc=manifest["extract_llmc"],
            predict_llmc=manifest["predict_llmc"],
            config=RRMConfig(**manifest["config"]),
            reason_temperature=manifest["reason_temperature"],
            predict_temperature=manifest["predict_temperature"],
            llm_semaphore_limit=manifest["llm_semaphore_limit"],
            save_path=str(base.parent),
            name=manifest["name"],
        )
        inst._task_description = manifest.get("task_description")
        inst._policy = manifest.get("policy")
        inst._rules = [ExtractedRule(**r) for r in manifest.get("rules", [])]
        inst._memory = manifest.get("memory", "")
        inst._alpha = manifest.get("alpha")
        inst._beta = manifest.get("beta")
        inst._bias = manifest.get("bias")
        inst._threshold = manifest.get("threshold")
        inst._validation_result = manifest.get("validation_result")
        tk = manifest.get("token_counter")
        if tk:
            inst._token_counter = TokenCounter.from_dict(tk)
        if manifest.get("ens_platt_file"):
            inst._ens_platt = joblib_load(base / manifest["ens_platt_file"])
        if manifest.get("mod_platt_file"):
            inst._mod_platt = joblib_load(base / manifest["mod_platt_file"])
        return inst

    def __repr__(self) -> str:
        """Return a concise representation."""
        return f"ReasonedRuleMining(name={self.name})"
