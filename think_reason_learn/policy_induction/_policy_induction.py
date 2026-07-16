"""Policy Induction.

An interpretable ensemble binary classifier using LLM-induced policies
weighted by logistic regression.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Self,
    Sequence,
    Tuple,
    Union,
    cast,
)
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import orjson
import pandas as pd
from joblib import dump as joblib_dump
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from think_reason_learn.core.exceptions import DataError, LLMError
from think_reason_learn.core.llms import LLMChoice, TokenCounter, llm
from ._prompts import (
    POLICY_GEN_INSTRUCTIONS,
    POLICY_PREDICT_INSTRUCTIONS,
    max_policy_num_tag,
)

logger = logging.getLogger(__name__)

# Compact terminal-style progress bar: "[TAG] ████░░░░ n/total · rate/s · eta MM:SS"
_BAR_FORMAT = "{desc} {bar} {n_fmt}/{total_fmt} · {rate_fmt} · eta {remaining}"
_BAR_ASCII = "░█"


# ── Schemas ────────────────────────────────────────────────────────────────────


class Policies(BaseModel):
    policies: List[str] = Field(..., description="The list of generated policies.")


class Answer(BaseModel):
    answer: Literal["YES", "NO"]


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class WeightTrainerConfig:
    """Configuration for training and optimizing ensemble weights.

    Args:
        beta: Beta for F-beta score (e.g. 0.5 weights precision more).
        penalty: Regularization type for logistic regression.
        cv_folds: Number of StratifiedKFold splits.
        Cs: Candidate regularization strengths.
        threshold_grid: Decision thresholds to search over CV folds.
        class_weight_balanced: Use class_weight='balanced' in LR.
        random_state: Random seed for CV splits.
    """

    beta: float = 0.5
    penalty: Literal["l1", "l2"] = "l1"
    cv_folds: int = 5
    Cs: Iterable[float] = (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)
    threshold_grid: Iterable[float] = tuple(np.linspace(0.01, 0.99, 99))
    class_weight_balanced: bool = False
    random_state: int = 0


# ── PolicyInduction ────────────────────────────────────────────────────────────


class PolicyInduction:
    """Interpretable ensemble binary classifier.

    Induces natural-language policies from labeled data via LLM, scores each
    policy against every sample, then trains a logistic regression to find the
    optimal weighted combination for YES/NO prediction.

    Args:
        gen_llmc: LLMs for policy generation, in priority order.
        predict_llmc: LLMs for prediction. Defaults to gen_llmc.
        config: Weight training configuration.
        gen_temperature: Sampling temperature for generation.
        predict_temperature: Sampling temperature for prediction.
        llm_semaphore_limit: Max concurrent LLM calls.
        max_policy_length: Max total policies to induce (< 500).
        class_ratio: Target YES/NO mix per generation batch, not the dataset's
            actual ratio. Generation stops once either class can no longer
            fill its share, so imbalanced datasets won't have every
            majority-class row shown during generation.
        max_samples_as_context: Samples per generation batch (max 100).
        p_predict_update_interval: Log progress every N policies during scoring.
        save_path: Directory for checkpoints and saved models.
        name: Instance name (alphanumeric + underscores only).
        random_state: Base random seed.
        confirm_requests: Before fit()/predict() make any LLM calls, print an
            estimated request count per model and require a y/n confirmation.
    """

    def __init__(
        self,
        gen_llmc: List[LLMChoice],
        predict_llmc: List[LLMChoice] | None = None,
        config: WeightTrainerConfig | dict | None = None,
        gen_temperature: float = 1.0,
        predict_temperature: float = 0.0,
        llm_semaphore_limit: int = 3,
        max_policy_length: int = 20,
        class_ratio: Tuple[float, float] = (1.0, 1.0),
        max_samples_as_context: int = 10,
        p_predict_update_interval: int = 10,
        save_path: str | PathLike[str] | None = None,
        name: str | None = None,
        random_state: int = 0,
        confirm_requests: bool = True,
    ):
        self._validate_init(
            max_policy_length=max_policy_length,
            class_ratio=class_ratio,
            llm_semaphore_limit=llm_semaphore_limit,
            p_predict_update_interval=p_predict_update_interval,
            save_path=save_path,
            name=name,
            gen_temperature=gen_temperature,
            predict_temperature=predict_temperature,
        )
        self.gen_llmc = gen_llmc
        self.predict_llmc = predict_llmc or gen_llmc
        self.gen_temperature = gen_temperature
        self.predict_temperature = predict_temperature
        self._llm_semaphore_limit = llm_semaphore_limit
        self.config: WeightTrainerConfig = self._parse_config(config)
        self.class_ratio = class_ratio
        self.max_policy_length = max_policy_length
        self.random_state = random_state
        self.max_samples_as_context = max_samples_as_context
        self.p_predict_update_interval = p_predict_update_interval
        self.confirm_requests = confirm_requests
        self.name: str = self._parse_name(name)
        self.save_path: Path = self._parse_save_path(save_path)

        self._token_counter: TokenCounter = TokenCounter()
        self._llm_semaphore = asyncio.Semaphore(llm_semaphore_limit)
        self._pgen_instructions_template: str | None = None
        self._task_description: str | None = None

        self._X: pd.DataFrame | None = None
        self._y: npt.NDArray[np.str_] | None = None
        self._policy_memory: pd.DataFrame = self._empty_memory()
        self._threshold: float = 0.0
        self._lr: LogisticRegression | None = None
        self._validation_result: dict | None = None
        self._fit_duration_seconds: float | None = None
        self._fit_completed_at: str | None = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("Threshold must be numeric.")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("Threshold must be between 0 and 1.")
        self._threshold = float(value)

    @property
    def lr(self) -> LogisticRegression:
        if self._lr is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._lr

    @property
    def validation_result(self) -> dict:
        if self._validation_result is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._validation_result

    @property
    def llm_semaphore_limit(self) -> int:
        return self._llm_semaphore_limit

    @llm_semaphore_limit.setter
    def llm_semaphore_limit(self, value: int) -> None:
        self._llm_semaphore_limit = value
        self._llm_semaphore = asyncio.Semaphore(value)

    @property
    def token_usage(self) -> TokenCounter:
        return self._token_counter

    @property
    def task_description(self) -> str | None:
        return self._task_description

    @property
    def policy_gen_instructions_template(self) -> str | None:
        return self._pgen_instructions_template

    # ── Validation helpers ──────────────────────────────────────────────────────

    def _validate_init(self, **kw: Any) -> None:
        if not (0 < kw["max_policy_length"] < 500):
            raise ValueError("max_policy_length must be > 0 and < 500")
        if not (len(kw["class_ratio"]) == 2 and all(v > 0 for v in kw["class_ratio"])):
            raise ValueError("class_ratio must be two positive floats")
        if kw["llm_semaphore_limit"] <= 0:
            raise ValueError("llm_semaphore_limit must be > 0")
        if kw["p_predict_update_interval"] <= 0:
            raise ValueError("p_predict_update_interval must be > 0")
        if not (kw["save_path"] is None or isinstance(kw["save_path"], (str, Path))):
            raise ValueError("save_path must be None, str, or Path")
        if not (kw["name"] is None or isinstance(kw["name"], str)):
            raise ValueError("name must be None or str")
        if not (0 <= kw["gen_temperature"] <= 2):
            raise ValueError("gen_temperature must be in [0, 2]")
        if not (0 <= kw["predict_temperature"] <= 2):
            raise ValueError("predict_temperature must be in [0, 2]")

    def _parse_name(self, name: str | None) -> str:
        if name is None:
            name = str(uuid4()).replace("-", "_")
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError("Name must be alphanumeric and underscores only")
        return name

    def _parse_config(
        self, cfg: Union[None, WeightTrainerConfig, dict]
    ) -> WeightTrainerConfig:
        if cfg is None:
            return WeightTrainerConfig()
        if isinstance(cfg, WeightTrainerConfig):
            return cfg
        if isinstance(cfg, dict):
            try:
                result = WeightTrainerConfig(**cfg)
            except TypeError as e:
                raise ValueError(f"Invalid config keys: {e}")
            if result.penalty not in ("l1", "l2"):
                raise ValueError("penalty must be 'l1' or 'l2'")
            if result.beta <= 0:
                raise ValueError("beta must be positive")
            return result
        raise ValueError("config must be WeightTrainerConfig, dict, or None")

    def _parse_save_path(self, save_path: str | PathLike[str] | None) -> Path:
        if save_path is None:
            return (Path(os.getcwd()) / "policy_induction" / self.name).resolve()
        p = Path(save_path).resolve()
        if p.is_file():
            raise ValueError("save_path must be a directory, not a file")
        return p

    # ── Data management ─────────────────────────────────────────────────────────

    def _empty_memory(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "policy": pd.Series([], dtype=str),
                "predictions": pd.Series([], dtype=object),
            }
        )

    def _set_data(self, X: pd.DataFrame, y: Sequence[str]) -> None:
        if not all(isinstance(v, str) for v in y):
            raise DataError("y must be a sequence of strings")
        if len(y) != X.shape[0]:
            raise DataError("X and y must have the same number of rows")
        if set(np.unique(y)) != {"YES", "NO"}:
            raise DataError("y must contain only 'YES' or 'NO'")
        y_arr = np.array([v.upper() for v in y], dtype=np.str_)
        self._X = deepcopy(X).reset_index(drop=True)
        self._y = deepcopy(y_arr)

    def _sample(
        self, n: int, seed: int | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        """Yield balanced batches of n samples, stopping once either class runs out.

        Each batch draws exactly the class_ratio-determined share of YES/NO
        samples. Generation stops as soon as one class can no longer fill its
        share, rather than topping batches up from the other class — on an
        imbalanced dataset, this means not every majority-class row is shown
        during generation (scoring and weight fitting still use all of them).
        """
        rng = np.random.default_rng(seed if seed is not None else self.random_state)
        yes_idx = rng.permutation(np.where(self._y == "YES")[0])
        no_idx = rng.permutation(np.where(self._y == "NO")[0])

        p_yes = self.class_ratio[0] / sum(self.class_ratio)
        want_yes = int(round(n * p_yes))
        want_no = n - want_yes

        taken_y = taken_n = 0
        len_yes, len_no = len(yes_idx), len(no_idx)

        while True:
            take_y = min(want_yes, len_yes - taken_y)
            take_n = min(want_no, len_no - taken_n)
            if take_y < want_yes or take_n < want_no:
                break

            batch = np.concatenate(
                [
                    yes_idx[taken_y : taken_y + take_y],
                    no_idx[taken_n : taken_n + take_n],
                ]
            )
            rng.shuffle(batch)
            df = self._X.iloc[batch].copy()  # type: ignore
            df["y"] = self._y[batch]  # type: ignore
            yield df

            taken_y += take_y
            taken_n += take_n

    # ── Task / instructions ─────────────────────────────────────────────────────

    async def set_task(
        self,
        task_description: str,
        instructions_template: str | None = None,
    ) -> str:
        """Set the task description and obtain the policy generation template.

        Either accepts a custom template or generates one via LLM from the
        task description. The template must contain '<max_policy_length>'.

        Args:
            task_description: Description of the binary classification task.
            instructions_template: Optional custom template. If None, generated
                via LLM.

        Returns:
            The policy generation instructions template string.
        """
        assert task_description, "task_description must be provided"
        self._task_description = task_description

        if instructions_template:
            if max_policy_num_tag not in instructions_template:
                raise ValueError(
                    f"instructions_template must contain '{max_policy_num_tag}'"
                )
            self._pgen_instructions_template = instructions_template
            return instructions_template

        async with self._llm_semaphore:
            response = await llm.respond(
                query=f"Generate policies for:\n{task_description}",
                llm_priority=self.gen_llmc,
                response_format=str,
                instructions=POLICY_GEN_INSTRUCTIONS,
                temperature=self.gen_temperature,
            )
        await self._token_counter.append(
            provider=response.provider_model.provider,
            model=response.provider_model.model,
            value=response.total_tokens,
            caller="PolicyInduction.set_task",
        )
        if not response.response or max_policy_num_tag not in response.response:
            raise ValueError(
                "Failed to generate a valid instructions template. "
                "Refine the task description or switch models."
            )
        self._pgen_instructions_template = response.response
        return response.response

    def _get_gen_instructions(self) -> str:
        if not self._pgen_instructions_template:
            raise ValueError("Call set_task() before fit().")
        return self._pgen_instructions_template.replace(
            max_policy_num_tag, str(self.max_policy_length)
        )

    # ── Request confirmation ────────────────────────────────────────────────────

    @staticmethod
    def _llmc_label(llmc: LLMChoice) -> str:
        return llmc["model"] if isinstance(llmc, dict) else llmc.model

    def _confirm_requests(self, estimates: Dict[str, int]) -> None:
        """Print an estimated request count per model and require y/n to proceed.

        No-op (and no prompt) if the total estimate is 0 — nothing new to do,
        e.g. a fully-resumed run with everything already checkpointed.
        """
        if not self.confirm_requests or sum(estimates.values()) == 0:
            return
        lines = ["Estimated API requests:"]
        for label, count in estimates.items():
            lines.append(f"  {label}: ~{count} requests")
        lines.append("Proceed? [y/N]: ")
        answer = input("\n".join(lines)).strip().lower()
        if answer not in ("y", "yes"):
            raise RuntimeError("Aborted by user before making API requests.")

    def _estimate_fit_requests(self) -> Dict[str, int]:
        """Estimate remaining LLM calls for fit(), accounting for any checkpoint."""
        ckpt = self._read_ckpt(self._FIT_CKPT_NAME) or {}
        batches_done: int = ckpt.get("batches_done", 0)
        total_batches = sum(
            1 for _ in self._sample(self.max_samples_as_context, seed=self.random_state)
        )
        gen_remaining = max(total_batches - batches_done, 0)

        policies = ckpt.get("policies")
        n_policies = len(policies) if policies else self.max_policy_length
        scores: dict = ckpt.get("scores", {})
        n_samples = len(self._X) if self._X is not None else 0

        score_remaining = 0
        for i in range(n_policies):
            existing = scores.get(str(i))
            if not existing:
                score_remaining += n_samples
            else:
                score_remaining += sum(1 for v in existing.values() if v is None)
                score_remaining += max(n_samples - len(existing), 0)

        return {
            f"{self._llmc_label(self.gen_llmc[0])} (generation)": gen_remaining,
            f"{self._llmc_label(self.predict_llmc[0])} (scoring)": score_remaining,
        }

    def _estimate_predict_requests(self, samples: pd.DataFrame) -> Dict[str, int]:
        """Estimate remaining LLM calls for predict(), accounting for any checkpoint."""
        ckpt = self._read_ckpt(self._PREDICT_CKPT_NAME) or {}
        done_ids = set(ckpt.get("completed", {}).keys())
        remaining_samples = sum(1 for idx in samples.index if str(idx) not in done_ids)
        if hasattr(self, "_feature_order_") and self._lr is not None:
            # Zero-weight policies (common with L1) are skipped by
            # _predict_single, so only count the ones actually queried.
            n_policies = int(np.count_nonzero(self._lr.coef_[0]))
        elif hasattr(self, "_feature_order_"):
            n_policies = len(self._feature_order_)
        else:
            n_policies = self.max_policy_length
        label = self._llmc_label(self.predict_llmc[0])
        return {f"{label} (predict)": remaining_samples * n_policies}

    # ── Checkpointing ───────────────────────────────────────────────────────────

    def _ckpt_path(self, name: str) -> Path:
        return self.save_path / name

    def _write_ckpt(self, name: str, data: dict) -> None:
        """Atomically write a checkpoint file."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self._ckpt_path(name)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(data))
        tmp.replace(path)

    def _read_ckpt(self, name: str) -> dict | None:
        path = self._ckpt_path(name)
        return orjson.loads(path.read_bytes()) if path.exists() else None

    def _del_ckpt(self, *names: str) -> None:
        for name in names:
            p = self._ckpt_path(name)
            if p.exists():
                p.unlink()
        # Clean up directories the checkpoint's mkdir(parents=True) created and
        # that are now empty — including the intermediate parents (e.g. the
        # default save_path is cwd/policy_induction/<name>, so both levels get
        # created). Walk up removing empty dirs; rmdir raises OSError on the
        # first non-empty ancestor (one holding save() artifacts or anything
        # else), so this stops there and never deletes a dir with content.
        d = self.save_path
        while d != d.parent:
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent

    # Single checkpoint file spanning the whole fit() pipeline (generation +
    # scoring), so resuming never mixes progress from two independently
    # cleared checkpoints that could describe different generated policies.
    _FIT_CKPT_NAME = "fit_checkpoint.json"

    # Flush a checkpoint after this many samples complete within a single
    # policy's scoring pass, so interrupting mid-policy on a large dataset
    # still saves whatever finished instead of losing the whole policy.
    _SCORING_CKPT_EVERY = 25

    # Single checkpoint file for predict(), same automatic/unconditional
    # design as _FIT_CKPT_NAME: always on, always at self.save_path, deleted
    # once every sample is predicted.
    _PREDICT_CKPT_NAME = "predict_checkpoint.json"
    _PREDICT_CKPT_EVERY = 25

    # ── Policy generation ───────────────────────────────────────────────────────

    async def _run_generation(self, instructions: str) -> List[str]:
        """Run policy induction. Resumes from checkpoint if present."""
        ckpt = self._read_ckpt(self._FIT_CKPT_NAME) or {}
        all_policies: List[str] = ckpt.get("policies", [])
        batches_done: int = ckpt.get("batches_done", 0)

        # Skip already-done batches via islice (still advances _sample()'s
        # internal RNG state correctly) instead of iterating through them
        # inside the progress bar, so a resumed run starts the bar at
        # batches_done instead of flashing through 0..batches_done.
        remaining_batches = itertools.islice(
            self._sample(self.max_samples_as_context, seed=self.random_state),
            batches_done,
            None,
        )
        for batch_idx, sample_df in enumerate(
            tqdm(
                remaining_batches,
                initial=batches_done,
                desc="[GEN]",
                unit="",
                bar_format=_BAR_FORMAT,
                ascii=_BAR_ASCII,
            ),
            start=batches_done,
        ):
            samples_str = "\n".join(
                "\n".join(f"{col}: {val}" for col, val in row.items()) + ";"
                for row in sample_df.to_dict(orient="records")
            )
            query = (
                f"TASK DESCRIPTION:\n{self._task_description}\n\n"
                f"EXISTING POLICIES:\n{chr(10).join(all_policies)}\n\n"
                f"SAMPLES:\n{samples_str}\n\n"
            )
            async with self._llm_semaphore:
                response = await llm.respond(
                    query=query,
                    llm_priority=self.gen_llmc,
                    response_format=Policies,
                    instructions=instructions,
                    temperature=self.gen_temperature,
                )
            await self._token_counter.append(
                provider=response.provider_model.provider,
                model=response.provider_model.model,
                value=response.total_tokens,
                caller="PolicyInduction.generate",
            )
            if response.response is None:
                raise LLMError(f"Batch {batch_idx}: no response from LLM.")
            all_policies = response.response.policies[: self.max_policy_length]
            logger.info(f"Batch {batch_idx + 1}: {len(all_policies)} policies")
            self._write_ckpt(
                self._FIT_CKPT_NAME,
                {"policies": all_policies, "batches_done": batch_idx + 1, "scores": {}},
            )

        return all_policies

    async def _generate_policies(self) -> None:
        if len(self._policy_memory) > 0:
            logger.info(
                f"Skipping generation: {len(self._policy_memory)} policies in memory."
            )
            return

        instructions = self._get_gen_instructions()
        policies = await self._run_generation(instructions)

        self._policy_memory = pd.DataFrame(
            {
                "policy": policies,
                "predictions": [None] * len(policies),
            }
        )
        logger.info(f"Generation complete: {len(policies)} policies.")

    # ── Policy scoring ──────────────────────────────────────────────────────────

    def _fix_memory(self) -> None:
        """Drop rows with missing policies or malformed prediction series.

        A prediction Series with some null entries is kept as-is — it
        represents a policy that's partially scored, not a broken one. Only
        the wrong length (stale/mismatched data) resets it to unscored.
        """
        pm = self._policy_memory.dropna(subset=["policy"]).copy()
        expected = len(self._X) if self._X is not None else 0
        for idx in pm.index:
            val = pm.at[idx, "predictions"]
            if isinstance(val, pd.Series):
                if len(val) != expected:
                    pm.at[idx, "predictions"] = None
            elif val is not None:
                pm.at[idx, "predictions"] = None
        self._policy_memory = pm

    def _load_scoring_ckpt(self) -> None:
        """Restore scored policy predictions from disk into _policy_memory.

        Reads the "scores" section of the same checkpoint file that
        generation writes to, so restored scores can never describe a
        different set of policies than the ones currently in memory.
        """
        ckpt = self._read_ckpt(self._FIT_CKPT_NAME)
        if ckpt is None or self._X is None:
            return
        scores: dict = ckpt.get("scores", {})
        restored = 0
        for str_pid, pred_dict in scores.items():
            pid = int(str_pid)
            if pid not in self._policy_memory.index:
                continue
            s = pd.Series(pred_dict, dtype="object")
            s.index = s.index.astype(type(self._X.index[0]))
            self._policy_memory.at[pid, "predictions"] = s.reindex(self._X.index)
            restored += 1
        if restored:
            logger.info(f"Restored scoring checkpoint: {restored} policies pre-scored.")

    def _save_scoring_ckpt(self) -> None:
        """Atomically persist current policy predictions to disk, complete or not.

        Partial (in-progress) Series are saved too, not just fully-scored
        ones, so interrupting mid-policy still keeps whatever finished.
        Preserves the "policies"/"batches_done" section already written by
        generation, only replacing "scores" — both sections live in the
        same file for the whole fit() run.
        """
        scores: dict[str, dict] = {}
        for idx, row in self._policy_memory.iterrows():
            val = row["predictions"]
            if isinstance(val, pd.Series):
                scores[str(idx)] = {
                    str(k): (None if pd.isna(v) else v) for k, v in val.items()
                }
        ckpt = self._read_ckpt(self._FIT_CKPT_NAME) or {}
        ckpt["scores"] = scores
        self._write_ckpt(self._FIT_CKPT_NAME, ckpt)

    async def _score_single_policy(
        self,
        policy: str,
        samples: pd.DataFrame,
        existing: pd.Series | None = None,
        on_progress: Callable[[pd.Series], None] | None = None,
    ) -> pd.Series:
        """Score one policy, resuming from `existing` and checkpointing as it goes.

        Only samples still null in `existing` are (re)scored. `on_progress`
        is called with the current partial Series every
        `_SCORING_CKPT_EVERY` completions, and once more on cancellation, so
        interrupting mid-policy on a large dataset still keeps whatever
        finished instead of losing the whole policy.
        """
        out = (
            existing.copy()
            if existing is not None
            else pd.Series([None] * len(samples), index=samples.index, dtype="object")
        )
        pending = samples.loc[out.isna()]
        done_q: asyncio.Queue[None] = asyncio.Queue()

        async def worker(row_idx: Any, row: pd.Series) -> None:
            try:
                sample_str = "\n".join(f"{col}: {row[col]}" for col in row.index)
                query = (
                    f"Task description:\n{self._task_description}\n\n"
                    f"Policy:\n{policy}\n\n"
                    f"Sample:\n{sample_str}\n\n"
                )
                async with self._llm_semaphore:
                    response = await llm.respond(
                        query=query,
                        llm_priority=self.predict_llmc,
                        instructions=POLICY_PREDICT_INSTRUCTIONS,
                        response_format=Answer,
                        temperature=self.predict_temperature,
                    )
                await self._token_counter.append(
                    provider=response.provider_model.provider,
                    model=response.provider_model.model,
                    value=response.total_tokens,
                    caller="PolicyInduction.score_policy",
                )
                if response.response is None:
                    raise LLMError("No response from LLM")
                txt = str(response.response.answer).strip().upper().strip('".,;:')
                out.at[row_idx] = (
                    txt
                    if txt in {"YES", "NO"}
                    else "YES"
                    if "YES" in txt
                    else "NO"
                    if "NO" in txt
                    else None
                )
            except Exception:
                logger.warning("Scoring worker error", exc_info=True)
                out.at[row_idx] = None
            finally:
                done_q.put_nowait(None)

        it = iter(pending.iterrows())
        in_flight = 0
        since_checkpoint = 0
        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(self.llm_semaphore_limit):
                    try:
                        idx, row = next(it)
                        tg.create_task(worker(idx, row))
                        in_flight += 1
                    except StopIteration:
                        break
                while in_flight > 0:
                    await done_q.get()
                    in_flight -= 1
                    since_checkpoint += 1
                    if (
                        on_progress is not None
                        and since_checkpoint >= self._SCORING_CKPT_EVERY
                    ):
                        on_progress(out.copy())
                        since_checkpoint = 0
                    try:
                        idx, row = next(it)
                        tg.create_task(worker(idx, row))
                        in_flight += 1
                    except StopIteration:
                        pass
        finally:
            if on_progress is not None and since_checkpoint > 0:
                on_progress(out.copy())

        return out

    async def _score_policies(self) -> None:
        """Score all unscored/partially-scored policies; checkpoint as it goes."""
        if self._X is None or self._y is None:
            raise ValueError("X and y must be set before scoring.")

        self._fix_memory()
        self._load_scoring_ckpt()
        self._fix_memory()

        def needs_work(val: Any) -> bool:
            return not isinstance(val, pd.Series) or bool(val.isnull().any())

        unscored = [
            idx
            for idx in self._policy_memory.index
            if needs_work(self._policy_memory.at[idx, "predictions"])
        ]
        total = len(unscored)
        already_done = len(self._policy_memory) - total
        logger.info(f"Scoring {total} policies.")

        for i, policy_idx in enumerate(
            tqdm(
                unscored,
                initial=already_done,
                total=len(self._policy_memory),
                desc="[SCORE]",
                unit="",
                bar_format=_BAR_FORMAT,
                ascii=_BAR_ASCII,
            )
        ):
            policy = str(self._policy_memory.at[policy_idx, "policy"])
            existing = self._policy_memory.at[policy_idx, "predictions"]
            existing = existing if isinstance(existing, pd.Series) else None

            def on_progress(partial: pd.Series, _idx: Any = policy_idx) -> None:
                self._policy_memory.at[_idx, "predictions"] = partial  # type: ignore
                self._save_scoring_ckpt()

            result = await self._score_single_policy(
                policy, self._X, existing=existing, on_progress=on_progress
            )
            self._policy_memory.at[policy_idx, "predictions"] = result  # type: ignore
            self._save_scoring_ckpt()
            if (i + 1) % self.p_predict_update_interval == 0:
                logger.info(f"Scored {i + 1}/{total} policies.")

        logger.info("Scoring complete.")

    # ── Weight fitting ──────────────────────────────────────────────────────────

    def _check_memory(self, require_predictions: bool = False) -> None:
        pm = self._policy_memory
        if not isinstance(pm, pd.DataFrame) or not {"policy", "predictions"}.issubset(
            pm.columns
        ):
            raise ValueError("_policy_memory is invalid or missing required columns.")
        is_str = pm["policy"].apply(lambda x: isinstance(x, str))
        if require_predictions:
            is_pred = pm["predictions"].apply(lambda x: isinstance(x, pd.Series))
            pm = pm[is_str & is_pred]
        else:
            pm = pm[is_str]
        if len(pm) == 0:
            raise ValueError("No valid policies in memory.")
        self._policy_memory = pm  # type: ignore

    def _build_feature_matrix(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index, np.ndarray]:
        self._check_memory()
        pm = self._policy_memory
        sample_index = self._X.index  # type: ignore
        col_data: dict[str, pd.Series] = {}
        col_names: list[str] = []

        def to_binary(v: Any) -> int:
            return 1 if str(v).strip().lower() in {"1", "true", "yes", "y", "t"} else 0

        for idx, row in pm.iterrows():
            col = str(idx)
            col_data[col] = (
                row["predictions"]
                .map(to_binary)  # type: ignore
                .reindex(sample_index)
                .fillna(0)
                .astype(np.float32)
            )
            col_names.append(col)

        X_df = pd.DataFrame(col_data, index=sample_index)
        y_num = pd.Series(self._y, index=sample_index).map(to_binary).astype(int)
        return (
            X_df.values,
            y_num.to_numpy(),
            sample_index,
            np.array(col_names, dtype=str),
        )

    def _fit_weights(self) -> None:
        cfg = self.config
        X, y, _, col_names = self._build_feature_matrix()
        n_samples, n_policies = X.shape
        logger.info(f"Fitting weights: {n_samples} samples × {n_policies} policies.")

        self._feature_order_ = col_names
        self._policy_pos_ = {name: i for i, name in enumerate(col_names)}
        self._n_features_ = n_policies

        skf = StratifiedKFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )
        best_C, best_score, best_thresholds = None, -np.inf, []

        for C in tqdm(
            list(cfg.Cs),
            desc="[FIT]",
            unit="",
            bar_format=_BAR_FORMAT,
            ascii=_BAR_ASCII,
        ):
            fold_scores, fold_thresholds = [], []
            for tr_idx, val_idx in skf.split(X, y):
                lr = LogisticRegression(
                    C=C,
                    penalty=cfg.penalty,
                    solver="liblinear",
                    max_iter=500,
                    class_weight="balanced" if cfg.class_weight_balanced else None,
                    random_state=cfg.random_state,
                )
                lr.fit(X[tr_idx], y[tr_idx])
                probs = lr.predict_proba(X[val_idx])[:, 1]
                best_f, best_t = -1.0, 0.5
                for t in cfg.threshold_grid:
                    f = fbeta_score(
                        y[val_idx],
                        (probs >= t).astype(int),
                        beta=cfg.beta,
                        zero_division=0,  # type: ignore
                    )
                    if f > best_f:
                        best_f, best_t = f, t
                fold_scores.append(best_f)
                fold_thresholds.append(best_t)

            mean_f = float(np.mean(fold_scores))
            logger.info(f"C={C} → mean F{cfg.beta}={mean_f:.5f}")
            if mean_f > best_score:
                best_C, best_score, best_thresholds = C, mean_f, fold_thresholds

        if best_C is None:
            raise ValueError("config.Cs must contain at least one candidate C value.")

        final_lr = LogisticRegression(
            C=best_C,
            penalty=cfg.penalty,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced" if cfg.class_weight_balanced else None,
            random_state=cfg.random_state,
        )
        final_lr.fit(X, y)
        self._lr = final_lr
        self.threshold = float(np.median(best_thresholds)) if best_thresholds else 0.5

        self._validation_result = {
            "best_C": best_C,
            "avg_cv_fbeta": best_score,
            "beta": cfg.beta,
            "thresholds_per_fold": best_thresholds,
            "recommended_threshold": self._threshold,
            "config": asdict(cfg),
        }
        logger.info(
            f"Fit complete. C={best_C}, F{cfg.beta}={best_score:.4f}, "
            f"threshold={self._threshold:.3f}."
        )

    # ── Prediction ──────────────────────────────────────────────────────────────

    def _lr_predict(self, raw: NDArray) -> Literal["YES", "NO"]:
        if self._lr is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        vec = np.asarray(raw, dtype=float).reshape(1, -1)
        prob = self._lr.predict_proba(vec)[0, 1]
        return "YES" if prob >= self._threshold else "NO"

    async def _predict_single(
        self,
        sample_index: Any,
        sample: str,
        token_counter: TokenCounter,
    ) -> Tuple[Any, NDArray, Literal["YES", "NO"]]:
        if not hasattr(self, "_feature_order_") or self._lr is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        policies = self._policy_memory["policy"].copy()
        policies.index = policies.index.map(str)
        policies = policies.reindex(self._feature_order_)

        # Zero-weight policies (common with L1) can't affect lr.predict_proba
        # regardless of their answer, so skip querying the LLM for them —
        # `results` stays 0.0 at those positions, which is correct either way.
        weights = self._lr.coef_[0]
        tasks_to_run = [
            (pos, pt)
            for pos, pt in enumerate(policies.fillna("").astype(str).values)
            if pt.strip() and weights[pos] != 0
        ]

        results = np.zeros(len(self._feature_order_), dtype=float)
        done_q: asyncio.Queue[None] = asyncio.Queue()

        async def worker(pos: int, policy_text: str) -> None:
            try:
                query = (
                    f"Task description:\n{self._task_description}\n\n"
                    f"Policy:\n{policy_text}\n\n"
                    f"Sample:\n{sample}\n\n"
                )
                async with self._llm_semaphore:
                    response = await llm.respond(
                        query=query,
                        llm_priority=self.predict_llmc,
                        instructions=POLICY_PREDICT_INSTRUCTIONS,
                        response_format=Answer,
                        temperature=self.predict_temperature,
                    )
                await token_counter.append(
                    provider=response.provider_model.provider,
                    model=response.provider_model.model,
                    value=response.total_tokens,
                    caller="PolicyInduction.predict_single",
                )
                if response.response is None:
                    raise LLMError("No response from LLM")
                txt = str(response.response.answer).strip().upper().strip('".,;:')
                results[pos] = 1.0 if txt == "YES" or "YES" in txt else 0.0
            except Exception:
                logger.warning("Predict worker error", exc_info=True)
                results[pos] = 0.0
            finally:
                done_q.put_nowait(None)

        in_flight = 0
        async with asyncio.TaskGroup() as tg:
            for _ in range(min(self.llm_semaphore_limit, len(tasks_to_run))):
                pos, pt = tasks_to_run.pop(0)
                tg.create_task(worker(pos, pt))
                in_flight += 1
            while in_flight > 0:
                await done_q.get()
                in_flight -= 1
                if tasks_to_run:
                    pos, pt = tasks_to_run.pop(0)
                    tg.create_task(worker(pos, pt))
                    in_flight += 1

        return sample_index, results, self._lr_predict(results)

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_memory(self) -> pd.DataFrame:
        """Return the policy memory DataFrame (policy text + predictions)."""
        return self._policy_memory

    async def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[str],
    ) -> Self:
        """Fit the PolicyInduction model.

        Runs policy generation, scoring, and weight fitting in sequence. A
        single checkpoint file spans generation and scoring, so resuming
        after an interruption never restores scores for a different set of
        policies than the ones actually in memory.

        Args:
            X: Feature DataFrame.
            y: Labels ('YES'/'NO').

        Returns:
            Self.
        """
        self._set_data(X, y)
        self._confirm_requests(self._estimate_fit_requests())
        start = time.monotonic()

        await self._generate_policies()
        await self._score_policies()
        self._fit_weights()
        self._fit_duration_seconds = time.monotonic() - start
        self._fit_completed_at = datetime.now(timezone.utc).isoformat()
        self._del_ckpt(self._FIT_CKPT_NAME)
        logger.info("PolicyInduction fit complete.")
        return self

    async def predict(
        self, samples: pd.DataFrame
    ) -> AsyncGenerator[Tuple[Any, NDArray, Literal["YES", "NO"], TokenCounter], None]:
        """Yield predictions for each sample in the DataFrame.

        Automatically checkpoints to self.save_path as it goes, resuming any
        matching in-progress checkpoint found there — same unconditional,
        single-file design as fit(). The checkpoint is deleted once every
        sample has been predicted.

        Args:
            samples: DataFrame of samples to classify.

        Yields:
            (sample_index, policy_vector, prediction, token_counter)
        """
        self._check_memory()
        if not hasattr(self, "_feature_order_") or self._lr is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._confirm_requests(self._estimate_predict_requests(samples))

        token_counter = TokenCounter()
        completed: Dict[str, Tuple[List[float], str]] = {}
        remaining_samples = samples

        ckpt = self._read_ckpt(self._PREDICT_CKPT_NAME)
        if ckpt is not None:
            completed = {
                sidx: (v["vector"], v["prediction"])
                for sidx, v in ckpt["completed"].items()
            }
            token_counter = TokenCounter.from_dict(ckpt["token_counter"])
            idx_type = type(samples.index[0]) if len(samples.index) else str
            for sidx_str, (vec, pred) in completed.items():
                yield (
                    idx_type(sidx_str),
                    np.array(vec, dtype=float),
                    cast(Literal["YES", "NO"], pred),
                    token_counter,
                )
            done = set(completed.keys())
            remaining_samples = samples[~samples.index.map(str).isin(done)]
            if completed:
                logger.info(
                    f"Resumed predict checkpoint: {len(completed)} samples pre-scored."
                )

        def save_ckpt() -> None:
            self._write_ckpt(
                self._PREDICT_CKPT_NAME,
                {
                    "completed": {
                        sidx: {"vector": vec, "prediction": pred}
                        for sidx, (vec, pred) in completed.items()
                    },
                    "token_counter": token_counter.to_dict(),
                },
            )

        queue: asyncio.Queue = asyncio.Queue()
        sem = asyncio.Semaphore(self.llm_semaphore_limit)

        async def worker(idx: Any, sample_str: str) -> None:
            await sem.acquire()
            try:
                rec = await self._predict_single(idx, sample_str, token_counter)
                await queue.put(rec)
            finally:
                await queue.put("DONE")
                sem.release()

        tasks = [
            asyncio.create_task(
                worker(idx, "\n".join(f"{col}: {val}" for col, val in row.items()))
            )
            for idx, row in remaining_samples.iterrows()
        ]
        remaining = len(tasks)
        since_checkpoint = 0
        success = False
        pbar = tqdm(
            total=len(samples),
            initial=len(completed),
            desc="[PREDICT]",
            unit="",
            bar_format=_BAR_FORMAT,
            ascii=_BAR_ASCII,
        )
        try:
            while remaining > 0:
                item = await queue.get()
                if item == "DONE":
                    remaining -= 1
                else:
                    sidx, vec, pred = item
                    completed[str(sidx)] = (vec.tolist(), pred)
                    since_checkpoint += 1
                    pbar.update(1)
                    if since_checkpoint >= self._PREDICT_CKPT_EVERY:
                        save_ckpt()
                        since_checkpoint = 0
                    yield item + (token_counter,)
            success = True
        except asyncio.CancelledError:
            pass
        finally:
            pbar.close()
            for t in tasks:
                if not t.done():
                    t.cancel()
            if success:
                self._del_ckpt(self._PREDICT_CKPT_NAME)
            elif since_checkpoint > 0:
                save_ckpt()

    # ── Persistence ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_duration(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _build_report(self) -> str:
        """Build a human-readable summary of the last fit: time + ranked weights."""
        lines = [f"# PolicyInduction Report: {self.name}", ""]

        if self._fit_completed_at is not None:
            lines.append(f"**Fit completed:** {self._fit_completed_at}")
        if self._fit_duration_seconds is not None:
            lines.append(
                f"**Fit duration:** {self._format_duration(self._fit_duration_seconds)}"
            )
        lines.append(f"**Token usage:** {self._token_counter.to_dict()}")
        lines.append("")

        if self._validation_result is not None:
            v = self._validation_result
            lines.append("## Validation")
            lines.append(f"- Best C: {v['best_C']}")
            lines.append(f"- CV F{v['beta']}: {v['avg_cv_fbeta']:.4f}")
            lines.append(f"- Decision threshold: {v['recommended_threshold']:.4f}")
            lines.append("")

        if self._lr is not None and hasattr(self, "_feature_order_"):
            policies = self._policy_memory["policy"].copy()
            policies.index = policies.index.map(str)

            def clean(text: object) -> str:
                return str(text).replace("|", "\\|").replace("\n", " ")

            weights = dict(
                zip(self._feature_order_.tolist(), self._lr.coef_[0].tolist())
            )
            ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
            used = [(pid, w) for pid, w in ranked if w != 0]
            dropped = [(pid, w) for pid, w in ranked if w == 0]

            lines.append(
                f"## Policies used by the model ({len(used)}/{len(ranked)}, "
                "ranked by |weight|)"
            )
            lines.append("")
            lines.append("| Rank | Weight | Policy |")
            lines.append("|---|---|---|")
            for rank, (pid, w) in enumerate(used, 1):
                lines.append(f"| {rank} | {w:+.4f} | {clean(policies.get(pid, '?'))} |")
            lines.append("")

            if dropped:
                lines.append(f"## Policies dropped (zero weight, {len(dropped)})")
                lines.append("")
                for pid, _ in dropped:
                    lines.append(f"- {clean(policies.get(pid, '?'))}")
                lines.append("")
        else:
            lines.append("## Policies")
            lines.append("Model has not been fitted yet — no weights available.")
            lines.append("")

        return "\n".join(lines)

    def save(
        self,
        dir_path: str | PathLike[str] | None = None,
        for_production: bool = False,
    ) -> None:
        """Persist model state to disk.

        Layout::

            policy_induction.json        manifest, config, state
            policies.parquet             policy texts
            policy_predictions.parquet   scored YES/NO matrix  (dev only)
            data.parquet                 training data          (dev only)
            lr.joblib                    trained logistic regression
            report.md                    human-readable fit summary

        Args:
            dir_path: Target directory. Defaults to self.save_path.
            for_production: Strip training data; keep inference artifacts only.
        """
        base = Path(dir_path) if dir_path else self.save_path
        if base.is_file():
            raise ValueError("dir_path must be a directory, not a file.")
        base.mkdir(parents=True, exist_ok=True)

        # Training data (dev mode only)
        if not for_production:
            if (self._X is None) != (self._y is None):
                raise ValueError(
                    "Corrupted state: X and y must both be set or both None."
                )
            if self._X is not None:
                df = deepcopy(self._X)
                df["y"] = self._y
                df.to_parquet(base / "data.parquet")

        # Policy texts
        pm = self._policy_memory.copy()
        policies_df = (
            pm.drop(columns=["predictions"], errors="ignore")
            .reset_index()
            .rename(columns={"index": "policy_id"})
        )
        policies_df["policy_id"] = policies_df["policy_id"].astype(str)
        policies_df.to_parquet(base / "policies.parquet")

        # Policy predictions on training set (dev mode only)
        if not for_production and "predictions" in pm.columns:
            pred_rows = [
                {
                    "sample_index": sidx,
                    "policy_id": str(pid),
                    "pred": None if pd.isna(val) else str(val),
                }
                for pid, preds in pm["predictions"].items()
                if isinstance(preds, pd.Series)
                for sidx, val in preds.items()
            ]
            if pred_rows:
                pd.DataFrame(pred_rows).to_parquet(base / "policy_predictions.parquet")

        # LR model
        model_file = None
        if self._lr is not None:
            model_file = "lr.joblib"
            joblib_dump(self._lr, base / model_file)

        # Manifest
        feature_order = (
            self._feature_order_.tolist()
            if hasattr(self, "_feature_order_") and self._feature_order_ is not None
            else None
        )
        manifest = {
            "version": 2,
            "name": self.name,
            "gen_llmc": [
                lc if isinstance(lc, dict) else lc.model_dump() for lc in self.gen_llmc
            ],
            "predict_llmc": [
                lc if isinstance(lc, dict) else lc.model_dump()
                for lc in self.predict_llmc
            ],
            "gen_temperature": self.gen_temperature,
            "predict_temperature": self.predict_temperature,
            "llm_semaphore_limit": self.llm_semaphore_limit,
            "max_policy_length": self.max_policy_length,
            "class_ratio": list(self.class_ratio),
            "max_samples_as_context": self.max_samples_as_context,
            "p_predict_update_interval": self.p_predict_update_interval,
            "random_state": self.random_state,
            "task_description": self._task_description,
            "policy_gen_instructions_template": (
                self._pgen_instructions_template if not for_production else None
            ),
            "token_counter": None if for_production else self._token_counter.to_dict(),
            "save_path": str(self.save_path) if not for_production else None,
            "threshold": self._threshold,
            "validation_result": self._validation_result,
            "feature_order": feature_order,
            "n_features": getattr(self, "_n_features_", None),
            "model_file": model_file,
            "config": asdict(self.config),
        }
        (base / "policy_induction.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_SERIALIZE_NUMPY)
        )

        # Human-readable report
        (base / "report.md").write_text(self._build_report(), encoding="utf-8")

        logger.info(f"Model saved to {base}")

    @classmethod
    def _load(cls, dir_path: str | PathLike[str]) -> "PolicyInduction":
        from joblib import load as joblib_load

        base = Path(dir_path)
        if not base.is_dir():
            raise ValueError("dir_path must be a directory.")
        manifest_path = base / "policy_induction.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"policy_induction.json not found in {base}")

        m = orjson.loads(manifest_path.read_bytes())

        inst = cls(
            gen_llmc=m["gen_llmc"],
            predict_llmc=m["predict_llmc"],
            config=m.get("config"),
            gen_temperature=m["gen_temperature"],
            predict_temperature=m["predict_temperature"],
            llm_semaphore_limit=m["llm_semaphore_limit"],
            max_policy_length=m["max_policy_length"],
            class_ratio=tuple(m["class_ratio"]),
            max_samples_as_context=m["max_samples_as_context"],
            p_predict_update_interval=m["p_predict_update_interval"],
            random_state=m["random_state"],
            save_path=str(base),
            name=m["name"],
        )
        inst._task_description = m.get("task_description")
        inst._pgen_instructions_template = m.get("policy_gen_instructions_template")
        if tk := m.get("token_counter"):
            inst._token_counter = TokenCounter.from_dict(tk)
        inst._threshold = m.get("threshold", 0.0)
        inst._validation_result = m.get("validation_result")
        inst._llm_semaphore = asyncio.Semaphore(inst.llm_semaphore_limit)

        if (fo := m.get("feature_order")) is not None:
            inst._feature_order_ = np.array(fo, dtype=str)
            inst._n_features_ = int(m.get("n_features") or len(inst._feature_order_))
            inst._policy_pos_ = {name: i for i, name in enumerate(inst._feature_order_)}

        # Training data
        data_path = base / "data.parquet"
        if data_path.exists():
            data_df = pd.read_parquet(data_path)
            if "y" in data_df.columns:
                inst._y = data_df["y"].to_numpy(dtype=np.str_)
                inst._X = data_df.drop(columns=["y"])

        # Policies
        policies_path = base / "policies.parquet"
        if not policies_path.exists():
            raise FileNotFoundError(f"policies.parquet not found in {base}")
        p_df = pd.read_parquet(policies_path)
        if not {"policy_id", "policy"}.issubset(p_df.columns):
            raise ValueError(
                "policies.parquet must contain 'policy_id' and 'policy' columns."
            )
        p_df = p_df.set_index(p_df["policy_id"].astype(str))
        inst._policy_memory = pd.DataFrame(
            {
                "policy": p_df["policy"],
                "predictions": pd.Series(
                    [None] * len(p_df), index=p_df.index, dtype=object
                ),
            }
        )

        # Predictions on training set
        preds_path = base / "policy_predictions.parquet"
        if preds_path.exists() and inst._X is not None:
            pp = pd.read_parquet(preds_path)
            if not {"sample_index", "policy_id", "pred"}.issubset(pp.columns):
                raise ValueError("policy_predictions.parquet missing required columns.")
            pp["policy_id"] = pp["policy_id"].astype(str)
            for pid, g in pp.groupby("policy_id"):
                s = pd.Series(
                    g["pred"].values, index=g["sample_index"].values, dtype="object"
                )
                if pid in inst._policy_memory.index:
                    inst._policy_memory.at[pid, "predictions"] = s.reindex(  # type: ignore
                        inst._X.index
                    )

        # LR model
        if mf := m.get("model_file"):
            mp = base / mf
            if mp.exists():
                inst._lr = joblib_load(mp)

        return inst

    @classmethod
    def load(cls, dir_path: str | PathLike[str]) -> "PolicyInduction":
        """Load a previously saved PolicyInduction instance.

        Args:
            dir_path: Directory produced by save().
        """
        try:
            return cls._load(dir_path)
        except KeyError as e:
            raise ValueError(f"Manifest corrupted or missing key: {e}") from e

    def __repr__(self) -> str:
        return f"PolicyInduction(name={self.name})"

    def __str__(self) -> str:
        return f"PolicyInduction(name={self.name})"
