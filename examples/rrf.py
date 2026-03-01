"""RRF example — founder-success classifier.

A quick walkthrough of the Random Rule Forest (RRF) workflow:
  1. Define a task and generate YES/NO questions via an LLM
  2. Fit questions on labelled data (compute metrics)
  2b. Early semantic filtering — deduplicate questions *before* answering
  3. Filter redundant questions (prediction similarity + semantic similarity)
  4. Inspect the exclusion report to see *why* questions were dropped
  5. Compare F1 vs F-beta scoring
  6. Predict on new data
  7. Save / load and verify predictions match
  8. Founder-level prediction with train/test split

Prerequisites:
  export OPENAI_API_KEY="sk-..."
  python examples/rrf.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.core.llms._schemas import LLMChoice
from think_reason_learn.rrf import RRF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    """Print a visible section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def print_df_compact(df: pd.DataFrame, cols: list[str], n: int = 5) -> None:
    """Print the first *n* rows of *df*, showing only *cols*."""
    display = df[cols].head(n)
    print(display.to_string(index=True))
    if len(df) > n:
        print(f"  ... ({len(df)} rows total)")
    print()


# ---------------------------------------------------------------------------
# Toy dataset  (20 founder profiles)
# ---------------------------------------------------------------------------

PERSONS = [
    # --- Original 8 ---
    (
        "A: 30yo woman, CS Stanford, 6yr Google, AI healthcare startup, $2M seed.",
        "YES",
    ),
    (
        "B: 25yo man, marketing NYU, 3yr Apple marketing mgr, social media app.",
        "NO",
    ),
    (
        "C: LA doctor, UCLA, remote monitoring platform, no tech/startup exp.",
        "NO",
    ),
    (
        "D: 40yo man, law UChicago, corporate lawyer, legal-tech idea, no tech bg.",
        "YES",
    ),
    (
        "E: 28yo woman, CE Berkeley, fintech YC startup, own fintech product.",
        "NO",
    ),
    (
        "F: 32yo man, MBA Columbia, marketing Apple/Spotify, subscription box.",
        "NO",
    ),
    (
        "G: 27yo woman, IE MIT, PM Amazon logistics, 2 cofounders, accelerator.",
        "YES",
    ),
    (
        "H: 7 companies 3 industries, PM fintech startup, edutech idea.",
        "NO",
    ),
    # --- Extra 12 for train/test split ---
    (
        "I: 35yo man, PhD robotics CMU, 4yr Tesla autopilot, autonomous drones.",
        "YES",
    ),
    (
        "J: 22yo woman, dropped out art school, sells NFTs, no revenue.",
        "NO",
    ),
    (
        "K: 38yo man, 2 prior exits (acqui-hired), SaaS analytics platform.",
        "YES",
    ),
    (
        "L: 29yo woman, biology PhD, first-time founder, biotech diagnostics.",
        "NO",
    ),
    (
        "M: 45yo man, 20yr banking VP, left Goldman, wealth-tech startup.",
        "YES",
    ),
    (
        "N: 26yo man, bootcamp grad, 1yr junior dev, wants to build CRM.",
        "NO",
    ),
    (
        "O: 33yo woman, ex-Stripe engineer, YC alum, payments API startup.",
        "YES",
    ),
    (
        "P: 30yo man, MBA Wharton, 2yr McKinsey, marketplace for tutors.",
        "NO",
    ),
    (
        "Q: 28yo woman, CS MIT, 3yr Meta AI research, computer vision startup.",
        "YES",
    ),
    (
        "R: 40yo man, real estate agent, no tech bg, proptech idea.",
        "NO",
    ),
    (
        "S: 31yo woman, ex-Airbnb PM, repeat founder, travel-tech platform.",
        "YES",
    ),
    (
        "T: 24yo man, economics major, no work exp, crypto trading bot.",
        "NO",
    ),
]


async def main() -> None:  # noqa: D103
    # ------------------------------------------------------------------
    # 0. Check API key
    # ------------------------------------------------------------------
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY before running this example.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    X = pd.DataFrame({"data": [p for p, _ in PERSONS]})
    y = [label for _, label in PERSONS]

    llm_choices: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]

    # ------------------------------------------------------------------
    # 1. Create RRF and generate questions
    # ------------------------------------------------------------------
    print_section("1. Create RRF and generate questions")

    rrf = RRF(
        qgen_llmc=llm_choices,
        name="example_rrf",
        answer_similarity_func="hamming",
        max_samples_as_context=5,
        max_generated_questions=8,
        question_scoring_f_beta=0.5,  # precision-weighted F-beta
    )

    template = await rrf.set_tasks(
        task_description="Classify founders as likely to succeed or not"
    )
    print(f"Instruction template:\n  {template[:120]}...\n")

    # ------------------------------------------------------------------
    # 2. Fit (generate + answer questions, compute metrics)
    # ------------------------------------------------------------------
    print_section("2. Fit on labelled data")

    rrf = await rrf.fit(X, y, reset=True)
    qdf = rrf.get_questions()
    print(f"Generated {len(qdf)} questions.\n")

    # ------------------------------------------------------------------
    # 2b. Early semantic filtering (issue #44)
    # ------------------------------------------------------------------
    print_section("2b. Early semantic filtering during fit")

    print(
        "By default, RRF generates questions then answers ALL of\n"
        "them. With semantic_filtering_during_fit=True, near-\n"
        "duplicate questions are removed *before* the expensive\n"
        "answering step, saving LLM calls.\n"
    )
    print(
        "  RRF(\n"
        "      ...,\n"
        "      semantic_filtering_during_fit=True,\n"
        "      semantic_similarity_threshold=0.45,\n"
        "  )\n"
    )
    print(
        "This uses hashed_bag_of_words embeddings (no API calls)\n"
        "to deduplicate questions immediately after generation.\n"
        "The exclusion_report() will show 'semantic_similarity'\n"
        "entries for any questions removed during this step."
    )

    # ------------------------------------------------------------------
    # 3. Compare F1 vs F-beta scores
    # ------------------------------------------------------------------
    print_section("3. F1 vs F-beta (beta=0.5, precision-weighted)")

    score_cols = ["question", "precision", "recall", "f1_score", "f_beta_score"]
    available = [c for c in score_cols if c in qdf.columns]
    print_df_compact(qdf, available, n=8)

    # ------------------------------------------------------------------
    # 4. Filter redundant questions
    # ------------------------------------------------------------------
    print_section("4. Filter redundant questions")

    # 4a — prediction similarity (identical answer patterns)
    rrf.filter_questions_on_pred_similarity(threshold=0.9)
    pred_excluded = qdf[qdf["exclusion"] == "prediction_similarity"]
    print(f"Prediction-similarity filter: {len(pred_excluded)} excluded")

    # 4b — semantic similarity (similar wording)
    await rrf.filter_questions_on_semantics(
        threshold=0.45, emb_model="hashed_bag_of_words"
    )
    qdf = rrf.get_questions()  # refresh after filtering
    sem_excluded = qdf[qdf["exclusion"] == "semantics"]
    print(f"Semantic-similarity filter:   {len(sem_excluded)} excluded")

    active = qdf[qdf["exclusion"].isna()]
    print(f"Active questions remaining:   {len(active)}\n")

    # ------------------------------------------------------------------
    # 5. Exclusion report (issue #45)
    # ------------------------------------------------------------------
    print_section("5. Exclusion report — why were questions dropped?")

    report = rrf.exclusion_report()
    assert isinstance(report, pd.DataFrame)
    if len(report) == 0:
        print("No exclusions recorded.\n")
    else:
        # Summary counts by reason
        counts: dict[str, int] = defaultdict(int)
        for reason in report["exclusion_reason"]:
            counts[reason] += 1
        print("Exclusion counts by reason:")
        for reason, count in counts.items():
            print(f"  {reason}: {count}")
        print()

        # Detailed table
        detail_cols = [
            "excluded_question_id",
            "exclusion_reason",
            "reference_question_id",
            "similarity_score",
            "threshold",
        ]
        available_detail = [c for c in detail_cols if c in report.columns]
        print_df_compact(report, available_detail, n=10)

    # ------------------------------------------------------------------
    # 6. Predict
    # ------------------------------------------------------------------
    print_section("6. Predict on the same samples")

    # Collect predictions grouped by sample index
    sample_votes: dict[int, list[str]] = defaultdict(list)
    async for sample_idx, _qid, answer, _tc in rrf.predict(X):
        sample_votes[int(sample_idx)].append(answer)

    print(f"{'Idx':<5} {'Truth':<7} {'YES':>4} {'NO':>4}  {'Majority':<10}")
    print("-" * 38)
    for idx in sorted(sample_votes):
        votes = sample_votes[idx]
        yes_count = votes.count("YES")
        no_count = votes.count("NO")
        majority = "YES" if yes_count >= no_count else "NO"
        truth = y[idx]
        marker = " *" if majority != truth else ""
        print(
            f"{idx:<5} {truth:<7} {yes_count:>4} {no_count:>4}  {majority:<10}{marker}"
        )
    print("\n(* = mismatch with ground truth)")

    # ------------------------------------------------------------------
    # 7. Save / load / verify
    # ------------------------------------------------------------------
    print_section("7. Save, load, and verify predictions")

    save_dir = Path("_artifacts/example_rrf")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rrf.save(str(save_dir), for_production=True)
    print(f"Saved to {save_dir}/")

    loaded = RRF.load(str(save_dir))
    loaded_votes: dict[int, list[str]] = defaultdict(list)
    async for sample_idx, _qid, answer, _tc in loaded.predict(X):
        loaded_votes[int(sample_idx)].append(answer)

    # Compare majority labels
    match = True
    for idx in sorted(sample_votes):
        orig = (
            "YES"
            if sample_votes[idx].count("YES") >= sample_votes[idx].count("NO")
            else "NO"
        )
        load = (
            "YES"
            if loaded_votes[idx].count("YES") >= loaded_votes[idx].count("NO")
            else "NO"
        )
        if orig != load:
            match = False
            print(f"  Mismatch at sample {idx}: original={orig}, loaded={load}")

    if match:
        print("Predictions match after save/load.\n")

    # Clean up
    shutil.rmtree(save_dir)

    # ------------------------------------------------------------------
    # 8. Founder-level prediction (issue #47)
    # ------------------------------------------------------------------
    print_section("8. Founder-level prediction (train/test split)")

    # Split into train (first 16) / test (last 4) — 80/20
    split_idx = 16
    X_all = pd.DataFrame({"data": [p for p, _ in PERSONS]})
    y_all = [label for _, label in PERSONS]
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train = y_all[:split_idx]
    y_test = y_all[split_idx:]

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")

    # Fit: generates questions, answers them, computes metrics,
    # and tunes (K, T) for founder-level aggregation — all on train.
    rrf_fl = RRF(
        qgen_llmc=llm_choices,
        name="example_founder_level",
        max_samples_as_context=8,
        max_generated_questions=8,
        question_scoring_f_beta=0.5,
    )
    await rrf_fl.set_tasks(
        task_description="Classify founders as likely to succeed or not"
    )
    await rrf_fl.fit(X_train, y_train, reset=True)

    print(f"Tuned aggregation: K={rrf_fl._aggregation_k}, T={rrf_fl._aggregation_t}\n")

    # Predict on held-out test set
    results = await rrf_fl.predict_founder_level(X_test)
    print(results[["prediction", "yes_count"]].to_string())
    print()

    # Compare to ground truth
    n_correct = sum(
        results.loc[i, "prediction"] == y_test[j] for j, i in enumerate(X_test.index)
    )
    print(f"Test accuracy: {n_correct}/{len(y_test)}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
