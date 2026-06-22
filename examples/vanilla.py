"""Vanilla classifier example — direct-prompting founder screener.

The vanilla classifier is the zero-shot baseline: each profile is sent to the
LLM once with a fixed scoring prompt, and the returned score is thresholded.
This walkthrough shows the recommended way to run it:

  1. Build a classifier from the recommended preset (5-point @ >=4, T=0)
  2. Score profiles and inspect scores / reasoning
  3. Evaluate against ground-truth labels (precision / recall / F0.5)
  4. Sweep thresholds cheaply (one set of LLM calls, re-thresholded)
  5. Compare a precision preset vs a coverage preset

Prerequisites:
  export GOOGLE_AI_API_KEY="..."        # or put it in a .env file
  # optional: pick a model your key can access
  export VANILLA_MODEL="gemini-2.5-flash"
  python examples/vanilla.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import pandas as pd

from think_reason_learn.core.llms import GoogleChoice
from think_reason_learn.core.llms._schemas import LLMChoice
from think_reason_learn.vanilla import (
    FLASH_5POINT,
    PRO_PRECISION,
    VanillaClassifier,
)


def print_section(title: str) -> None:
    """Print a visible section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Toy dataset (12 founder profiles, label 1 = likely success)
# ---------------------------------------------------------------------------

PERSONS = [
    ("A: 30yo woman, CS Stanford, 6yr Google, AI healthcare startup, $2M seed.", 1),
    ("B: 25yo man, marketing NYU, 3yr Apple marketing mgr, social media app.", 0),
    ("C: LA doctor, UCLA, remote monitoring platform, no tech/startup exp.", 0),
    ("I: 35yo man, PhD robotics CMU, 4yr Tesla autopilot, autonomous drones.", 1),
    ("J: 22yo woman, dropped out art school, sells NFTs, no revenue.", 0),
    ("K: 38yo man, 2 prior exits (acqui-hired), SaaS analytics platform.", 1),
    ("L: 29yo woman, biology PhD, first-time founder, biotech diagnostics.", 0),
    ("M: 45yo man, 20yr banking VP, left Goldman, wealth-tech startup.", 1),
    ("N: 26yo man, bootcamp grad, 1yr junior dev, wants to build CRM.", 0),
    ("O: 33yo woman, ex-Stripe engineer, YC alum, payments API startup.", 1),
    ("R: 40yo man, real estate agent, no tech bg, proptech idea.", 0),
    ("Q: 28yo woman, CS MIT, 3yr Meta AI research, computer vision startup.", 1),
]


async def main() -> None:  # noqa: D103
    if not os.environ.get("GOOGLE_AI_API_KEY"):
        print("Error: set GOOGLE_AI_API_KEY before running this example.")
        print("  export GOOGLE_AI_API_KEY='...'")
        sys.exit(1)

    model = os.environ.get("VANILLA_MODEL", "gemini-2.5-flash")
    llmc: list[LLMChoice] = [GoogleChoice(model=model)]

    X = pd.DataFrame({"data": [p for p, _ in PERSONS]})
    y = [label for _, label in PERSONS]

    # ------------------------------------------------------------------
    # 1. Recommended preset (5-point @ >=4, T=0)
    # ------------------------------------------------------------------
    print_section("1. Build from the recommended preset")

    clf = VanillaClassifier.from_preset(FLASH_5POINT, llmc=llmc)
    print(f"Preset:        {FLASH_5POINT.name}")
    print(f"Model:         {model}")
    print(f"Format:        {clf.scoring_format}  (scores {list(clf.scale.values)})")
    print(f"Threshold:     score >= {clf.threshold}")
    print(f"Temperature:   {clf.temperature}\n")

    # ------------------------------------------------------------------
    # 2. Score and inspect
    # ------------------------------------------------------------------
    print_section("2. Score profiles")

    scored = await clf.score(X)
    view = scored[["score", "predicted", "confidence"]].copy()
    view["truth"] = y
    print(view.to_string())
    print()

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    print_section("3. Evaluate (precision / recall / F0.5)")

    metrics = await clf.evaluate(X, y)
    print(f"Precision:     {metrics.precision:.3f}")
    print(f"Recall:        {metrics.recall:.3f}")
    print(f"F0.5:          {metrics.f_beta:.3f}")
    print(f"TP/FP/TN/FN:   {metrics.tp}/{metrics.fp}/{metrics.tn}/{metrics.fn}")
    print(f"Tokens used:   {clf.token_usage.to_dict()}\n")

    # ------------------------------------------------------------------
    # 4. Threshold sweep (cheap — one set of calls, re-thresholded)
    # ------------------------------------------------------------------
    print_section("4. Threshold sweep")

    sweep = await clf.sweep_thresholds(X, y)
    print(sweep.to_string(index=False))
    print("\nThe paper recommends >=4 for 5-point; never >=5.\n")

    # ------------------------------------------------------------------
    # 5. Precision preset (thinking model -> ternary @ >=2)
    # ------------------------------------------------------------------
    print_section("5. A precision-leaning preset")

    print(
        "PRO_PRECISION uses ternary @ >=2. The thinking model is format-\n"
        "agnostic on F0.5, so the format is an operational choice: ternary@2\n"
        "for precision, 5-point@4 for coverage.\n"
    )
    print(f"  format={PRO_PRECISION.scoring_format}  "
          f"threshold>={PRO_PRECISION.threshold}  T={PRO_PRECISION.temperature}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
