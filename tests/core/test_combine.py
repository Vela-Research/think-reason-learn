"""The shared combiner: F-beta thresholds and the logistic fit."""

from __future__ import annotations

import numpy as np

from think_reason_learn.core._combine import best_threshold, fit_logistic


class TestCombine:
    def test_best_threshold(self) -> None:
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        threshold = best_threshold(scores, np.array([False, False, True, True]), 1.0)
        assert threshold == 0.5  # midway between 0.4 and 0.6

    def test_logistic_learns_a_separable_feature(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.random(80) > 0.5
        features = np.column_stack([y.astype(float), rng.random(80)])
        fitted = fit_logistic(
            features,
            y,
            l1_ratios=(0.5,),
            cs=4,
            cv=3,
            beta=1.0,
            class_weight=None,
            random_state=0,
        )
        assert fitted.coef[0] > abs(fitted.coef[1])
        assert ((fitted.scores(features) >= fitted.threshold) == y).mean() == 1.0


# A local model's answers to five rules for 24 loan applicants (7 defaulted): the
# first rule is right on 6 of the 7. One 2-fold split made every weight look
# useless, so Policy Induction weighed no rule and flagged everyone.
RULE_ANSWERS = np.array(
    [
        [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1], [1, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1],
    ],
    dtype=float,
)  # fmt: skip
DEFAULTED = np.array(
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    dtype=bool,
)


def test_a_small_dataset_keeps_its_one_good_rule() -> None:
    combiner = fit_logistic(
        RULE_ANSWERS,
        DEFAULTED,
        l1_ratios=(1.0,),
        cv=2,
        beta=0.5,
        class_weight=None,
        random_state=42,
    )
    assert combiner.coef[0] > 1
