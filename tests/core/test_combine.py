"""The shared combiner: F-beta thresholds and the logistic fit."""

from __future__ import annotations

import numpy as np

from think_reason_learn.core._combine import best_threshold, fit_logistic


class TestCombine:
    def test_best_threshold(self) -> None:
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        threshold, f = best_threshold(scores, np.array([False, False, True, True]), 1.0)
        assert threshold == 0.5 and f == 1.0  # midway between 0.4 and 0.6

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
