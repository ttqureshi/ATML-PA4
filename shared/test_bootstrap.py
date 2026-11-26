"""CPU checks for the shared measurement helpers."""

from __future__ import annotations

import unittest

import numpy as np

from shared.bootstrap import bootstrap_paired_mean_diff, bootstrap_pooled_nll_diff
from shared.seed import DEFAULT_SEED


class BootstrapTests(unittest.TestCase):
    def test_paired_mean_matches_difference(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 1.0, 1.0, 1.0])
        out = bootstrap_paired_mean_diff(a, b, n_resamples=200, seed=DEFAULT_SEED)
        self.assertAlmostEqual(out["mean_diff"], 1.5)
        self.assertLessEqual(out["ci_low"], out["mean_diff"])
        self.assertGreaterEqual(out["ci_high"], out["mean_diff"])

    def test_pooled_nll_exponentiates_after_pooling(self) -> None:
        nll_a = np.array([10.0, 30.0])
        count_a = np.array([10.0, 10.0])
        nll_b = np.array([20.0, 20.0])
        count_b = np.array([10.0, 10.0])
        out = bootstrap_pooled_nll_diff(
            nll_a, count_a, nll_b, count_b, n_resamples=50, seed=DEFAULT_SEED
        )
        self.assertAlmostEqual(out["nll_a"], 2.0)
        self.assertAlmostEqual(out["ppl_a"], float(np.exp(2.0)))
        self.assertAlmostEqual(out["nll_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
