"""Paired bootstrap intervals used by the assignment's required comparisons."""

from __future__ import annotations

import numpy as np


def bootstrap_paired_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 6304,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Resample paired examples and return the mean of a - b with a percentile interval."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired bootstrap expects two equal-length 1-D arrays")
    if len(left) == 0:
        raise ValueError("paired bootstrap needs at least one example")
    diff = left - right
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[draws].mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": low,
        "ci_high": high,
        "n": int(len(diff)),
        "n_resamples": int(n_resamples),
    }


def bootstrap_pooled_nll_diff(
    window_nll_a: np.ndarray,
    window_count_a: np.ndarray,
    window_nll_b: np.ndarray,
    window_count_b: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 6304,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Resample whole evaluation windows, recompute pooled mean NLL, then exponentiate to perplexity.

    window_nll_* stores the SUM of token NLLs in that window, not the mean.
    The same resampled window indices are applied to both conditions.
    """
    nll_a = np.asarray(window_nll_a, dtype=np.float64)
    nll_b = np.asarray(window_nll_b, dtype=np.float64)
    count_a = np.asarray(window_count_a, dtype=np.float64)
    count_b = np.asarray(window_count_b, dtype=np.float64)
    if not (len(nll_a) == len(nll_b) == len(count_a) == len(count_b)):
        raise ValueError("window arrays must have the same length")
    n = len(nll_a)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))

    def _pooled(nll: np.ndarray, count: np.ndarray, idx: np.ndarray) -> np.ndarray:
        total_nll = nll[idx].sum(axis=1)
        total_count = count[idx].sum(axis=1)
        return total_nll / total_count

    mean_a = _pooled(nll_a, count_a, draws)
    mean_b = _pooled(nll_b, count_b, draws)
    delta = mean_a - mean_b
    point_a = float(nll_a.sum() / count_a.sum())
    point_b = float(nll_b.sum() / count_b.sum())
    return {
        "nll_a": point_a,
        "nll_b": point_b,
        "nll_diff": point_a - point_b,
        "nll_diff_ci_low": float(np.quantile(delta, alpha / 2)),
        "nll_diff_ci_high": float(np.quantile(delta, 1 - alpha / 2)),
        "ppl_a": float(np.exp(point_a)),
        "ppl_b": float(np.exp(point_b)),
        "ppl_diff": float(np.exp(point_a) - np.exp(point_b)),
        "n_windows": int(n),
        "n_resamples": int(n_resamples),
    }
