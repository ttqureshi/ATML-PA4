"""Magnitude pruning, Wanda scores, and the 2:4 diagnostic mask.

Per output row, the smallest ``floor(s * n_cols)`` scores are zeroed. Ties keep
the lower column index among the survivors: ``np.lexsort`` orders by score and
breaks ties by column index, and the first k indices in that order are removed.
"""

from __future__ import annotations

import math

import numpy as np

from task1.src.constants import SPARSITY_CLIP


def magnitude_scores(weight: np.ndarray) -> np.ndarray:
    """S_ij = |W_ij|."""
    return np.abs(np.asarray(weight, dtype=np.float64))


def column_l2(inputs: np.ndarray) -> np.ndarray:
    """||X_j||_2 for X shaped [tokens, in_features]."""
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("activation matrix must be [tokens, in_features]")
    return np.sqrt(np.sum(x * x, axis=0))


def wanda_scores(weight: np.ndarray, input_column_l2: np.ndarray) -> np.ndarray:
    """S_ij = |W_ij| * ||X_j||_2."""
    scores = magnitude_scores(weight)
    norms = np.asarray(input_column_l2, dtype=np.float64).reshape(1, -1)
    if norms.shape[1] != scores.shape[1]:
        raise ValueError("column-norm length must match the weight input dimension")
    return scores * norms


def zeros_per_row_uniform(n_cols: int, sparsity: float) -> int:
    """Wanda/magnitude per-row budget: floor(s * n_cols)."""
    if n_cols < 1:
        raise ValueError("n_cols must be positive")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must lie in [0, 1]")
    return int(math.floor(sparsity * n_cols + 1e-12))


def uniform_zero_budget(shapes: list[tuple[int, int]], sparsity: float) -> int:
    """Total zeros under per-row uniform pruning. ``shapes`` are (rows, cols)."""
    return sum(rows * zeros_per_row_uniform(cols, sparsity) for rows, cols in shapes)


def zeros_per_row_from_total(n_rows: int, n_zeros: int, n_cols: int) -> np.ndarray:
    """Spread an integer zero budget across rows, one weight at a time.

    The first ``n_zeros % n_rows`` rows receive one extra zero. Counts stay
    inside ``[0, n_cols]``.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("shape must be positive")
    if n_zeros < 0 or n_zeros > n_rows * n_cols:
        raise ValueError("zero count does not fit in the matrix")
    base, remainder = divmod(int(n_zeros), int(n_rows))
    counts = np.full(n_rows, base, dtype=np.int64)
    counts[:remainder] += 1
    if int(counts.max()) > n_cols:
        raise ValueError("per-row zero count exceeds the row length")
    return counts


def keep_mask(scores: np.ndarray, zero_counts: np.ndarray | int) -> np.ndarray:
    """Boolean mask, True where the weight is kept.

    ``zero_counts`` is one integer for every row, or a length-``rows`` vector.
    """
    metric = np.asarray(scores, dtype=np.float64)
    if metric.ndim != 2:
        raise ValueError("scores must have shape [rows, cols]")
    rows, cols = metric.shape
    if np.isscalar(zero_counts):
        counts = np.full(rows, int(zero_counts), dtype=np.int64)
    else:
        counts = np.asarray(zero_counts, dtype=np.int64)
        if counts.shape != (rows,):
            raise ValueError("zero_counts must be a scalar or one count per row")
    # A stable argsort orders by score and keeps the lower column index first
    # among equal scores, the same order as ``np.lexsort((columns, row))``.
    # A weight is zeroed when its rank in that order is below the row's count.
    order = np.argsort(metric, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(cols), (rows, cols)), axis=1)
    return ranks >= counts.reshape(-1, 1)


def uniform_keep_mask(scores: np.ndarray, sparsity: float) -> np.ndarray:
    zeros = zeros_per_row_uniform(scores.shape[1], sparsity)
    return keep_mask(scores, zeros)


def apply_keep_mask(weight: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pruned = np.array(weight, dtype=np.float64, copy=True)
    pruned[~mask] = 0.0
    return pruned


def two_four_keep_mask(scores: np.ndarray) -> np.ndarray:
    """Keep the two highest scores in every group of four input channels.

    Ties keep the lower column index. The width must be a multiple of four.
    """
    metric = np.asarray(scores, dtype=np.float64)
    if metric.ndim != 2:
        raise ValueError("scores must have shape [rows, cols]")
    rows, cols = metric.shape
    if cols % 4 != 0:
        raise ValueError("2:4 grouping needs a multiple of 4 input channels")
    blocks = metric.reshape(rows, cols // 4, 4)
    # Stable argsort on the negated scores: highest first, lower index on ties,
    # the same order as ``np.lexsort((local, -block))``.
    order = np.argsort(-blocks, axis=-1, kind="stable")
    mask = np.zeros(blocks.shape, dtype=bool)
    np.put_along_axis(mask, order[..., :2], True, axis=-1)
    return mask.reshape(rows, cols)


def score_summary(scores: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("score summary needs at least one score")
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "numel": int(flat.size),
    }


def relative_output_mse(weight: np.ndarray, pruned: np.ndarray, inputs: np.ndarray) -> float:
    """||(W - W_s) X||^2 / ||W X||^2 with X shaped [tokens, in].

    A zero-energy original output has sensitivity 0. The ratio is the P3
    distortion proxy and the P1/P2 calibration layer-output error.
    """
    w = np.asarray(weight, dtype=np.float64)
    ws = np.asarray(pruned, dtype=np.float64)
    x = np.asarray(inputs, dtype=np.float64)
    original = x @ w.T
    delta = x @ (w - ws).T
    denominator = float(np.sum(original * original))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(delta * delta) / denominator)


def sparsity_bounds(n_weights: int, low: float | None = None, high: float | None = None) -> tuple[int, int]:
    """Inclusive integer zero counts whose fraction lies in ``[low, high]``.

    0.10 and 0.85 are treated as the exact ratios 1/10 and 17/20 so the clip
    does not depend on binary rounding of those decimals.
    """
    if low is None or high is None:
        low_default, high_default = SPARSITY_CLIP
        low = low_default if low is None else low
        high = high_default if high is None else high
    if n_weights < 1:
        raise ValueError("n_weights must be positive")
    if abs(low - 0.10) < 1e-12 and abs(high - 0.85) < 1e-12:
        z_min = (n_weights + 9) // 10
        z_max = (17 * n_weights) // 20
    else:
        z_min = math.ceil(low * n_weights - 1e-12)
        z_max = math.floor(high * n_weights + 1e-12)
    if z_min > z_max:
        raise ValueError(f"no integer sparsity in [{low}, {high}] for n={n_weights}")
    return int(z_min), int(z_max)


def fraction_in_clip(n_zeros: int, n_weights: int, low: float = 0.10, high: float = 0.85) -> bool:
    if n_weights <= 0:
        return False
    fraction = n_zeros / n_weights
    return (low - 1e-12) <= fraction <= (high + 1e-12)
