"""The vectorized masks, quantizer and bulk P3 repair match the original loops exactly.

The reference functions below are the original per-row / per-group / per-weight
implementations. The fast versions must reproduce them bit for bit.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from task1.src.allocation import ProjectionSize, _proportional_fractions, allocate_p3_zeros
from task1.src.constants import SPARSITY_CLIP
from task1.src.pruning import (
    fraction_in_clip,
    keep_mask,
    sparsity_bounds,
    two_four_keep_mask,
    uniform_zero_budget,
)
from task1.src.quantize import quantize_weight


def ref_keep_mask(scores, zero_counts):
    metric = np.asarray(scores, dtype=np.float64)
    rows, cols = metric.shape
    if np.isscalar(zero_counts):
        counts = np.full(rows, int(zero_counts), dtype=np.int64)
    else:
        counts = np.asarray(zero_counts, dtype=np.int64)
    mask = np.ones((rows, cols), dtype=bool)
    columns = np.arange(cols)
    for row in range(rows):
        k = int(counts[row])
        if k <= 0:
            continue
        if k >= cols:
            mask[row] = False
            continue
        order = np.lexsort((columns, metric[row]))
        mask[row, order[:k]] = False
    return mask


def ref_two_four(scores):
    metric = np.asarray(scores, dtype=np.float64)
    rows, cols = metric.shape
    mask = np.zeros((rows, cols), dtype=bool)
    local = np.arange(4)
    for row in range(rows):
        for start in range(0, cols, 4):
            block = metric[row, start : start + 4]
            order = np.lexsort((local, -block))
            mask[row, start + order[:2]] = True
    return mask


def ref_quantize(weight, bits, group_size):
    matrix = np.asarray(weight, dtype=np.float64)
    rows, cols = matrix.shape
    qmax = (1 << bits) - 1
    n_groups = rows * math.ceil(cols / group_size)
    values = np.empty_like(matrix)
    scales = np.empty(n_groups)
    zero_points = np.empty(n_groups, dtype=np.int64)
    pad_counts = np.empty(n_groups, dtype=np.int64)
    codes = np.empty(rows * cols, dtype=np.int64)
    g = 0
    c = 0
    for row in range(rows):
        for start in range(0, cols, group_size):
            end = min(start + group_size, cols)
            chunk = matrix[row, start:end]
            pad = group_size - chunk.shape[0]
            ranged = np.concatenate([chunk, np.full(pad, chunk[-1])]) if pad else chunk
            w_min = float(ranged.min())
            w_max = float(ranged.max())
            scale = (w_max - w_min) / qmax
            if scale == 0.0 or not math.isfinite(scale):
                scale = 1.0
            zero = int(np.clip(np.rint(-w_min / scale), 0, qmax))
            q = np.clip(np.rint(chunk / scale) + zero, 0, qmax).astype(np.int64)
            values[row, start:end] = scale * (q - zero)
            scales[g] = scale
            zero_points[g] = zero
            pad_counts[g] = pad
            codes[c : c + q.shape[0]] = q
            c += q.shape[0]
            g += 1
    return values, scales, zero_points, pad_counts, codes


def ref_allocate(sizes, sensitivities, sparsity, tau):
    low, high = SPARSITY_CLIP
    target = uniform_zero_budget([(s.rows, s.cols) for s in sizes], sparsity)
    fractions = _proportional_fractions(sizes, sensitivities, target, tau)
    clipped = [min(high, max(low, f)) for f in fractions]
    zeros, bounds = [], []
    for size, fraction in zip(sizes, clipped):
        z_min, z_max = sparsity_bounds(size.numel, low, high)
        bounds.append((z_min, z_max))
        zeros.append(min(z_max, max(z_min, int(fraction * size.numel + 0.5))))
    while sum(zeros) != target:
        direction = 1 if sum(zeros) < target else -1
        candidates = [
            i
            for i, (z_min, z_max) in enumerate(bounds)
            if (direction > 0 and zeros[i] < z_max) or (direction < 0 and zeros[i] > z_min)
        ]

        def distance(i):
            fraction = zeros[i] / sizes[i].numel
            return ((high - fraction) if direction > 0 else (fraction - low), -i)

        zeros[max(candidates, key=distance)] += direction
    assert all(fraction_in_clip(z, s.numel, low, high) for z, s in zip(zeros, sizes))
    return zeros


class FastPathEquivalence(unittest.TestCase):
    def test_keep_mask_matches_lexsort_loop(self):
        rng = np.random.default_rng(1)
        for trial in range(30):
            rows, cols = int(rng.integers(1, 40)), int(rng.integers(1, 60))
            # Coarse values force many ties.
            scores = rng.integers(0, 4, size=(rows, cols)).astype(np.float64) * (trial % 2) + rng.random((rows, cols)) * (1 - trial % 2)
            counts = rng.integers(-2, cols + 3, size=rows)
            np.testing.assert_array_equal(keep_mask(scores, counts), ref_keep_mask(scores, counts))
            k = int(rng.integers(0, cols + 1))
            np.testing.assert_array_equal(keep_mask(scores, k), ref_keep_mask(scores, k))

    def test_two_four_matches_loop(self):
        rng = np.random.default_rng(2)
        for trial in range(30):
            rows, cols = int(rng.integers(1, 20)), 4 * int(rng.integers(1, 20))
            scores = rng.integers(0, 3, size=(rows, cols)).astype(np.float64) if trial % 2 else rng.standard_normal((rows, cols))
            np.testing.assert_array_equal(two_four_keep_mask(scores), ref_two_four(scores))

    def test_quantizer_bit_identical(self):
        rng = np.random.default_rng(3)
        for trial in range(40):
            rows, cols = int(rng.integers(1, 12)), int(rng.integers(1, 300))
            gs = int(rng.choice([4, 8, 16, 128]))
            bits = int(rng.choice([2, 3, 4, 8]))
            w = rng.standard_normal((rows, cols)) * 10 ** rng.uniform(-4, 1)
            if trial % 5 == 0:
                w[0] = 0.0  # degenerate groups
            if trial % 7 == 0:
                w[-1] = 0.37
            got = quantize_weight(w, bits, gs)
            want = ref_quantize(w, bits, gs)
            for a, b in zip((got.values, got.scales, got.zero_points, got.pad_counts, got.codes), want):
                np.testing.assert_array_equal(a, b)

    def test_bulk_repair_matches_unit_steps(self):
        rng = np.random.default_rng(4)
        for trial in range(60):
            n = int(rng.integers(2, 25))
            sizes = [ProjectionSize(f"p{i}", int(rng.integers(1, 60)), int(rng.integers(1, 400))) for i in range(n)]
            if trial % 4 == 0:  # identical shapes and sensitivities: ties everywhere
                sizes = [ProjectionSize(f"p{i}", 16, 64) for i in range(n)]
                sens = [0.3] * n
            else:
                sens = list(np.exp(rng.normal(-3, 2, n)))
                if trial % 3 == 0:
                    sens[0] = 0.0
            sparsity = float(rng.choice([0.25, 0.5, 0.75]))
            tau = float(rng.choice([0.5, 1.0, 2.0, 4.0]))
            try:
                want = ref_allocate(sizes, sens, sparsity, tau)
            except Exception:
                with self.assertRaises(Exception):
                    allocate_p3_zeros(sizes, sens, sparsity, tau)
                continue
            self.assertEqual(allocate_p3_zeros(sizes, sens, sparsity, tau), want)

    def test_bulk_repair_on_qwen_shapes_is_fast_and_exact(self):
        shapes = {"q": (1536, 1536), "k": (256, 1536), "v": (256, 1536), "o": (1536, 1536),
                  "g": (8960, 1536), "u": (8960, 1536), "d": (1536, 8960)}
        sizes = [ProjectionSize(f"{layer}.{k}", *s) for layer in range(28) for k, s in shapes.items()]
        sens = list(np.exp(np.random.default_rng(0).normal(-3, 1.5, len(sizes))))
        for sparsity in (0.5, 0.75):
            for tau in (0.5, 1.0, 2.0):
                zeros = allocate_p3_zeros(sizes, sens, sparsity, tau)
                self.assertEqual(sum(zeros), uniform_zero_budget([(s.rows, s.cols) for s in sizes], sparsity))


    def test_fp16_storage_is_exact_for_every_consumer(self):
        from task1.src.apply import prune_matrix, quantize_awq, quantize_uniform
        from task1.src.pipeline import _as_fp16

        rng = np.random.default_rng(5)
        w = rng.standard_normal((16, 256)) * 0.05
        mean_abs = np.abs(rng.standard_normal(256)) + 0.1
        col = np.abs(rng.standard_normal(256)) + 0.1
        payloads = [quantize_uniform(w, 4)["values_original_space"]]
        awq = quantize_awq(w, mean_abs, 0.5, 3)
        payloads += [awq["values_original_space"], awq["values_scaled_space"]]
        payloads.append(prune_matrix(w, col, "wanda", 0.5)["values"])
        for values in payloads:
            half = _as_fp16(values)
            self.assertEqual(half.dtype, np.float16)
            # write_weight and measure_reconstruction both read through float32.
            np.testing.assert_array_equal(half.astype(np.float32), np.asarray(values, dtype=np.float32))
        with self.assertRaises(RuntimeError):
            _as_fp16(np.array([[0.1]]))


if __name__ == "__main__":
    unittest.main()
