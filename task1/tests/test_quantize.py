"""Quantizer formula, partial groups, AWQ identity, and packed bytes."""

from __future__ import annotations

import unittest

import numpy as np

from task1.src.quantize import (
    awq_reconstruct,
    channel_scales,
    pack_codes_little_endian,
    quantize_weight,
    relative_rescale_error,
    tensor_packed_bytes,
)
from task1.src.storage import quantized_model_bytes, sparse_storage_bytes


class QuantizeTests(unittest.TestCase):
    def test_round_trip_including_partial_group(self) -> None:
        # bits = 2, qmax = 3. Full group [0, 1, 2, 3] has scale 1 and zero-point 0.
        # The partial group is [0, 3]. Repeating the edge value 3 does not move
        # min or max, so the same scale reconstructs both real weights.
        # Padding is still recorded: two replicated positions.
        weight = np.array([[0.0, 1.0, 2.0, 3.0, 0.0, 3.0]], dtype=np.float64)
        quantized = quantize_weight(weight, bits=2, group_size=4)
        np.testing.assert_allclose(quantized.values, weight)
        self.assertEqual(quantized.pad_counts.tolist(), [0, 2])
        self.assertAlmostEqual(float(quantized.scales[0]), 1.0)
        self.assertAlmostEqual(float(quantized.scales[1]), 1.0)
        self.assertEqual(int(quantized.zero_points[0]), 0)

        # Negative endpoint: min -2, max 1, scale 1, zero-point round(2) = 2.
        # q = clip(round(w) + 2, 0, 3) and w_hat = q - 2 recovers the vector.
        negative = np.array([[-2.0, -1.0, 0.0, 1.0]], dtype=np.float64)
        quantized_negative = quantize_weight(negative, bits=2, group_size=4)
        np.testing.assert_allclose(quantized_negative.values, negative)
        self.assertEqual(int(quantized_negative.zero_points[0]), 2)
        self.assertEqual(int(quantized_negative.pad_counts[0]), 0)

    def test_rescaling_before_rounding_is_an_identity(self) -> None:
        rng = np.random.default_rng(6304)
        weight = rng.normal(size=(5, 7))
        inputs = rng.normal(size=(9, 7))
        scales = rng.uniform(0.3, 3.0, size=7)
        self.assertLess(relative_rescale_error(weight, inputs, scales), 1e-10)

    def test_alpha_zero_reproduces_uniform_quantization(self) -> None:
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(3, 6))
        inputs = rng.normal(size=(8, 6))
        mean_abs = np.mean(np.abs(inputs), axis=0)
        scales, original, _scaled = awq_reconstruct(weight, mean_abs, alpha=0.0, bits=4, group_size=4)
        np.testing.assert_allclose(scales, np.ones(6))
        np.testing.assert_allclose(channel_scales(mean_abs, 0.0), np.ones(6))
        uniform = quantize_weight(weight, bits=4, group_size=4)
        np.testing.assert_allclose(original, uniform.values)

    def test_packed_bytes_match_a_hand_count(self) -> None:
        # 4 x 6 matrix, 3 bits, group size 4.
        # Groups per row = 2, so 8 groups. Metadata = 8 * 4 = 32 bytes.
        # Payload bits = 24 * 3 = 72, which is 9 bytes. Total = 41.
        self.assertEqual(tensor_packed_bytes(4, 6, bits=3, group_size=4), 41)
        weight = np.arange(24, dtype=np.float64).reshape(4, 6)
        quantized = quantize_weight(weight, bits=3, group_size=4)
        payload = pack_codes_little_endian(quantized.codes, 3)
        self.assertEqual(len(payload) + 4 * quantized.n_groups, 41)

    def test_sparse_estimate_hand_count(self) -> None:
        # 24 eligible weights, 12 of them nonzero, 4 excluded parameters.
        # Bits = 24 mask bits + 12 * 16 value bits = 216 bits = 27 bytes.
        # Excluded FP16 = 8 bytes. Sparse total = 35. Dense FP16 = 56.
        report = sparse_storage_bytes(24, 12, 4)
        self.assertEqual(report["eligible_sparse_bytes"], 27)
        self.assertEqual(report["excluded_fp16_bytes"], 8)
        self.assertEqual(report["unstructured_sparse_bytes"], 35)
        self.assertEqual(report["dense_fp16_bytes"], 56)
        self.assertEqual(report["padding_bits"], 0)

    def test_quantized_model_bytes_include_scales_metadata_and_excluded(self) -> None:
        # One 4 x 6 INT3 group-4 matrix is 41 bytes, plus 6 FP16 scales (12)
        # and 4 excluded parameters (8). Total 61.
        report = quantized_model_bytes([{"rows": 4, "cols": 6, "bits": 3, "scale_bytes": 12}], n_excluded=4, group_size=4)
        self.assertEqual(report["metadata_bytes"], 32)
        self.assertEqual(report["payload_bytes"], 9)
        self.assertEqual(report["serialized_scale_bytes"], 12)
        self.assertEqual(report["packed_bytes"], 61)


if __name__ == "__main__":
    unittest.main()
