"""CPU checks for dense, tiled, and SDPA attention."""

from __future__ import annotations

import unittest

import torch

from task2.src.attention import (
    dense_causal_attention,
    future_perturbation_changes_only_future,
    sdpa_causal_attention,
    tiled_causal_attention,
)


class DenseCausalityTests(unittest.TestCase):
    def test_future_perturbation_does_not_change_past_outputs(self) -> None:
        torch.manual_seed(6304)
        query = torch.randn(2, 4, 32, 64)
        key = torch.randn(2, 4, 32, 64)
        value = torch.randn(2, 4, 32, 64)
        delta = future_perturbation_changes_only_future(
            dense_causal_attention, query, key, value, position=7
        )
        self.assertEqual(float(delta.max()), 0.0)
        tiled_delta = future_perturbation_changes_only_future(
            lambda q, k, v: tiled_causal_attention(q, k, v, tile_size=8),
            query,
            key,
            value,
            position=7,
        )
        self.assertEqual(float(tiled_delta.max()), 0.0)

    def test_all_masked_row_and_extreme_scores_stay_finite(self) -> None:
        torch.manual_seed(6304)
        query = torch.randn(2, 4, 8, 16, requires_grad=True)
        key = torch.randn(2, 4, 8, 16)
        value = torch.randn(2, 4, 8, 16)
        key = key.clone()
        key[:, :, 0, :] = 1e4
        allowed = torch.ones(8, 8, dtype=torch.bool).tril()
        allowed[4, :] = False
        output = dense_causal_attention(query, key, value, allowed)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.all(output[:, :, 4, :] == 0))
        reference = dense_causal_attention(query.detach().float(), key.float(), value.float(), allowed)
        self.assertTrue(torch.allclose(output.float(), reference, atol=1e-5, rtol=1e-5))
        output.sum().backward()
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(query.grad).all())


class TiledAttentionTests(unittest.TestCase):
    def test_tiled_matches_dense_fp32_for_two_tile_sizes(self) -> None:
        torch.manual_seed(6304)
        query = torch.randn(2, 4, 32, 64)
        key = torch.randn(2, 4, 32, 64)
        value = torch.randn(2, 4, 32, 64)
        reference = dense_causal_attention(query, key, value)
        for tile in (8, 16):
            tiled = tiled_causal_attention(query, key, value, tile_size=tile)
            self.assertTrue(
                torch.allclose(tiled, reference, atol=1e-4, rtol=1e-4),
                msg=f"tile {tile} max {(tiled - reference).abs().max()}",
            )

    def test_sdpa_matches_dense_on_a_short_sequence(self) -> None:
        torch.manual_seed(6304)
        query = torch.randn(2, 4, 16, 32)
        key = torch.randn(2, 4, 16, 32)
        value = torch.randn(2, 4, 16, 32)
        reference = dense_causal_attention(query, key, value)
        sdpa = sdpa_causal_attention(query, key, value)
        self.assertTrue(torch.allclose(sdpa, reference, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
