"""CPU checks for linear attention, masks, the forget gate, and one training step."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from task2.src.linear import (
    GATE_BIAS_INIT,
    explicit_feature_sum,
    forget_gate,
    gated_linear_attention,
    linear_attention_prefix,
    linear_attention_recurrent,
)
from task2.src.masks import (
    DILATION_OFFSETS,
    allowed_key_counts,
    dilation_allowed_count,
    dilation_keep,
    mixed_local_dilated_keep,
    window_allowed_count,
    window_keep,
)
from task2.src.model import MQARModel, query_cross_entropy
from task2.src.mqar_data import check_generator_invariants, frozen_path, load_generator
from task2.src.schedule import PEAK_LR, learning_rate
from task2.src.theory import RECEPTIVE_FIELD_NOTE, linear_state_bytes, receptive_field
from task2.src.train import JOBS, SEEDS, smoke_optimizer_step
import json
import gzip


class LinearAttentionTests(unittest.TestCase):
    def test_recurrence_matches_explicit_feature_sum(self) -> None:
        torch.manual_seed(6304)
        query = torch.randn(2, 4, 12, 32)
        key = torch.randn(2, 4, 12, 32)
        value = torch.randn(2, 4, 12, 32)
        recurrent = linear_attention_recurrent(query, key, value)
        explicit = explicit_feature_sum(query, key, value)
        prefix = linear_attention_prefix(query, key, value)
        self.assertTrue(torch.allclose(recurrent, explicit, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(prefix, recurrent, atol=1e-4, rtol=1e-4))


class MaskTests(unittest.TestCase):
    def test_window_and_dilation_allowed_key_counts(self) -> None:
        length = 256
        window = window_keep(length)
        window_counts = allowed_key_counts(window)
        for position in (0, 10, 63, 64, 255):
            self.assertEqual(int(window_counts[position]), window_allowed_count(position))
        dilated = dilation_keep(length)
        dilated_counts = allowed_key_counts(dilated)
        for position in (0, 10, 64, 255):
            self.assertEqual(int(dilated_counts[position]), dilation_allowed_count(position))
        self.assertEqual(len(DILATION_OFFSETS), 64)
        self.assertEqual(DILATION_OFFSETS[0], 0)
        self.assertEqual(DILATION_OFFSETS[-1], 252)
        mixed = mixed_local_dilated_keep(length)
        self.assertEqual(int(mixed[0, 0, 255].sum()), 64)
        self.assertEqual(int(mixed[0, 1, 10].sum()), window_allowed_count(10))
        self.assertEqual(int(mixed[0, 2, 255].sum()), 64)
        self.assertEqual(int(mixed[0, 3, 10].sum()), dilation_allowed_count(10))
        self.assertEqual(receptive_field(), 126)
        self.assertIn("2*(64-1)=126", RECEPTIVE_FIELD_NOTE)


class GateTests(unittest.TestCase):
    def test_gate_adds_under_five_percent_and_keeps_state_bytes(self) -> None:
        baseline = MQARModel("linear")
        gated = MQARModel("gated")
        base_count = sum(parameter.numel() for parameter in baseline.parameters())
        gated_count = sum(parameter.numel() for parameter in gated.parameters())
        self.assertLess((gated_count - base_count) / base_count, 0.05)
        self.assertEqual(linear_state_bytes(), linear_state_bytes())
        self.assertTrue(torch.allclose(gated.blocks[0].attn.gate_bias.detach(), torch.full((4,), GATE_BIAS_INIT)))
        self.assertTrue(torch.all(gated.blocks[0].attn.gate_weight.detach() == 0))
        opened = torch.sigmoid(torch.tensor(GATE_BIAS_INIT))
        self.assertAlmostEqual(float(opened), 0.880797, places=4)
        query = torch.randn(2, 4, 6, 32)
        key = torch.randn(2, 4, 6, 32)
        value = torch.randn(2, 4, 6, 32)
        output = gated_linear_attention(
            query,
            key,
            value,
            gated.blocks[0].attn.gate_weight.detach(),
            gated.blocks[0].attn.gate_bias.detach(),
        )
        self.assertTrue(torch.isfinite(output).all())
        initial = forget_gate(torch.randn(2, 4, 32), torch.zeros(4, 32), torch.full((4,), GATE_BIAS_INIT))
        self.assertTrue(torch.allclose(initial, opened.expand_as(initial)))


class TrainingStepTests(unittest.TestCase):
    def test_schedule_and_job_count(self) -> None:
        self.assertAlmostEqual(learning_rate(0), PEAK_LR / 200)
        self.assertAlmostEqual(learning_rate(199), PEAK_LR)
        self.assertAlmostEqual(learning_rate(4499), 0.15 * PEAK_LR)
        self.assertEqual(len(JOBS) * len(SEEDS), 42)

    def test_loss_ignores_non_query_positions(self) -> None:
        logits = torch.zeros(1, 4, 16)
        logits[0, 1, 5] = 4
        positions = torch.tensor([[1]])
        answers = torch.tensor([[5]])
        first = query_cross_entropy(logits, positions, answers)
        edited = logits.clone()
        edited[0, 3, 0] = 50
        second = query_cross_entropy(edited, positions, answers)
        self.assertAlmostEqual(float(first), float(second))

    def test_one_optimizer_step_on_random_tokens(self) -> None:
        loss = smoke_optimizer_step()
        self.assertTrue(loss == loss)
        model = MQARModel("gated")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        tokens = torch.randint(0, 256, (2, 16))
        positions = torch.tensor([[1, 4], [2, 6]])
        answers = torch.tensor([[130, 140], [150, 160]])
        step_loss = query_cross_entropy(model(tokens), positions, answers)
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        self.assertTrue(torch.isfinite(step_loss))


class GeneratorInvariantTests(unittest.TestCase):
    def test_course_generator_and_frozen_record(self) -> None:
        check_generator_invariants(sample=4)
        path = frozen_path("test")
        self.assertTrue(path.is_file(), msg=str(path))
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            frozen = json.loads(handle.readline())
        _split, regime, k_token, index_token = frozen["id"].split("-")
        made = load_generator().make_example("test", int(k_token[1:]), regime, int(index_token))
        self.assertEqual(made["tokens"], frozen["tokens"])
        self.assertEqual(made["answer_ids"], frozen["answer_ids"])
        self.assertEqual(made["source_positions"], frozen["source_positions"])
        load_generator().validate(frozen)

    def test_prediction_was_written(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("PREDICTION.md").read_text(encoding="utf-8")
        self.assertIn("raise far-regime accuracy", text)
        self.assertIn("near-regime accuracy should stay close", text)


if __name__ == "__main__":
    unittest.main()
