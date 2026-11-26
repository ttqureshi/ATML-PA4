"""Accounting, divergence, LoRA, and stage-guard checks that stay on CPU."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from task3.src.distill import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LEARNING_RATE,
    MATH_QUOTA,
    MATH_RESPONSE_CAP,
    SCORED_TOKEN_QUOTA,
    SHORT_QUOTA,
    SHORT_RESPONSE_CAP,
    STEP_TOKENS,
    WEIGHT_DECAY,
    LoRALinearStandIn,
    build_lora_config,
    cycle_indices,
    loss_for_method,
    lora_trainable_count,
    optimizer_update_sizes,
    per_token_forward_kl,
    presentations_for_quota,
)
from task3.src.evaluate import (
    extract_gsm8k_number,
    extract_short_form,
    numbers_equal,
    repetition_order,
    score_standalone,
    short_form_match,
)
from task3.src.metrics import (
    analytical_speedup,
    bootstrap_mean,
    break_even_acceptance,
    expected_acceptance,
    forward_kl,
    position_bin,
    prefix_alignment,
    requests_to_amortize,
    reverse_kl,
    total_variation,
)


class AccountingTests(unittest.TestCase):
    def test_forward_kl_hand_example(self) -> None:
        teacher = np.array([0.5, 0.5])
        student = np.array([0.25, 0.75])
        hand = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        self.assertAlmostEqual(forward_kl(teacher, student), hand, places=12)
        self.assertAlmostEqual(reverse_kl(teacher, student), forward_kl(student, teacher), places=12)
        self.assertGreater(abs(forward_kl(teacher, student) - reverse_kl(teacher, student)), 1e-6)
        logits_p = torch.log(torch.tensor(teacher))
        logits_q = torch.log(torch.tensor(student))
        learned = per_token_forward_kl(logits_p, logits_q)
        self.assertAlmostEqual(float(learned), hand, places=6)
        self.assertEqual(loss_for_method("off_policy"), "forward_kl")
        self.assertEqual(loss_for_method("on_policy"), "forward_kl")

    def test_expected_acceptance_is_one_minus_tv(self) -> None:
        teacher = np.array([0.7, 0.2, 0.1])
        draft = np.array([0.4, 0.4, 0.2])
        tv = total_variation(teacher, draft)
        self.assertAlmostEqual(tv, 0.3)
        self.assertAlmostEqual(expected_acceptance(teacher, draft), 1.0 - tv)
        self.assertAlmostEqual(prefix_alignment(teacher, draft)["expected_acceptance"], 1.0 - tv)

    def test_lora_rank_and_stand_in_count(self) -> None:
        config = build_lora_config()
        self.assertEqual(config.r, LORA_R)
        self.assertEqual(config.lora_alpha, LORA_ALPHA)
        self.assertEqual(config.lora_dropout, LORA_DROPOUT)
        self.assertEqual(set(config.target_modules), set(LORA_TARGET_MODULES))
        self.assertEqual(LORA_R, 8)
        self.assertEqual(LORA_ALPHA, 16)
        self.assertEqual(LORA_DROPOUT, 0.0)
        shapes = [(16, 16)] * 4
        self.assertEqual(lora_trainable_count(shapes, LORA_R), 4 * 8 * (16 + 16))
        layer = LoRALinearStandIn(16, 16)
        trainable = sum(parameter.numel() for parameter in layer.parameters() if parameter.requires_grad)
        self.assertEqual(trainable, 8 * (16 + 16))
        inputs = torch.randn(3, 16)
        self.assertTrue(torch.allclose(layer(inputs), layer.base(inputs)))

    def test_quota_schedule_and_updates(self) -> None:
        self.assertEqual(MATH_QUOTA + SHORT_QUOTA, SCORED_TOKEN_QUOTA)
        self.assertEqual(SCORED_TOKEN_QUOTA, 40_000)
        self.assertEqual(MATH_RESPONSE_CAP, 192)
        self.assertEqual(SHORT_RESPONSE_CAP, 32)
        self.assertEqual(STEP_TOKENS, 256)
        self.assertAlmostEqual(LEARNING_RATE, 3e-4)
        self.assertAlmostEqual(WEIGHT_DECAY, 0.01)
        indices = cycle_indices(3, 5, 6304)
        self.assertEqual(indices[3], indices[0])
        self.assertEqual(indices[4], indices[1])
        self.assertEqual(sorted(indices[:3]), [0, 1, 2])
        presentations = presentations_for_quota([10, 10, 10], 25)
        self.assertEqual([item["scored_tokens"] for item in presentations], [10, 10, 5])
        self.assertTrue(presentations[-1]["truncated"])
        self.assertFalse(presentations[0]["truncated"])
        self.assertEqual(optimizer_update_sizes([100, 100, 100]), [256, 44])
        self.assertEqual(optimizer_update_sizes([256, 256]), [256, 256])
        self.assertEqual(optimizer_update_sizes([200, 50]), [250])

    def test_speedup_break_even_and_bootstrap(self) -> None:
        self.assertAlmostEqual(analytical_speedup(1.0, 2, 0.5, 1.0), 1.5)
        root = break_even_acceptance(2, 0.5, 1.0)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, (math.sqrt(5.0) - 1.0) / 2.0, places=6)
        self.assertIsNone(break_even_acceptance(2, 1.0, 2.0))
        self.assertIsNone(requests_to_amortize(10.0, 0.0))
        self.assertIsNone(requests_to_amortize(10.0, -1.0))
        self.assertAlmostEqual(requests_to_amortize(10.0, 2.0), 5.0)
        report = bootstrap_mean([1.0, 2.0, 3.0, 4.0], seed=6304)
        self.assertAlmostEqual(report["mean_diff"], 2.5)
        self.assertLessEqual(report["ci_low"], report["mean_diff"])
        self.assertGreaterEqual(report["ci_high"], report["mean_diff"])
        self.assertEqual(position_bin(1), "1-16")
        self.assertEqual(position_bin(32), "17-32")
        self.assertEqual(position_bin(64), "33-64")
        self.assertEqual(position_bin(65), "65-192")
        self.assertEqual(repetition_order(0), ("target_only", "speculative"))
        self.assertEqual(repetition_order(1), ("speculative", "target_only"))

    def test_answer_extractors_ignore_speculative_ranking(self) -> None:
        self.assertEqual(extract_gsm8k_number("work\n#### 1,024"), "1024")
        self.assertIsNone(extract_gsm8k_number("no digits here"))
        self.assertTrue(numbers_equal("1024", "1024.0"))
        self.assertEqual(extract_short_form("May\nignore me"), "May")
        self.assertTrue(short_form_match("pk", "PK"))
        scored = score_standalone("math", "#### 3", "the answer is #### 3", length_capped=True)
        self.assertTrue(scored["correct"])
        self.assertTrue(scored["truncated"])
        missing = score_standalone("familiar", "", "May", length_capped=False)
        self.assertTrue(missing["unextractable"])
        self.assertFalse(missing["correct"])

    def test_gpu_stages_refuse_without_cuda(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available, so the refusal path is not the local one")
        from task3.run import stage_distill, stage_smoke

        with self.assertRaises(SystemExit):
            stage_smoke()
        with self.assertRaises(SystemExit):
            stage_distill()


if __name__ == "__main__":
    unittest.main()
