"""CPU checks for the KV budget, eviction, quantization, and rotary position ids."""

from __future__ import annotations

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from task2.src.kv_budget import (
    eviction_bytes,
    hybrid_bytes,
    kv_dims_from_config,
    memory_budget,
    memory_fp16,
)
from task2.src.kv_cache import RotaryCacheEditor
from task2.src.kv_eval import split_matches_full
from task2.src.kv_prompts import policy_fields, split_user_text, tokens_overlapping
from task2.src.kv_quant import (
    assert_int4_channel_token_matches_budget,
    dequantize_groups,
    pack_codes,
    quantize_groups,
    unpack_codes,
)
from task2.src.kv_select import (
    protected_positions,
    select_eviction,
    select_hybrid,
    select_question_aware,
    select_sink_plus_recent,
)


class BudgetTests(unittest.TestCase):
    def test_mb_matches_a_hand_example(self) -> None:
        n_layers, h_kv, d_h, length, bits = 2, 3, 5, 80, 4
        groups = length // 64
        remainder = length % 64
        per_channel = groups * ((64 * bits) // 8 + 4) + 2 * remainder
        expected = 2 * n_layers * h_kv * d_h * per_channel
        self.assertEqual(memory_budget(length, n_layers, h_kv, d_h, bits), expected)
        self.assertEqual(memory_fp16(length, n_layers, h_kv, d_h), 4 * n_layers * h_kv * d_h * length)
        self.assertEqual(groups, 1)
        self.assertEqual(remainder, 16)
        self.assertEqual(expected, 4080)

    def test_dims_are_read_from_the_config(self) -> None:
        config = Qwen2Config(
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        self.assertEqual(kv_dims_from_config(config), (3, 2, 16))


class EvictionTests(unittest.TestCase):
    def test_keeps_sinks_and_recent_inside_the_budget(self) -> None:
        length = 512
        n_layers, h_kv, d_h = 1, 1, 64
        for mode in ("sink_recent", "question"):
            scores = torch.zeros(n_layers, length)
            scores[:, 80] = 10
            chosen, n_keep, total = select_eviction(length, n_layers, h_kv, d_h, mode, scores)
            budget = memory_budget(length, n_layers, h_kv, d_h, 4)
            self.assertLessEqual(total, budget)
            self.assertEqual(eviction_bytes(n_keep, n_layers, h_kv, d_h), total)
            protected = set(protected_positions(length))
            self.assertGreaterEqual(n_keep, len(protected))
            for layer in chosen:
                self.assertEqual(len(layer), n_keep)
                self.assertTrue(protected.issubset(layer))
            if mode == "question":
                self.assertIn(80, chosen[0])
        explicit = select_sink_plus_recent(200, 80)
        self.assertTrue(set(protected_positions(200)).issubset(explicit))
        self.assertEqual(len(explicit), 80)
        self.assertLessEqual(eviction_bytes(len(explicit), 1, 1, 64), eviction_bytes(80, 1, 1, 64))
        ranked_scores = torch.zeros(200)
        ranked_scores[20] = 5
        ranked = select_question_aware(ranked_scores, 80)
        self.assertIn(20, ranked)
        self.assertTrue(set(range(8)).issubset(ranked))
        self.assertTrue(set(range(200 - 64, 200)).issubset(ranked))

    def test_hybrid_stays_inside_the_budget(self) -> None:
        # A 512-token cache cannot keep the sink and recent tokens in FP16 and
        # still pack the rest into complete INT2 groups under M_b. 2048 can.
        length = 2048
        scores = torch.randn(1, length)
        plan = select_hybrid(scores, 1, 1, 64)
        self.assertLessEqual(plan["bytes"], plan["budget"])
        self.assertEqual(plan["int2_count"] % 64, 0)
        self.assertEqual(plan["bytes"], hybrid_bytes(plan["fp16_count"], plan["int2_count"], 1, 1, 64, bits=2))
        protected = set(protected_positions(length))
        for kept in plan["fp16_indices"]:
            self.assertTrue(protected.issubset(kept))


class QuantizationTests(unittest.TestCase):
    def test_completed_int4_group_unpacks_to_quantized_values(self) -> None:
        values = torch.linspace(-3, 3, 64).view(1, 64)
        codes, scale, offset = quantize_groups(values, 4)
        packed = pack_codes(codes, 4)
        unpacked = unpack_codes(packed, 64, 4)
        self.assertTrue(torch.equal(unpacked, codes.view(-1)))
        reconstructed = dequantize_groups(unpacked.view(1, 64), scale, offset)
        direct = dequantize_groups(codes, scale, offset)
        self.assertTrue(torch.allclose(reconstructed, direct))
        self.assertTrue(torch.allclose(reconstructed, scale.view(1, 1) * codes.to(scale.dtype) + offset.view(1, 1)))
        assert_int4_channel_token_matches_budget(128, 1, 1, 64)


class PositionIdTests(unittest.TestCase):
    def test_retained_tokens_keep_original_rotary_ids(self) -> None:
        cache = DynamicCache()
        originals = []
        for layer in range(2):
            keys = torch.zeros(1, 2, 10, 4)
            keys[0, :, :, 0] = torch.arange(10).float()
            values = torch.zeros_like(keys)
            cache.update(keys, values, layer)
            originals.append(keys.clone())
        editor = RotaryCacheEditor(cache, 10)
        editor.retain([[0, 2, 5, 9], [0, 2, 5, 9]])
        self.assertTrue(torch.equal(editor.position_ids[0], torch.tensor([0, 2, 5, 9])))
        self.assertTrue(torch.equal(editor.position_ids[1], torch.tensor([0, 2, 5, 9])))
        kept = cache.layers[0].keys[0, 0, :, 0]
        self.assertTrue(torch.equal(kept, torch.tensor([0.0, 2.0, 5.0, 9.0])))
        position_ids, cache_position = editor.decode_inputs(torch.device("cpu"))
        self.assertEqual(int(position_ids), 10)
        self.assertEqual(int(cache_position), 4)
        self.assertNotEqual(int(position_ids), int(cache_position))
        for layer in cache.layers:
            layer.keys = torch.cat((layer.keys, torch.zeros(1, 2, 1, 4)), dim=-2)
            layer.values = torch.cat((layer.values, torch.zeros(1, 2, 1, 4)), dim=-2)
        appended = editor.note_appended()
        self.assertEqual(appended, 10)
        self.assertEqual(int(editor.position_ids[0][-1]), 10)
        self.assertEqual(editor.next_absolute, 11)
        self.assertEqual(editor.physical_length(), 5)
        nxt, physical = editor.decode_inputs(torch.device("cpu"))
        self.assertEqual(int(nxt), 11)
        self.assertEqual(int(physical), 5)

    def test_full_cache_decode_matches_a_single_forward(self) -> None:
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_sliding_window=False,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(6304)
        model = Qwen2ForCausalLM(config).eval()
        tokens = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            report = split_matches_full(model, tokens)
        self.assertTrue(report["finite"])
        self.assertLess(report["max_abs"], 1e-4)
        self.assertEqual(report["position_ids"], list(range(12)))

    def test_policy_view_omits_gold_spans(self) -> None:
        context, question = split_user_text("ledger\nQUESTION_BEGIN\nWhat is the access code?")
        self.assertTrue(question.startswith("QUESTION_BEGIN"))
        self.assertNotIn("QUESTION_BEGIN", context)
        visible = policy_fields(
            {
                "id": "KV",
                "user_text": "ledger",
                "code": "26836",
                "evidence_char_start": 1,
                "evidence_char_end": 4,
                "position": "start",
                "distractors": 0,
                "target_tokens": 2048,
            }
        )
        self.assertNotIn("code", visible)
        self.assertNotIn("evidence_char_start", visible)
        self.assertEqual(tokens_overlapping([(0, 0), (0, 4), (4, 8)], 4, 8), [2])


if __name__ == "__main__":
    unittest.main()
