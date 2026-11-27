"""Prefix-bank parsing, grouped scoring, summaries and the consistency check (CPU, tiny models)."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from task3.src.alignment import (
    SPEC_BANK,
    compact,
    consistency_check,
    decode_disagreements,
    load_prefix_bank,
    prefix_rows_from_generation,
    score_bank,
    select_disagreements,
    summarize_scores,
)
from task3.src.metrics import prefix_alignment
from task3.tests.test_cached_decoder import VOCAB, tiny_qwen


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **_kwargs):
        text = messages[0]["content"]
        return [1, 4] + [5 + (ord(ch) % (VOCAB - 6)) for ch in text[:5]] + [3]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{int(token)}" for token in ids)


def _supplied_records(rng) -> list[dict]:
    """Records in the exact shape setup_data.py prefix_banks() writes."""
    records = []
    prompts = [("GSM8K-0000", "add numbers", "math"), ("SHORT-familiar-0000", "month?", "short"), ("SHORT-shifted-0003", "code?", "short")]
    for bank in ("teacher", "original_draft"):
        for j, (pid, prompt, domain) in enumerate(prompts):
            seq = [int(x) for x in rng.integers(5, VOCAB, size=int(rng.integers(20, 80)))]
            for row in prefix_rows_from_generation(pid, prompt, domain, seq, bank, producer=bank):
                records.append(
                    dict(prompt_id=pid, domain=domain, bank=bank, position=row["position"], prompt=prompt,
                         response_prefix_token_ids=row["response_prefix_token_ids"], source_length=len(seq),
                         source_generation_seed=6304 + 10007 * j)
                )
    return records


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "prefix_banks_locked.jsonl.gz"
        with gzip.open(cls.path, "wt", encoding="utf-8") as handle:
            for record in _supplied_records(np.random.default_rng(3)):
                handle.write(json.dumps(record) + "\n")
        cls.teacher = tiny_qwen(31, sharpness=3.0)
        cls.drafts = {"original": tiny_qwen(32, sharpness=3.0), "off_policy_seed6304": tiny_qwen(33, sharpness=3.0)}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_missing_bank_fails_loudly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_prefix_bank(Path(self.tmp.name) / "absent.jsonl.gz")

    def test_supplied_format_parses_with_domains(self) -> None:
        rows = load_prefix_bank(self.path)
        self.assertEqual({row["bank"] for row in rows}, {"teacher", "original_draft"})
        domains = {row["prompt_id"]: row["domain"] for row in rows}
        self.assertEqual(domains, {"GSM8K-0000": "math", "SHORT-familiar-0000": "familiar", "SHORT-shifted-0003": "shifted"})
        self.assertTrue(all(len(row["response_prefix_token_ids"]) == row["position"] - 1 for row in rows))
        self.assertTrue(any(row["position"] > 64 for row in rows))
        self.assertTrue(all(row["source_length"] is not None for row in rows))

    def test_grouped_scores_equal_per_prefix_reference(self) -> None:
        tokenizer = FakeTokenizer()
        rows = load_prefix_bank(self.path)
        scored = score_bank(self.teacher, self.drafts, tokenizer, rows)
        self.assertEqual(len(scored), len(rows) * len(self.drafts))
        lookup = {(row["bank"], row["prompt_id"], row["position"], row["draft"]): row for row in scored}
        for row in rows[::7]:
            ids = torch.tensor([tokenizer.apply_chat_template([{"content": row["prompt"]}]) + row["response_prefix_token_ids"]])
            with torch.no_grad():
                p = torch.softmax(self.teacher(input_ids=ids).logits[0, -1].double(), -1).numpy()
                for name, model in self.drafts.items():
                    q = torch.softmax(model(input_ids=ids).logits[0, -1].double(), -1).numpy()
                    expected = prefix_alignment(p, q)
                    got = lookup[(row["bank"], row["prompt_id"], row["position"], name)]
                    for key, value in expected.items():
                        self.assertAlmostEqual(got[key], value, places=5, msg=key)

        summary = summarize_scores(scored)
        self.assertEqual(set(summary), {"teacher", "original_draft"})
        for bank in summary.values():
            self.assertTrue(bank["same_prefix_ids_for_every_draft"])
            n = bank["drafts"]["original"]["overall"]["n_tokens"]
            self.assertEqual(n, bank["drafts"]["off_policy_seed6304"]["overall"]["n_tokens"])
            parts = sum(item["n_tokens"] for item in bank["drafts"]["original"]["by_domain"].values())
            self.assertEqual(parts, n)
            self.assertEqual(
                bank["drafts"]["original"]["positions_1_64"]["n_tokens"] + bank["drafts"]["original"]["positions_65_192"]["n_tokens"], n
            )
            diff = bank["paired_differences"]["off_policy_seed6304_minus_original"]["expected_acceptance"]
            self.assertLessEqual(diff["ci_low"], diff["mean_diff"])
            self.assertEqual(diff["unmatched_tokens"], 0)

        examples = decode_disagreements(scored, rows, tokenizer, k=1)
        self.assertEqual(select_disagreements(scored, k=1), select_disagreements(list(reversed(scored)), k=1))
        self.assertIn("largest TV", examples["rule"])
        for example in examples["examples"]:
            self.assertEqual(set(example["drafts"]), set(self.drafts))
            self.assertEqual(example["drafts"]["original"]["greedy_agreement"], 0.0)
        self.assertNotIn("teacher_top", compact(scored)[0])

    def test_consistency_check_on_identical_law(self) -> None:
        tokenizer = FakeTokenizer()
        rows = [row for row in load_prefix_bank(self.path) if row["bank"] == "teacher"]
        twin = [dict(row, bank=SPEC_BANK, producer="original_draft_gamma4") for row in rows]
        scored = score_bank(self.teacher, {"original": self.drafts["original"]}, tokenizer, rows + twin)
        report = consistency_check(scored)
        self.assertEqual(report["n_ci_excluding_zero"], 0)
        self.assertAlmostEqual(report["overall"]["teacher_entropy"]["mean_diff"], 0.0, places=9)
        self.assertTrue(report["overall"]["source_length"]["consistent"])


if __name__ == "__main__":
    unittest.main()
