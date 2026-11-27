"""The eval/analyze record path on CPU with tiny random Qwen2 models (no downloads).

Covers the speed benchmark rows and their accounting identities, warm-setup,
isolated costs at several cache lengths, the speed projection fields, the
workload and output-length breakdowns, standalone scoring records, matched
continuations, the bank-vs-measured table, deployment break-even and the
wrapper-overhead summary.
"""

from __future__ import annotations

import unittest

import numpy as np

from task3.run import _benchmark, _continuation_ids, _continuations, _standalone
from task3.src.analysis import (
    bank_vs_measured,
    deployment_break_even,
    standalone_summary,
    training_summary,
    workload_breakdown,
    wrapper_overhead_summary,
)
from task3.src.distill import PromptExample
from task3.src.evaluate import SpeculativeDecoder, gsm8k_extraction, score_standalone, timed_speculative
from task3.tests.test_alignment import FakeTokenizer
from task3.tests.test_cached_decoder import tiny_qwen


def _examples() -> list[PromptExample]:
    return [
        PromptExample("GSM8K-0000", "two plus two", "math", "#### 4"),
        PromptExample("GSM8K-0001", "three times", "math", "#### 9"),
        PromptExample("SHORT-familiar-0000", "month?", "familiar", "May"),
        PromptExample("SHORT-shifted-0000", "code?", "shifted", "PK"),
    ]


class PipelineCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = tiny_qwen(41, sharpness=3.0)
        cls.draft = tiny_qwen(42, sharpness=3.0)
        cls.tokenizer = FakeTokenizer()
        cls.examples = _examples()
        cls.report = _benchmark(cls.target, cls.draft, cls.tokenizer, cls.examples, label="t", max_new=12, ignore_eos=False, seed=6304)

    def test_benchmark_rows_and_speed_fields(self) -> None:
        report = self.report
        self.assertEqual(len(report["rows"]), 2 * len(self.examples) * 2)
        self.assertIn("first", report["warm_setup"])
        self.assertEqual(report["isolated"]["4"]["verify_tokens"], 5)
        self.assertEqual(set(report["isolated"]["2"]["per_prompt_length"]), {"32", "128", "512"})
        for row in report["rows"]:
            spec = row["speculative"]
            parts = sum(spec[key] for key in ("prefill_s", "draft_proposal_s", "target_verification_s", "acceptance_residual_s", "cache_rollback_s", "host_overhead_s"))
            self.assertAlmostEqual(parts, spec["elapsed_s"], places=6)
            self.assertEqual(spec["target_passes"], spec["rounds"])
            self.assertEqual(spec["draft_proposal_forwards"], spec["proposed"])
            self.assertEqual(spec["target_verify_input_tokens"], spec["proposed"] + spec["rounds"])
            self.assertLessEqual(spec["output_tokens"], 12)
            self.assertIn("resident_memory_bytes", spec)
            self.assertEqual(row["target_only"]["target_passes"], row["target_only"]["output_tokens"])
            self.assertEqual(row["order"][0], "target_only" if row["repetition"] == 0 else "speculative")
        for gamma in ("2", "4"):
            speed = report["speed"][gamma]
            self.assertGreaterEqual(speed["alpha"], speed["alpha_accepted_over_proposed"])
            self.assertIn("throughput_ratio_mean", speed["bootstrap"])
            self.assertIn("draft_proposal_s", speed["seconds_per_output_token"])
            self.assertEqual(speed["n_prompts"], len(self.examples))
        workload = workload_breakdown(report["rows"])
        self.assertEqual(sum(cell["n_rows"] for cell in workload["2"]["by_domain"].values()), 2 * len(self.examples))
        self.assertEqual(sum(cell["n_rows"] for cell in workload["4"]["by_output_length"].values()), 2 * len(self.examples))

    def test_standalone_records_and_summary(self) -> None:
        rows = _standalone(self.draft, self.tokenizer, self.examples[2:])
        for row in rows:
            self.assertLessEqual(row["n_tokens"], 32)
            for key in ("text", "correct", "correct_lenient", "format_compliant", "unextractable", "truncated", "repeated_4gram_fraction"):
                self.assertIn(key, row)
        summary = standalone_summary({"original": rows, "off_policy_seed6304": rows})
        self.assertEqual(summary["off_policy_seed6304"]["all"]["accuracy_minus_original"]["mean_diff"], 0.0)
        continuations = _continuations(self.tokenizer, self.examples, {"original": self.report}, {"original": rows})
        self.assertEqual(_continuation_ids(self.examples), ["GSM8K-0000", "GSM8K-0001", "SHORT-familiar-0000", "SHORT-shifted-0000"])
        self.assertIn("original", continuations["examples"][0]["speculative_gamma4_rep0_64"])

    def test_extractors(self) -> None:
        self.assertEqual(gsm8k_extraction("so the answer is \\boxed{1,024}. Done in 5 steps"), ("1024", "boxed"))
        scored = score_standalone("familiar", "Destination code: PK", "PK", length_capped=False)
        self.assertTrue(scored["correct"] and scored["format_compliant"])
        verbose = score_standalone("familiar", "The code provided is PK.\nMore text", "PK", length_capped=False)
        self.assertFalse(verbose["correct"])
        self.assertTrue(verbose["correct_lenient"])
        self.assertFalse(verbose["format_compliant"])

    def test_cost_wrapper_and_bank_tables(self) -> None:
        log = {"end_to_end_cost_s": 100.0, "teacher_scoring_s": 10.0, "backward_s": 5.0, "updates": [{"loss": 1.0}, {"loss": 0.5}], "optimizer_steps": 2}
        bench = {"original": self.report, "off_policy_seed6304": self.report}
        cost = deployment_break_even({"off_policy_seed6304": log}, {"off_policy_seed6304": {"merge_cost_s": 3.0}}, bench)
        entry = cost["off_policy_seed6304"]
        self.assertEqual(entry["upfront_seconds"], 103.0)
        self.assertAlmostEqual(entry["per_gamma"]["2"]["seconds_saved_per_request_vs_original_draft"], 0.0)
        self.assertIsNone(entry["per_gamma"]["2"]["requests_to_amortize_vs_original_draft"])
        self.assertAlmostEqual(entry["components_s"]["model_load_and_other"], 85.0)
        train = training_summary({"off_policy_seed6304": log, "off_policy_seed6305": dict(log, end_to_end_cost_s=120.0)})
        self.assertAlmostEqual(train["seed_dispersion"]["off_policy"]["end_to_end_cost_s"]["range"], 20.0)

        decoder = SpeculativeDecoder(self.target, self.draft, np.random.default_rng(0))
        twin = SpeculativeDecoder(self.target, self.draft, np.random.default_rng(0))
        prompt = self.tokenizer.apply_chat_template([{"content": "x"}])
        rows = [
            {
                "gamma": 2,
                "merged": timed_speculative(decoder, prompt, gamma=2, max_new_tokens=8, eos_id=None, seed=s),
                "unmerged": timed_speculative(twin, prompt, gamma=2, max_new_tokens=8, eos_id=None, seed=s),
            }
            for s in range(3)
        ]
        wrapper = wrapper_overhead_summary(rows)
        self.assertEqual(wrapper["2"]["same_tokens_fraction"], 1.0)
        self.assertIsNotNone(wrapper["2"]["draft_forward_overhead_ratio"])

        alignment = {
            "teacher": {
                "drafts": {
                    "original": {
                        "positions_1_64": {"expected_acceptance": 0.8},
                        "by_domain": {d: {"expected_acceptance": 0.7} for d in ("math", "familiar", "shifted")},
                    }
                }
            }
        }
        table = bank_vs_measured(alignment, {"original": self.report})
        self.assertEqual(table["original"]["bank_expected_acceptance"]["teacher"]["positions_1_64"], 0.8)
        self.assertIn("4", table["original"]["measured"])


if __name__ == "__main__":
    unittest.main()
