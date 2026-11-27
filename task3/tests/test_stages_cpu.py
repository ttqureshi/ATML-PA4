"""Drive the real baseline/align/eval/analyze stage functions on CPU with tiny models.

Loaders, data paths and output directories are patched; caps are shrunk so the
run takes seconds. This catches wiring errors (cache parts, resume, outputs,
analysis inputs) before a GPU session is spent on them. No weights are downloaded.
"""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from task3 import run
from task3.src import distill
from task3.src.distill import PromptExample, build_lora_config
from task3.tests.test_alignment import FakeTokenizer, _supplied_records
from task3.tests.test_cached_decoder import tiny_qwen


def _pool() -> list[PromptExample]:
    import numpy as np

    rng = np.random.default_rng(0)
    records = _supplied_records(rng)
    texts = {record["prompt_id"]: record["prompt"] for record in records}
    return [
        PromptExample("GSM8K-0000", texts["GSM8K-0000"], "math", "#### 4"),
        PromptExample("GSM8K-0001", "more math", "math", "#### 7"),
        PromptExample("SHORT-familiar-0000", texts["SHORT-familiar-0000"], "familiar", "May"),
        PromptExample("SHORT-shifted-0003", texts["SHORT-shifted-0003"], "shifted", "PK"),
    ]


class StagesCpuTests(unittest.TestCase):
    def test_baseline_align_eval_analyze(self) -> None:
        import numpy as np
        from peft import get_peft_model

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        results = root / "results"
        checkpoints = root / "checkpoints"
        bank = root / "prefix_banks_locked.jsonl.gz"
        with gzip.open(bank, "wt", encoding="utf-8") as handle:
            for record in _supplied_records(np.random.default_rng(0)):
                handle.write(json.dumps(record) + "\n")
        for method in distill.METHODS:
            for seed in distill.TRAIN_SEEDS:
                (checkpoints / method / f"seed{seed}" / "merged").mkdir(parents=True)
        get_peft_model(tiny_qwen(42), build_lora_config()).save_pretrained(checkpoints / "off_policy" / "seed6304" / "adapter")

        tokenizer = FakeTokenizer()
        target = tiny_qwen(41)
        drafts = {name: tiny_qwen(42 + index) for index, name in enumerate(run._draft_names())}
        patches = [
            mock.patch.object(run, "require_cuda", lambda stage: None),
            mock.patch.object(run, "results_dir", lambda: results),
            mock.patch.object(run, "checkpoint_dir", lambda: checkpoints),
            mock.patch.object(run, "_load_target", lambda: (target, tokenizer)),
            mock.patch.object(run, "_load_draft", lambda name, tok: drafts[name]),
            mock.patch.object(run, "load_eval_pool", _pool),
            mock.patch.object(run, "default_data_paths", lambda: {"prefix_banks": bank}),
            mock.patch.object(run, "SPEED_MAX_NEW", 6),
            mock.patch.object(run, "DIAGNOSTIC_MAX_NEW", 6),
            mock.patch.object(run, "diagnostic_math_prompts", lambda examples: examples[:2]),
            mock.patch.object(run, "answer_cap", lambda domain: 6),
            mock.patch.object(run, "estimate_speed_model", lambda d, t, g: {"gamma": g, "c": 0.8, "v_gamma": 1.1, "verify_tokens": g + 1}),
            mock.patch.object(distill, "load_frozen_causal", lambda name, dtype=None: (tiny_qwen(42), tokenizer)),
            mock.patch.object(distill, "distill_results_dir", lambda method, seed: results / "distill" / f"{method}_seed{seed}"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        # A part from an older code version is recomputed, not reused.
        stale = results / "baseline_parts" / "primary.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(json.dumps({"value": {"stale": True}}), encoding="utf-8")
        run.stage_baseline()
        self.assertNotIn("stale", json.loads((results / "baseline.json").read_text(encoding="utf-8"))["primary"])
        run.stage_align()
        # A second call resumes entirely from the cached parts.
        run.stage_align()
        run.stage_eval()
        for method in distill.METHODS:
            for seed in distill.TRAIN_SEEDS:
                (results / "distill" / f"{method}_seed{seed}").mkdir(parents=True)
                (results / "distill" / f"{method}_seed{seed}" / "train_log.json").write_text(
                    json.dumps({"end_to_end_cost_s": 50.0, "updates": [{"loss": 1.0}], "optimizer_steps": 1}), encoding="utf-8"
                )
                run.merge_report_path(method, seed).parent.mkdir(parents=True)
                run.merge_report_path(method, seed).write_text(json.dumps({"merge_cost_s": 2.0}), encoding="utf-8")
        run.stage_analyze()

        alignment = json.loads((results / "alignment.json").read_text(encoding="utf-8"))
        self.assertEqual(alignment["n_supplied_prefixes"], len(_supplied_records(np.random.default_rng(0))))
        banks = set(alignment["summary"])
        self.assertTrue({"teacher", "original_draft", "exact_spec_original_gamma4"} <= banks)
        self.assertEqual(sum(1 for bank in banks if bank.startswith("student_")), 4)
        for table in alignment["summary"].values():
            self.assertEqual(set(table["drafts"]), set(run._draft_names()))
            self.assertTrue(table["same_prefix_ids_for_every_draft"])
        self.assertIn("overall", alignment["target_vs_speculative_consistency"])
        self.assertTrue(alignment["aligned_disagreements"]["examples"])

        evaluation = json.loads((results / "eval.json").read_text(encoding="utf-8"))
        self.assertEqual(set(evaluation["benchmarks"]), set(run._draft_names()))
        self.assertEqual(set(evaluation["diagnostic_ignore_eos"]), set(run._draft_names()))
        self.assertTrue(all(row["speculative"]["output_tokens"] == 6 for row in evaluation["diagnostic_ignore_eos"]["original"]["rows"]))
        self.assertEqual(evaluation["wrapper_overhead"]["run"], "off_policy_seed6304")
        self.assertTrue(evaluation["continuations"]["examples"])

        analysis = json.loads((results / "analyze.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["missing"], ["smoke"])
        for key in ("baseline_speed", "eval_speed", "eval_workload", "standalone", "wrapper_overhead", "bank_vs_measured_acceptance", "deployment_break_even", "training", "alignment_overall"):
            self.assertIn(key, analysis)


if __name__ == "__main__":
    unittest.main()
