"""Drive the real stage functions on CPU with tiny models (no downloads).

`test_full_stage_sequence_smoke_mode` runs prefix_banks -> smoke -> baseline ->
distill -> merge -> align -> eval -> analyze exactly as `run.py --smoke` would,
against a miniature course data repo in the real file layout; only the model
and tokenizer loaders, CUDA checks and output roots are patched. The smoke
analyze stage fails unless every code path left its evidence.

`test_baseline_align_eval_analyze` runs the full-mode (non-smoke) stages that
read GPU outputs, with resume from cached parts and a stale part.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from task3 import run
from task3.src import distill, runtime
from task3.src.distill import PromptExample, build_lora_config
from task3.tests.test_alignment import FakeTokenizer, _supplied_records
from task3.tests.test_cached_decoder import VOCAB, tiny_qwen
from task3.tests.test_prefix_banks import qwen_like, write_course_data


class CharTokenizer(FakeTokenizer):
    """One token per character (up to 50), so real probe prompts are long enough for the split-prefill check."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **_kwargs):
        return [1, 4] + self.encode(messages[0]["content"][:50]) + [3]

    def encode(self, text):
        return [5 + (ord(ch) % (VOCAB - 6)) for ch in text]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "fake_tokenizer.json").write_text("{}", encoding="utf-8")


def _tiny_loader(name, dtype=None):
    from transformers import Qwen2ForCausalLM

    if Path(name).is_dir():
        return Qwen2ForCausalLM.from_pretrained(name).eval(), CharTokenizer()
    return qwen_like({distill.TARGET_MODEL: 41, distill.DRAFT_MODEL: 42}[name]), CharTokenizer()


def _fake_isolated(draft, target, gamma):
    return {"gamma": gamma, "c": 0.8, "v_gamma": 1.1, "verify_tokens": gamma + 1}


def _start(case: unittest.TestCase, patches) -> None:
    for patch in patches:
        patch.start()
        case.addCleanup(patch.stop)


class FullPipelineCpuTests(unittest.TestCase):
    def test_full_stage_sequence_smoke_mode(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        data = root / "data_repo"
        write_course_data(data, n_math=3, n_familiar=2, n_shifted=2, n_train_math=4, n_train_short=4)
        results = root / "results_smoke"
        checkpoints = root / "checkpoints_smoke"
        patches = [
            mock.patch.dict(os.environ, {"PA4_DATA_REPO": str(data), runtime.SMOKE_ENV: "1"}),
            *(mock.patch.object(module, name, value) for module, name, value in run.smoke_overrides()),
            mock.patch.object(runtime, "results_dir", lambda: results),
            mock.patch.object(runtime, "checkpoint_dir", lambda: checkpoints),
            mock.patch.object(run, "require_cuda", lambda stage: None),
            mock.patch.object(run, "EXPECTED_EVAL_COUNTS", {"math": 3, "familiar": 2, "shifted": 2}),
            mock.patch.object(run, "EXPECTED_TRAIN_COUNTS", {"math": 4, "short": 4}),
            mock.patch.object(run, "estimate_speed_model", _fake_isolated),
            mock.patch.object(run, "_load_bank_tokenizers", lambda revisions: (CharTokenizer(), CharTokenizer())),
            mock.patch.object(run, "_load_bank_model", lambda key, revisions: qwen_like(41 if key == "model_target" else 42)),
            mock.patch.object(distill, "load_frozen_causal", _tiny_loader),
            mock.patch.object(distill, "_require_cuda", lambda action: None),
        ]
        _start(self, patches)

        for stage in ("prefix_banks", "smoke", "baseline", "distill", "merge", "align", "eval", "analyze"):
            run.STAGES[stage]()

        analysis = json.loads((results / "analyze.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["missing"], [])
        self.assertTrue(analysis["smoke_coverage"] and all(analysis["smoke_coverage"].values()), analysis["smoke_coverage"])
        self.assertIn(runtime.SMOKE_MARKER, analysis)
        self.assertFalse(checkpoints.exists(), "smoke checkpoints are removed after a passing smoke analyze")
        # Every JSON object the smoke run wrote is marked, and nothing went to the real bank path.
        for path in results.rglob("*.json"):
            self.assertIn(runtime.SMOKE_MARKER, json.loads(path.read_text(encoding="utf-8")), path)
        self.assertFalse((data / "data" / "prefix_banks").exists())

        bank = results / "prefix_banks" / "prefix_banks_locked.jsonl.gz"
        with gzip.open(bank, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual({record["prompt_id"] for record in records}, {"GSM8K-0000", "GSM8K-0001", "SHORT-familiar-0000", "SHORT-shifted-0000"})
        seeds = {(r["bank"], r["prompt_id"]): r["source_generation_seed"] for r in records}
        # Full-list index j is kept, so smoke rows use the same seeds as the full bank.
        self.assertEqual(seeds[("teacher", "SHORT-shifted-0000")], 6304 + 10007 * 5)
        meta = json.loads((results / "prefix_banks_build.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["sampling"]["teacher"]["repetition_penalty"], 1.1)
        self.assertLess(meta["generate_equivalence_check"]["teacher"]["max_abs_prob_gap"], 1e-5)

        evaluation = json.loads((results / "eval.json").read_text(encoding="utf-8"))
        self.assertEqual(set(evaluation["benchmarks"]), set(run._draft_names()))
        self.assertEqual(len(evaluation["benchmarks"]["original"]["rows"]), 4 * 2 * 2)
        self.assertTrue(all(row["speculative"]["output_tokens"] == 24 for row in evaluation["diagnostic_ignore_eos"]["on_policy_seed6305"]["rows"]))
        alignment = json.loads((results / "alignment.json").read_text(encoding="utf-8"))
        self.assertIn("teacher_no_rep_penalty", alignment["summary"])
        self.assertIn("overall", alignment["target_law_control_consistency"])
        for name, log in analysis["training"]["runs"].items():
            self.assertEqual(log["teacher_scored_tokens"], 320 + 64, name)

        # Rerunning resumes: the bank is validated and skipped, cached parts are reused.
        run.STAGES["prefix_banks"]()
        run.STAGES["baseline"]()


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
        # Both supplied banks must cover every evaluation prompt.
        records = _supplied_records(np.random.default_rng(0))
        extra = [dict(record, prompt_id="GSM8K-0001", prompt="more math") for record in records if record["prompt_id"] == "GSM8K-0000"]
        with gzip.open(bank, "wt", encoding="utf-8") as handle:
            for record in records + extra:
                handle.write(json.dumps(record) + "\n")
        for method in distill.METHODS:
            for seed in distill.TRAIN_SEEDS:
                merged = checkpoints / method / f"seed{seed}" / "merged"
                merged.mkdir(parents=True)
                for name in ("merge_error.json", "config.json", "model.safetensors"):
                    (merged / name).write_text("{}", encoding="utf-8")
        wrapper = checkpoints / "off_policy" / "seed6304"
        get_peft_model(tiny_qwen(42), build_lora_config()).save_pretrained(wrapper / "adapter")
        (wrapper / "train_log.json").write_text("{}", encoding="utf-8")

        tokenizer = FakeTokenizer()
        target = tiny_qwen(41)
        drafts = {name: tiny_qwen(42 + index) for index, name in enumerate(run._draft_names())}
        patches = [
            mock.patch.dict(os.environ, {runtime.SMOKE_ENV: ""}),
            mock.patch.object(run, "require_cuda", lambda stage: None),
            mock.patch.object(runtime, "results_dir", lambda: results),
            mock.patch.object(runtime, "checkpoint_dir", lambda: checkpoints),
            mock.patch.object(run, "_load_target", lambda: (target, tokenizer)),
            mock.patch.object(run, "_load_draft", lambda name, tok: drafts[name]),
            mock.patch.object(run, "load_eval_pool", _pool),
            mock.patch.object(run, "EXPECTED_EVAL_COUNTS", {"math": 2, "familiar": 1, "shifted": 1}),
            mock.patch.object(run, "default_data_paths", lambda: {"prefix_banks": bank}),
            mock.patch.object(run, "SPEED_MAX_NEW", 6),
            mock.patch.object(run, "DIAGNOSTIC_MAX_NEW", 6),
            mock.patch.object(run, "DIAGNOSTIC_PROMPTS", 2),
            mock.patch.object(run, "answer_cap", lambda domain: 6),
            mock.patch.object(run, "estimate_speed_model", _fake_isolated),
            mock.patch.object(distill, "load_frozen_causal", lambda name, dtype=None: (tiny_qwen(42), tokenizer)),
        ]
        _start(self, patches)

        # A part from an older code version, and a truncated part, are recomputed, not reused.
        stale = results / "baseline_parts" / "primary.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(json.dumps({"value": {"stale": True}}), encoding="utf-8")
        (results / "baseline_parts" / "math_diagnostic_64_ignore_eos.json").write_text('{"value": {"rows": [', encoding="utf-8")
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
        self.assertEqual(alignment["n_supplied_prefixes"], len(records) + len(extra))
        banks = set(alignment["summary"])
        self.assertTrue({"teacher", "original_draft", "exact_spec_original_gamma4", "teacher_no_rep_penalty"} <= banks)
        self.assertEqual(sum(1 for bank in banks if bank.startswith("student_")), 4)
        for table in alignment["summary"].values():
            self.assertEqual(set(table["drafts"]), set(run._draft_names()))
            self.assertTrue(table["same_prefix_ids_for_every_draft"])
        self.assertIn("overall", alignment["target_vs_speculative_consistency"])
        self.assertTrue(alignment["aligned_disagreements"]["examples"])
        self.assertNotIn(runtime.SMOKE_MARKER, alignment)

        evaluation = json.loads((results / "eval.json").read_text(encoding="utf-8"))
        self.assertEqual(set(evaluation["benchmarks"]), set(run._draft_names()))
        self.assertEqual(set(evaluation["diagnostic_ignore_eos"]), set(run._draft_names()))
        self.assertTrue(all(row["speculative"]["output_tokens"] == 6 for row in evaluation["diagnostic_ignore_eos"]["original"]["rows"]))
        self.assertEqual(evaluation["wrapper_overhead"]["run"], "off_policy_seed6304")
        self.assertTrue(evaluation["continuations"]["examples"])

        analysis = json.loads((results / "analyze.json").read_text(encoding="utf-8"))
        self.assertEqual(analysis["missing"], ["prefix_banks", "smoke"])
        self.assertEqual(analysis["gpu_stages_remaining"], ["prefix_banks", "smoke", "analyze"])
        for key in ("baseline_speed", "eval_speed", "eval_workload", "standalone", "wrapper_overhead", "bank_vs_measured_acceptance", "deployment_break_even", "training", "alignment_overall", "target_law_control_consistency"):
            self.assertIn(key, analysis)

        # A smoke output never enters a full analysis.
        (results / "smoke.json").write_text(json.dumps({runtime.SMOKE_MARKER: "x", "passed": True}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            run.stage_analyze()

    def test_inputs_are_checked_before_any_model_load(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        data = root / "data_repo"
        write_course_data(data, n_math=3, n_familiar=2, n_shifted=2, n_train_math=4, n_train_short=4)

        def no_load(*_args, **_kwargs):
            raise AssertionError("a model was loaded before the inputs were checked")

        patches = [
            mock.patch.dict(os.environ, {"PA4_DATA_REPO": str(data), runtime.SMOKE_ENV: ""}),
            mock.patch.object(runtime, "results_dir", lambda: root / "results"),
            mock.patch.object(runtime, "checkpoint_dir", lambda: root / "checkpoints"),
            mock.patch.object(run, "require_cuda", lambda stage: None),
            mock.patch.object(run, "_load_target", no_load),
            mock.patch.object(run, "_load_bank_model", no_load),
            mock.patch.object(distill, "load_frozen_causal", no_load),
        ]
        _start(self, patches)
        # The real counts (60/30/30, 500/300) do not match the miniature data.
        for stage in ("prefix_banks", "baseline", "distill", "align", "eval"):
            with self.assertRaises(SystemExit, msg=stage):
                run.STAGES[stage]()
        with mock.patch.object(run, "EXPECTED_EVAL_COUNTS", {"math": 3, "familiar": 2, "shifted": 2}):
            # No prefix bank and no merged drafts yet.
            with self.assertRaises(SystemExit):
                run.stage_align()
            with self.assertRaises(SystemExit):
                run.stage_eval()
            with self.assertRaises(SystemExit):
                run.stage_merge()


if __name__ == "__main__":
    unittest.main()
