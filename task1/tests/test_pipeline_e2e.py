"""CPU end-to-end: every GPU stage on a tiny random Qwen2 built from config.

No download happens: ``load_qwen`` is replaced by a seeded tiny ``Qwen2ForCausalLM``
and the locked splits are small synthetic files under a temporary ``PA4_DATA_REPO``.
The stage functions, allocation rules, test-file gate, resume logic, resources
rebuild and analyze (bootstrap, figures) are the real ones.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from task1.src import data as data_module
from task1.src import pipeline
from task1.src.analyze import OPTIONAL, OPTIONAL_PAIRS, PAIRS, REQUIRED
from task1.src.constants import CALIBRATION_SEQUENCES, DEVELOPMENT_SEQUENCES, TEST_FILENAMES
from task1.src.run_profile import SMOKE_ENV, SMOKE_MARKER, SMOKE_RESOURCE_CONDITIONS

VOCAB = 256
LAYERS = 3
ALL_CONDITIONS = list(REQUIRED) + list(OPTIONAL)


class CharTokenizer:
    """Prefix-consistent character tokenizer: encode(a) is a prefix of encode(a + b)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1 + (ord(char) % (VOCAB - 1)) for char in text]


def tiny_qwen():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1234)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(torch.float16).eval()
    return model, CharTokenizer()


def _write_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_locked_data(root: Path) -> None:
    rng = np.random.default_rng(0)
    locked = root / "data" / "public_locked"

    def seqs(count: int, length: int) -> list[dict]:
        return [{"id": i, "token_ids": rng.integers(1, VOCAB, length).tolist()} for i in range(count)]

    def windows(count: int) -> list[dict]:
        rows = []
        for i in range(count):
            length = int(rng.integers(12, 30))
            rows.append({"id": i, "token_ids": rng.integers(1, VOCAB, length).tolist(),
                         "score_target_offsets": [1, length], "scored_tokens": length - 1})
        return rows

    words = ["walks", "runs to the [title] door", "opens a box", "sits down", "jumps high", "eats lunch"]
    hellaswag = []
    for order in range(6):
        endings = [" ".join(rng.choice(words, size=2)) for _ in range(4)]
        hellaswag.append({"order": order, "source_row_index": 100 + order, "example": {
            "activity_label": "Cooking", "ctx_a": "A man stands in a kitchen.", "ctx_b": "he",
            "endings": endings, "label": str(order % 4)}})
    _write_gz(locked / "c4_calibration_locked.jsonl.gz", seqs(CALIBRATION_SEQUENCES, 10))
    _write_gz(locked / "c4_development_locked.jsonl.gz", seqs(DEVELOPMENT_SEQUENCES, 10))
    _write_gz(locked / "wikitext2_test_windows_locked.jsonl.gz", windows(3))
    _write_gz(locked / "c4_validation_windows_locked.jsonl.gz", windows(2))
    _write_gz(locked / "hellaswag_val_locked.jsonl.gz", hellaswag)


def fast_latency(model, vocab_size, **kwargs):
    """The real measure_latencies at short lengths (the CPU test is not a timing study)."""
    from task1.src.resources import measure_latencies

    kwargs["prefill_lengths"] = (16, 32)
    kwargs["decode_tokens"] = min(int(kwargs.get("decode_tokens", 4)), 4)
    kwargs["repeats"] = 1
    kwargs["warmup"] = 0
    return measure_latencies(model, vocab_size, **kwargs)


class PipelineEndToEnd(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.data = self.tmp / "data-repo"
        make_locked_data(self.data)
        self.loads = 0
        self.test_reads: list[tuple[str, str, list[str]]] = []
        self.stage = ""

        def loader():
            self.loads += 1
            return tiny_qwen()

        real_read = data_module.read_jsonl

        def spy_read(path: Path):
            if path.name in TEST_FILENAMES:
                allocation = sorted(p.name for p in self.output.glob("allocation/*.json"))
                self.test_reads.append((self.stage, path.name, allocation))
            return real_read(path)

        env = {"PA4_DATA_REPO": str(self.data), "PA4_FORBID_DOWNLOAD": "1"}
        env_patch = mock.patch.dict(os.environ, env)
        env_patch.start()
        os.environ.pop(SMOKE_ENV, None)
        self.addCleanup(env_patch.stop)
        for target, value in (
            ("task1.src.pipeline.load_qwen", loader),
            ("task1.src.pipeline.measure_latencies", fast_latency),
            ("task1.src.data.read_jsonl", spy_read),
        ):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, stage: str, **kwargs) -> None:
        self.stage = stage
        getattr(pipeline, f"run_{stage}")(self.output, **kwargs)

    def _run_all(self, skip_latency: bool) -> None:
        self._run("baseline", microbatch=64, skip_latency=skip_latency)
        self._run("prune", microbatch=64, skip_latency=skip_latency)
        self._run("quantize", microbatch=64, skip_latency=skip_latency)
        self._run("quantize_q3m", microbatch=64, skip_latency=skip_latency)
        self._run("resources", microbatch=64, remeasure=False)
        self._run("analyze")

    def _check_outputs(self, smoke: bool) -> None:
        out = self.output
        for condition in ALL_CONDITIONS:
            folder = out / "conditions" / condition
            for filename in ("summary.json", "hellaswag.jsonl", "window_nll_wikitext.npz", "window_nll_c4.npz"):
                self.assertTrue((folder / filename).is_file(), f"{condition}/{filename}")
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["smoke"], smoke)
            self.assertEqual(summary["runtime"]["smoke"], smoke)
            self.assertEqual("smoke_profile" in summary, smoke)
            for key in ("condition", "family", "method", "packed_bytes", "dense_fp16_bytes", "recipe", "projections"):
                self.assertIn(key, summary, f"{condition} lacks {key}")
            for suite in ("wikitext", "c4_validation"):
                self.assertTrue(np.isfinite(summary["quality"][suite]["mean_nll"]), condition)
                self.assertGreater(summary["quality"][suite]["n_tokens"], 0)
            self.assertGreater(summary["quality"]["hellaswag"]["n_examples"], 0)
            self.assertEqual(summary["latency"]["prefill"]["16"]["status"], "ok", condition)
            if condition != "fp16":
                self.assertTrue(summary["projections"], condition)
                self.assertTrue(all(row.get("mse") is not None for row in summary["projections"]), condition)
        allocation = {path.name for path in (out / "allocation").glob("*.json")}
        self.assertTrue({
            "p3_tau.json", "p3_sensitivity.json", "q2_alpha_search.json", "q3_b4.json", "q3_b3.json",
            "q3_development_diagnostic.json", "q3m_b4.json", "q3m_b3.json", "q3m_development_diagnostic.json",
        } <= allocation, allocation)
        for label, bits in (("q3m_b4", 4), ("q3m_b3", 3)):
            frozen = json.loads((out / "allocation" / f"{label}.json").read_text(encoding="utf-8"))
            q2 = json.loads((out / "conditions" / f"q2_b{bits}" / "summary.json").read_text(encoding="utf-8"))
            q3m = json.loads((out / "conditions" / label / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(frozen["budget_bytes"], q2["packed_bytes"])
            self.assertLessEqual(q3m["packed_bytes"], q2["packed_bytes"])
        bootstrap = json.loads((out / "bootstrap.json").read_text(encoding="utf-8"))
        self.assertEqual(set(bootstrap), {label for *_pair, label in PAIRS + OPTIONAL_PAIRS})
        for report in bootstrap.values():
            self.assertEqual(report["smoke"], smoke)
            for key in ("hellaswag_accuracy", "hellaswag_acc_norm", "wikitext", "c4_validation"):
                self.assertIn(key, report)
        comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual({row["condition"] for row in comparison}, set(ALL_CONDITIONS))
        self.assertTrue((out / "qualitative_hellaswag.json").is_file())
        figures = {path.name for path in (out / "figures").glob("*.png")}
        self.assertTrue({
            "pruning_quality_vs_sparsity.png", "quant_quality_vs_bytes.png", "p3_sensitivity.png",
            "p3_sparsity_s50.png", "p3_sparsity_s75.png", "q3_sensitivity.png", "q3_bits_int4_budget.png",
            "q3_bits_int3_budget.png", "q3m_bits_int4_budget.png", "q3m_bits_int3_budget.png", "q3m_sensitivity.png",
        } <= figures, figures)
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))["conditions"]
        self.assertEqual(set(index), set(ALL_CONDITIONS))

    def _check_gate(self) -> None:
        """No test file is read by a selection stage before its allocation is frozen."""
        needs = {"prune": "p3_tau.json", "quantize": "q3_b3.json", "quantize_q3m": "q3m_b3.json"}
        for stage, name, allocation in self.test_reads:
            if stage in needs:
                self.assertIn(needs[stage], allocation, f"{stage} read {name} before the freeze")
        self.assertTrue(any(stage == "prune" for stage, *_rest in self.test_reads))

    def test_full_profile_end_to_end_then_resume(self) -> None:
        self.output = self.tmp / "results"
        self._run_all(skip_latency=True)  # resources then rebuilds all 20 conditions from recipes
        self._check_outputs(smoke=False)
        self._check_gate()
        q3m_alloc = json.loads((self.output / "allocation" / "q3m_b4.json").read_text(encoding="utf-8"))
        alpha_file = json.loads((self.output / "allocation" / "q2_alpha_search.json").read_text(encoding="utf-8"))
        self.assertEqual(q3m_alloc["alphas"], alpha_file["bits"]["4"]["alphas"])
        census = json.loads((self.output / "conditions" / "fp16" / "summary.json").read_text(encoding="utf-8"))["census"]
        self.assertEqual(census["n_eligible_matrices"], LAYERS * 7)
        # Resume on a finished directory: no stage reloads the model or rewrites a summary.
        loads = self.loads
        before = {p: p.stat().st_mtime_ns for p in self.output.glob("conditions/*/summary.json")}
        for stage in ("baseline", "prune", "quantize", "quantize_q3m", "resources"):
            self._run(stage)
        self.assertEqual(self.loads, loads)
        after = {p: p.stat().st_mtime_ns for p in self.output.glob("conditions/*/summary.json")}
        self.assertEqual(before, after)
        # A real directory refuses smoke stages.
        with mock.patch.dict(os.environ, {SMOKE_ENV: "1"}):
            with self.assertRaises(SystemExit):
                pipeline.run_prune(self.output)

    def test_smoke_profile_end_to_end(self) -> None:
        self.output = self.tmp / "smoke"
        with mock.patch.dict(os.environ, {SMOKE_ENV: "1"}):
            self._run_all(skip_latency=False)
            self._run("resources", microbatch=64, remeasure=True, only=list(SMOKE_RESOURCE_CONDITIONS))
            self._check_outputs(smoke=True)
        self._check_gate()
        self.assertTrue((self.output / SMOKE_MARKER).is_file())
        census = json.loads((self.output / "conditions" / "fp16" / "summary.json").read_text(encoding="utf-8"))["census"]
        self.assertEqual(census["n_eligible_matrices"], 7)  # last block only
        layers = {row["layer"] for row in json.loads(
            (self.output / "conditions" / "p3_s50" / "summary.json").read_text(encoding="utf-8"))["projections"]}
        self.assertEqual(layers, {LAYERS - 1})
        # Real stages refuse a smoke directory, before loading anything.
        loads = self.loads
        with self.assertRaises(SystemExit):
            pipeline.run_quantize(self.output)
        with self.assertRaises(SystemExit):
            pipeline.run_analyze(self.output)
        self.assertEqual(self.loads, loads)

    def test_missing_test_file_fails_before_model_load(self) -> None:
        self.output = self.tmp / "results"
        (self.data / "data" / "public_locked" / "hellaswag_val_locked.jsonl.gz").unlink()
        for stage in ("prune", "quantize"):
            with self.assertRaises(SystemExit):
                getattr(pipeline, f"run_{stage}")(self.output)
        self.assertEqual(self.loads, 0)

    def test_q3m_without_quantize_fails_fast(self) -> None:
        self.output = self.tmp / "results"
        with self.assertRaises(SystemExit):
            pipeline.run_quantize_q3m(self.output)
        self.assertEqual(self.loads, 0)


if __name__ == "__main__":
    unittest.main()
