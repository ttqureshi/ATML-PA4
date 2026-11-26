"""Smoke must refuse the results directory and must not load Qwen."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task1.src.data import TestFileGate
from task1.src.smoke import assert_smoke_output, run_smoke


ROOT = Path(__file__).resolve().parents[2]


class SmokeTests(unittest.TestCase):
    def test_smoke_refuses_final_results_directory(self) -> None:
        for relative in (Path("results"), Path("task1") / "results", Path("task1") / "results" / "smoke"):
            with self.assertRaises(SystemExit) as caught:
                assert_smoke_output(ROOT / relative, ROOT)
            self.assertIn("final results", str(caught.exception).lower())

    def test_gate_blocks_test_files_until_freeze(self) -> None:
        gate = TestFileGate()
        with self.assertRaises(RuntimeError):
            gate.check(Path("wikitext2_test_windows_locked.jsonl.gz"))
        gate.freeze("tau chosen")
        gate.check(Path("wikitext2_test_windows_locked.jsonl.gz"))

    def test_smoke_runs_without_downloading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "smoke"
            run_smoke(output, ROOT)
            summary = (output / "summary.json").read_text(encoding="utf-8")
            self.assertIn('"downloaded_weights": false', summary)
            self.assertIn('"logits_finite": true', summary)


if __name__ == "__main__":
    unittest.main()
