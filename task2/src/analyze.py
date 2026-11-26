"""Aggregate saved Task 2 measurements. Missing GPU runs are listed, not invented."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared.bootstrap import bootstrap_paired_mean_diff
from task2.src.kv_eval import METHODS
from task2.src.theory import RECEPTIVE_FIELD_NOTE, theoretical_report
from task2.src.train import JOBS, SEEDS, results_dir

PRINCIPAL_PAIRS = (
    ("sink_recent", "fp16"),
    ("question", "fp16"),
    ("int4_token", "fp16"),
    ("int4_channel", "fp16"),
    ("hybrid", "fp16"),
    ("hybrid", "question"),
    ("hybrid", "int4_channel"),
)


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _paired(records: list[dict], left: str, right: str) -> dict:
    by_id = {}
    for row in records:
        by_id.setdefault(row["id"], {})[row["method"]] = row
    ids = sorted(item for item, methods in by_id.items() if left in methods and right in methods)
    a = np.array([by_id[item][left]["exact"] for item in ids], dtype=np.float64)
    b = np.array([by_id[item][right]["exact"] for item in ids], dtype=np.float64)
    summary = bootstrap_paired_mean_diff(a, b, n_resamples=2000, seed=6304)
    summary["left"] = left
    summary["right"] = right
    summary["contrast"] = f"{left} - {right}"
    return summary


def _conditional(records: list[dict], method: str) -> dict:
    rows = [row for row in records if row["method"] == method]
    if not rows:
        return {"n": 0}
    unconditional = sum(row["exact"] for row in rows) / len(rows)
    conditioned = [row for row in rows if row.get("reference_exact") == 1]
    conditional = None if not conditioned else sum(row["exact"] for row in conditioned) / len(conditioned)
    return {
        "n": len(rows),
        "unconditional_accuracy": unconditional,
        "n_reference_correct": len(conditioned),
        "conditional_accuracy": conditional,
    }


def kv_summary(records: list[dict]) -> dict:
    cells = {}
    for row in records:
        if row["method"] != "fp16":
            continue
        key = f"{row['target_tokens']}|{row['position']}|{row['distractors']}"
        cells.setdefault(key, []).append(row["exact"])
    cell_accuracy = {key: sum(values) / len(values) for key, values in cells.items()}
    return {
        "full_cache_by_cell": cell_accuracy,
        "cell_note": "Each factorial cell has 10 prompts and is descriptive only.",
        "methods": {method: _conditional(records, method) for method in METHODS},
        "paired_bootstrap": [_paired(records, left, right) for left, right in PRINCIPAL_PAIRS],
    }


def missing_gpu_work() -> list[str]:
    root = results_dir()
    missing = []
    if not (root / "attention.json").is_file():
        missing.append(
            "attention: dense, tiled, and SDPA at L in {256,512,1024,2048,4096,8192}, "
            "FP16 and FP32, ten warmed-up repetitions, with dense skipped at L=8192"
        )
    curve_dir = root / "learning_curves"
    expected = len(JOBS) * len(SEEDS)
    found = 0 if not curve_dir.is_dir() else len(list(curve_dir.glob("*.json")))
    if found < expected:
        missing.append(
            f"train: {expected} MQAR runs (14 architectures × seeds 6304, 6305, 6306), "
            f"{found} learning curves saved; then test-set evaluation and the test-time window control"
        )
    if not (root / "forward_sweep.json").is_file():
        missing.append("forward sweep: T in {256,1024,4096,8192} for each attention operator, no training")
    if not (root / "kv_records.json").is_file():
        missing.append(
            "kv: Qwen2.5-1.5B-Instruct FP16 greedy on all 180 retrieval prompts, "
            "five compressed methods plus the FP16 reference"
        )
    if not (root / "kv_pack_benchmark.json").is_file():
        missing.append(
            "kv pack benchmark: batch 1 and 8, context 2048 and 4096, "
            "ten warmed-up repetitions of pack, unpack, and decode-step latency"
        )
    return missing


def analyze() -> dict:
    root = results_dir()
    report = {
        "receptive_field_note": RECEPTIVE_FIELD_NOTE,
        "theory_T256": theoretical_report(256),
        "missing_gpu_stages": missing_gpu_work(),
    }
    attention = _load(root / "attention.json")
    if attention is not None:
        report["attention_rows"] = len(attention)
    kv = _load(root / "kv_records.json")
    if kv is not None:
        report["kv"] = kv_summary(kv)
    sweep = _load(root / "forward_sweep.json")
    if sweep is not None:
        report["forward_sweep_rows"] = len(sweep.get("rows", []))
    out = root / "analysis.json"
    if root.exists():
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
