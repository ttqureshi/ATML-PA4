"""Plots, paired bootstrap intervals, and a few matched HellaSwag items.

Perplexity intervals come from ``shared.bootstrap.bootstrap_pooled_nll_diff``:
windows are resampled together, pooled NLL is recomputed, and the point
perplexities are the exponentials of those pooled means. The interval itself
is the helper's interval on the pooled NLL difference. HellaSwag intervals
resample example scores with ``bootstrap_paired_mean_diff``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.bootstrap import bootstrap_paired_mean_diff, bootstrap_pooled_nll_diff
from task1.src.constants import ELIGIBLE_PROJECTIONS
from task1.src.data import TestFileGate, load_locked, locked_path, write_json
from task1.src.hellaswag import hellaswag_prompt

REQUIRED = (
    "fp16",
    "p1_s25",
    "p1_s50",
    "p1_s75",
    "p2_s25",
    "p2_s50",
    "p2_s75",
    "p2_2to4",
    "p3_s50",
    "p3_s75",
    "q1_b8",
    "q1_b4",
    "q1_b3",
    "q2_b8",
    "q2_b4",
    "q2_b3",
    "q3_b4",
    "q3_b3",
)

PAIRS = (
    ("p3_s50", "p2_s50", "P3-P2 at 50% sparsity"),
    ("p3_s75", "p2_s75", "P3-P2 at 75% sparsity"),
    ("q3_b4", "q2_b4", "Q3-Q2 at the uniform INT4 byte budget"),
    ("q3_b3", "q2_b3", "Q3-Q2 at the uniform INT3 byte budget"),
)


def _load_summary(output: Path, condition: str) -> dict:
    path = output / "conditions" / condition / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _heatmap(records: list[dict], field: str, path: Path, title: str) -> None:
    layers = sorted({int(row["layer"]) for row in records})
    grid = np.full((len(ELIGIBLE_PROJECTIONS), len(layers)), np.nan)
    for row in records:
        if row.get(field) is None:
            continue
        grid[ELIGIBLE_PROJECTIONS.index(row["projection"]), layers.index(int(row["layer"]))] = float(row[field])
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.28), 3.6))
    image = ax.imshow(grid, aspect="auto")
    ax.set_yticks(range(len(ELIGIBLE_PROJECTIONS)), ELIGIBLE_PROJECTIONS)
    ax.set_xticks(range(len(layers)), [str(layer) for layer in layers], rotation=90, fontsize=7)
    ax.set_xlabel("layer")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _series(summaries: dict[str, dict], conditions: list[str], x_key: str, metric: str):
    xs = []
    ys = []
    for condition in conditions:
        summary = summaries[condition]
        if x_key == "sparsity":
            xs.append(float(summary["achieved_sparsity"]))
        elif x_key == "bytes":
            xs.append(float(summary["packed_bytes"]))
        else:
            raise KeyError(x_key)
        quality = summary["quality"]
        if metric == "wikitext_ppl":
            ys.append(float(quality["wikitext"]["perplexity"]))
        elif metric == "c4_ppl":
            ys.append(float(quality["c4_validation"]["perplexity"]))
        elif metric == "hellaswag_acc":
            ys.append(float(quality["hellaswag"]["accuracy"]))
        elif metric == "hellaswag_acc_norm":
            ys.append(float(quality["hellaswag"]["acc_norm"]))
        else:
            raise KeyError(metric)
    return xs, ys


def _quality_figure(summaries, path: Path, panels: list[tuple[str, str, str]]) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (x_key, metric, title) in zip(axes, panels):
        for label, conditions, marker in (
            ("P1 magnitude", ["p1_s25", "p1_s50", "p1_s75"], "o"),
            ("P2 Wanda", ["p2_s25", "p2_s50", "p2_s75"], "s"),
            ("P3 nonuniform", ["p3_s50", "p3_s75"], "D"),
        ):
            if conditions[0] not in summaries or "achieved_sparsity" not in summaries[conditions[0]] and x_key == "sparsity":
                continue
            if x_key == "bytes":
                continue
            xs, ys = _series(summaries, conditions, x_key, metric)
            ax.plot(xs, ys, marker=marker, label=label)
        if x_key == "sparsity" and "p2_2to4" in summaries:
            xs, ys = _series(summaries, ["p2_2to4"], x_key, metric)
            ax.scatter(xs, ys, marker="x", s=80, label="Wanda 2:4")
        if x_key == "bytes":
            for label, conditions, marker in (
                ("Q1", ["q1_b8", "q1_b4", "q1_b3"], "o"),
                ("Q2", ["q2_b8", "q2_b4", "q2_b3"], "s"),
                ("Q3", ["q3_b4", "q3_b3"], "D"),
            ):
                xs, ys = _series(summaries, conditions, x_key, metric)
                ax.plot(xs, ys, marker=marker, label=label)
            if "fp16" in summaries:
                ax.scatter(
                    [summaries["fp16"]["packed_bytes"]],
                    [_metric(summaries["fp16"], metric)],
                    marker="*",
                    s=120,
                    label="FP16",
                )
        ax.set_title(title)
        ax.set_xlabel("achieved eligible sparsity" if x_key == "sparsity" else "packed bytes")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _metric(summary: dict, metric: str) -> float:
    quality = summary["quality"]
    if metric == "wikitext_ppl":
        return float(quality["wikitext"]["perplexity"])
    if metric == "c4_ppl":
        return float(quality["c4_validation"]["perplexity"])
    if metric == "hellaswag_acc":
        return float(quality["hellaswag"]["accuracy"])
    if metric == "hellaswag_acc_norm":
        return float(quality["hellaswag"]["acc_norm"])
    raise KeyError(metric)


def _hellaswag_vectors(output: Path, condition: str) -> dict[int, dict]:
    path = output / "conditions" / condition / "hellaswag.jsonl"
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["order"])] = row
    return rows


def _paired(output: Path, left: str, right: str) -> dict:
    a_rows = _hellaswag_vectors(output, left)
    b_rows = _hellaswag_vectors(output, right)
    orders = sorted(set(a_rows) & set(b_rows))
    if not orders:
        raise RuntimeError(f"no overlapping HellaSwag orders for {left} vs {right}")
    correct_a = np.asarray([a_rows[order]["correct"] for order in orders], dtype=np.float64)
    correct_b = np.asarray([b_rows[order]["correct"] for order in orders], dtype=np.float64)
    norm_a = np.asarray([a_rows[order]["correct_norm"] for order in orders], dtype=np.float64)
    norm_b = np.asarray([b_rows[order]["correct_norm"] for order in orders], dtype=np.float64)
    report = {
        "a": left,
        "b": right,
        "difference": "a-b",
        "n_hellaswag": len(orders),
        "hellaswag_accuracy": bootstrap_paired_mean_diff(correct_a, correct_b),
        "hellaswag_acc_norm": bootstrap_paired_mean_diff(norm_a, norm_b),
    }
    for suite, filename in (("wikitext", "window_nll_wikitext.npz"), ("c4_validation", "window_nll_c4.npz")):
        a_file = np.load(output / "conditions" / left / filename)
        b_file = np.load(output / "conditions" / right / filename)
        report[suite] = bootstrap_pooled_nll_diff(
            a_file["nll_sums"], a_file["counts"], b_file["nll_sums"], b_file["counts"]
        )
    return report


def _qualitative(output: Path, summaries: dict[str, dict]) -> list[dict]:
    gate = TestFileGate()
    gate.freeze("analyze reads finished results and does not choose an allocation")
    text = {}
    if locked_path("hellaswag").is_file():
        for row in load_locked("hellaswag", gate):
            order = int(row["order"])
            if order < 8:
                query, choices, gold = hellaswag_prompt(row["example"])
                text[order] = {"query": query, "choices": choices, "gold": gold}
    items = []
    for order in range(8):
        item = {"order": order, "text": text.get(order), "conditions": {}}
        for condition in REQUIRED:
            rows = _hellaswag_vectors(output, condition)
            if order in rows:
                row = rows[order]
                item["conditions"][condition] = {
                    "loglikelihoods": row["loglikelihoods"],
                    "prediction": row["prediction"],
                    "prediction_norm": row["prediction_norm"],
                    "correct": row["correct"],
                    "correct_norm": row["correct_norm"],
                    "n_chars": row["n_chars"],
                }
        items.append(item)
    return items


def _deltas(summaries: dict[str, dict]) -> list[dict]:
    base = summaries["fp16"]
    rows = []
    for condition, summary in summaries.items():
        rows.append(
            {
                "condition": condition,
                "packed_bytes": summary.get("packed_bytes"),
                "dense_fp16_bytes": summary.get("dense_fp16_bytes"),
                "achieved_sparsity": summary.get("achieved_sparsity"),
                "nominal_bits": summary.get("nominal_bits"),
                "wikitext_ppl": summary["quality"]["wikitext"]["perplexity"],
                "wikitext_ppl_minus_fp16": summary["quality"]["wikitext"]["perplexity"] - base["quality"]["wikitext"]["perplexity"],
                "c4_ppl": summary["quality"]["c4_validation"]["perplexity"],
                "c4_ppl_minus_fp16": summary["quality"]["c4_validation"]["perplexity"] - base["quality"]["c4_validation"]["perplexity"],
                "hellaswag_accuracy": summary["quality"]["hellaswag"]["accuracy"],
                "hellaswag_accuracy_minus_fp16": summary["quality"]["hellaswag"]["accuracy"] - base["quality"]["hellaswag"]["accuracy"],
                "hellaswag_acc_norm": summary["quality"]["hellaswag"]["acc_norm"],
                "hellaswag_n": summary["quality"]["hellaswag"]["n_examples"],
            }
        )
    return rows


def analyze(output: Path) -> None:
    missing = [condition for condition in REQUIRED if not (output / "conditions" / condition / "summary.json").is_file()]
    if missing:
        raise SystemExit(
            "analyze needs the finished condition summaries. Still missing: " + ", ".join(missing)
        )
    summaries = {condition: _load_summary(output, condition) for condition in REQUIRED}
    figures = output / "figures"
    _quality_figure(
        summaries,
        figures / "pruning_quality_vs_sparsity.png",
        [
            ("sparsity", "wikitext_ppl", "WikiText-2 perplexity"),
            ("sparsity", "c4_ppl", "C4 validation perplexity"),
            ("sparsity", "hellaswag_acc", "HellaSwag accuracy"),
        ],
    )
    _quality_figure(
        summaries,
        figures / "quant_quality_vs_bytes.png",
        [
            ("bytes", "wikitext_ppl", "WikiText-2 perplexity"),
            ("bytes", "c4_ppl", "C4 validation perplexity"),
            ("bytes", "hellaswag_acc", "HellaSwag accuracy"),
        ],
    )
    _heatmap(
        summaries["p3_s50"]["projections"],
        "sensitivity",
        figures / "p3_sensitivity.png",
        "P3 sensitivity: relative MSE of an isolated 50% Wanda prune",
    )
    _heatmap(summaries["p3_s50"]["projections"], "sparsity", figures / "p3_sparsity_s50.png", "P3 sparsity at the 50% budget")
    _heatmap(summaries["p3_s75"]["projections"], "sparsity", figures / "p3_sparsity_s75.png", "P3 sparsity at the 75% budget")
    _heatmap(
        summaries["q3_b4"]["projections"],
        "sensitivity",
        figures / "q3_sensitivity.png",
        "Q3 sensitivity: Q2 INT4 calibration output MSE",
    )
    _heatmap(summaries["q3_b4"]["projections"], "bits", figures / "q3_bits_int4_budget.png", "Q3 bits under the Q2 INT4 budget")
    _heatmap(summaries["q3_b3"]["projections"], "bits", figures / "q3_bits_int3_budget.png", "Q3 bits under the Q2 INT3 budget")
    bootstrap = {label: _paired(output, left, right) for left, right, label in PAIRS}
    # Keep the human label as the key rather than only the condition ids.
    labeled = {}
    for left, right, label in PAIRS:
        labeled[label] = bootstrap[label]
    write_json(output / "bootstrap.json", labeled)
    write_json(output / "comparison.json", _deltas(summaries))
    write_json(output / "qualitative_hellaswag.json", _qualitative(output, summaries))
    print("analyze: wrote figures, bootstrap.json, comparison.json, qualitative_hellaswag.json", flush=True)
