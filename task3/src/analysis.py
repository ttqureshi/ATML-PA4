"""CPU-side aggregation of the saved GPU records (speed, workload, accuracy, cost).

Nothing here runs a model. Every function takes the JSON the GPU stages wrote,
so the analyze stage and the unit tests can call it without CUDA.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from shared.bootstrap import bootstrap_paired_mean_diff
from task3.src.metrics import DOMAINS, analytical_speedup, break_even_acceptance, requests_to_amortize

OUTPUT_LENGTH_BINS = (("1-16", 1, 16), ("17-32", 17, 32), ("33-63", 33, 63), ("64+", 64, 10**9))
SPEC_TIME_PARTS = ("prefill_s", "draft_proposal_s", "target_verification_s", "acceptance_residual_s", "cache_rollback_s", "host_overhead_s")


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _pooled_alpha(rows: Sequence[dict], denominator: str = "examined") -> float | None:
    accepted = sum(row["speculative"].get("accepted", 0) for row in rows)
    total = sum(row["speculative"].get(denominator, 0) for row in rows)
    return _ratio(accepted, total)


def _row_ratio(row: dict) -> float | None:
    spec = row["speculative"].get("tokens_per_second")
    target = row["target_only"].get("tokens_per_second")
    if not spec or not target:
        return None
    return spec / target


def _per_prompt(rows: Sequence[dict]) -> list[dict]:
    """Average the paired repetitions of each prompt before any bootstrap."""
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    out = []
    for prompt_id in sorted(by_prompt):
        items = by_prompt[prompt_id]
        out.append(
            {
                "prompt_id": prompt_id,
                "domain": items[0].get("domain"),
                "speculative_tps": _mean(item["speculative"]["tokens_per_second"] for item in items) or 0.0,
                "target_tps": _mean(item["target_only"]["tokens_per_second"] for item in items) or 0.0,
                "ratio": _mean(_row_ratio(item) for item in items),
                "acceptance": _pooled_alpha(items) or 0.0,
                "speculative_latency_s": _mean(item["speculative"]["elapsed_s"] for item in items),
                "target_latency_s": _mean(item["target_only"]["elapsed_s"] for item in items),
            }
        )
    return out


def speed_summary(rows: Sequence[dict], isolated: dict, gamma: int) -> dict:
    """Projection, measured speedup, work, time decomposition and paired bootstrap for one gamma."""
    per_prompt = _per_prompt(rows)
    alpha = _pooled_alpha(rows, "examined") or 0.0
    alpha_proposed = _pooled_alpha(rows, "proposed") or 0.0
    c = float(isolated["c"])
    v = float(isolated["v_gamma"])
    spec_tokens = sum(row["speculative"]["output_tokens"] for row in rows)
    target_tokens = sum(row["target_only"]["output_tokens"] for row in rows)
    spec_time = sum(row["speculative"]["elapsed_s"] for row in rows)
    target_time = sum(row["target_only"]["elapsed_s"] for row in rows)
    spec_tps = _mean(item["speculative_tps"] for item in per_prompt) or 0.0
    target_tps = _mean(item["target_tps"] for item in per_prompt) or 0.0
    ratios = [item["ratio"] for item in per_prompt if item["ratio"] is not None]
    decomposition = {
        part: _ratio(sum(row["speculative"].get(part, 0.0) for row in rows), spec_tokens) for part in SPEC_TIME_PARTS
    }
    shares = {part: _ratio(sum(row["speculative"].get(part, 0.0) for row in rows), spec_time) for part in SPEC_TIME_PARTS}
    spec_rounds = sum(row["speculative"].get("rounds", row["speculative"].get("target_passes", 0)) for row in rows)
    report = {
        "gamma": gamma,
        "alpha": alpha,
        "alpha_definition": "pooled accepted / examined proposals (MLE of a position-independent acceptance rate)",
        "alpha_accepted_over_proposed": alpha_proposed,
        "c": c,
        "v_gamma": v,
        "verify_tokens": isolated.get("verify_tokens"),
        "predicted_speedup": analytical_speedup(min(max(alpha, 0.0), 1.0), gamma, c, v),
        "predicted_speedup_alpha_proposed": analytical_speedup(min(max(alpha_proposed, 0.0), 1.0), gamma, c, v),
        "break_even_alpha": break_even_acceptance(gamma, c, v),
        "measured_tokens_per_second_speculative": spec_tps,
        "measured_tokens_per_second_target": target_tps,
        "measured_speedup": _ratio(spec_tps, target_tps),
        "measured_speedup_pooled": _ratio(_ratio(spec_tokens, spec_time) or 0.0, _ratio(target_tokens, target_time) or 0.0),
        "mean_per_prompt_throughput_ratio": _mean(ratios),
        "target_passes_per_output_token": _ratio(sum(row["speculative"]["target_passes"] for row in rows), spec_tokens),
        "draft_forwards_per_output_token": _ratio(sum(row["speculative"].get("draft_proposal_forwards", 0) for row in rows), spec_tokens),
        "mean_accepted_per_round": _ratio(sum(row["speculative"].get("accepted", 0) for row in rows), spec_rounds),
        "mean_emitted_per_round": _ratio(spec_tokens, spec_rounds),
        "seconds_per_output_token": decomposition,
        "time_shares": shares,
        "target_only_seconds_per_output_token": {
            "prefill_s": _ratio(sum(row["target_only"]["prefill_s"] for row in rows), target_tokens),
            "decode_s": _ratio(sum(row["target_only"]["decode_s"] for row in rows), target_tokens),
        },
        "eos_rate": {
            "speculative": _mean(float(row["speculative"]["hit_eos"]) for row in rows),
            "target_only": _mean(float(row["target_only"]["hit_eos"]) for row in rows),
        },
        "length_capped_rate": {
            "speculative": _mean(float(row["speculative"]["length_capped"]) for row in rows),
            "target_only": _mean(float(row["target_only"]["length_capped"]) for row in rows),
        },
        "peak_memory_bytes": {
            kind: {
                "max": max((row[kind].get("peak_memory_bytes", 0) for row in rows), default=0),
                "mean": _mean(row[kind].get("peak_memory_bytes", 0) for row in rows),
            }
            for kind in ("speculative", "target_only")
        },
        "n_rows": len(rows),
        "n_prompts": len(per_prompt),
    }
    if per_prompt:
        report["bootstrap"] = {
            "throughput_spec_minus_target": bootstrap_paired_mean_diff(
                np.asarray([item["speculative_tps"] for item in per_prompt]), np.asarray([item["target_tps"] for item in per_prompt]), seed=6304
            ),
            "latency_target_minus_spec": bootstrap_paired_mean_diff(
                np.asarray([item["target_latency_s"] for item in per_prompt]), np.asarray([item["speculative_latency_s"] for item in per_prompt]), seed=6304
            ),
            "acceptance_mean": bootstrap_paired_mean_diff(
                np.asarray([item["acceptance"] for item in per_prompt]), np.zeros(len(per_prompt)), seed=6304
            ),
        }
        if ratios:
            report["bootstrap"]["throughput_ratio_mean"] = bootstrap_paired_mean_diff(np.asarray(ratios), np.zeros(len(ratios)), seed=6304)
    return report


def attach_speed(report: dict) -> None:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in report["rows"]:
        grouped[int(row["gamma"])].append(row)
    report["speed"] = {str(gamma): speed_summary(rows, report["isolated"][str(gamma)], gamma) for gamma, rows in sorted(grouped.items())}


def output_length_bin(n_tokens: int) -> str:
    for name, low, high in OUTPUT_LENGTH_BINS:
        if low <= n_tokens <= high:
            return name
    return "0"


def workload_breakdown(rows: Sequence[dict]) -> dict:
    """Speculative outcomes by domain and by actual (speculative) output-length bin, per gamma."""

    def cell(items: list[dict]) -> dict:
        if not items:
            return {"n_rows": 0}
        tokens = sum(item["speculative"]["output_tokens"] for item in items)
        return {
            "n_rows": len(items),
            "alpha_accepted_over_examined": _pooled_alpha(items, "examined"),
            "alpha_accepted_over_proposed": _pooled_alpha(items, "proposed"),
            "target_passes_per_output_token": _ratio(sum(item["speculative"]["target_passes"] for item in items), tokens),
            "mean_output_tokens": tokens / len(items),
            "mean_throughput_ratio": _mean(_row_ratio(item) for item in items),
            "mean_speculative_tps": _mean(item["speculative"]["tokens_per_second"] for item in items),
            "mean_target_tps": _mean(item["target_only"]["tokens_per_second"] for item in items),
            "eos_rate": _mean(float(item["speculative"]["hit_eos"]) for item in items),
        }

    out: dict = {}
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_gamma[int(row["gamma"])].append(row)
    for gamma, items in sorted(by_gamma.items()):
        out[str(gamma)] = {
            "by_domain": {domain: cell([row for row in items if row.get("domain") == domain]) for domain in DOMAINS},
            "by_output_length": {
                name: cell([row for row in items if output_length_bin(row["speculative"]["output_tokens"]) == name])
                for name, _lo, _hi in OUTPUT_LENGTH_BINS
            },
        }
    return out


def bank_vs_measured(alignment: dict, benchmarks: dict, banks: Sequence[str] = ("teacher", "exact_spec_original_gamma4", "original_draft")) -> dict:
    """Expected one-token acceptance (1 - TV) on fixed prefixes next to measured acceptance on trajectories."""
    out = {}
    for name, bench in benchmarks.items():
        entry: dict = {"bank_expected_acceptance": {}, "measured": {}}
        for bank in banks:
            drafts = alignment.get(bank, {}).get("drafts", {})
            if name in drafts:
                entry["bank_expected_acceptance"][bank] = {
                    "positions_1_64": drafts[name]["positions_1_64"].get("expected_acceptance"),
                    "by_domain": {domain: drafts[name]["by_domain"][domain].get("expected_acceptance") for domain in DOMAINS},
                }
        by_gamma: dict[int, list[dict]] = defaultdict(list)
        for row in bench.get("rows", []):
            by_gamma[int(row["gamma"])].append(row)
        for gamma, rows in sorted(by_gamma.items()):
            entry["measured"][str(gamma)] = {
                "overall": _pooled_alpha(rows),
                "by_domain": {domain: _pooled_alpha([row for row in rows if row.get("domain") == domain]) for domain in DOMAINS},
            }
        out[name] = entry
    return out


def standalone_summary(accuracy: dict) -> dict:
    """Per-draft answer quality by domain, with paired prompt bootstrap against the original draft."""
    out: dict = {}
    reference = {row["source_id"]: row for row in accuracy.get("original", [])}
    for name, rows in accuracy.items():
        entry = {}
        for domain in (*DOMAINS, "all"):
            items = [row for row in rows if domain == "all" or row["domain"] == domain]
            if not items:
                continue
            cell = {
                "n": len(items),
                "accuracy": _mean(float(row["correct"]) for row in items),
                "accuracy_lenient": _mean(float(row.get("correct_lenient", row["correct"])) for row in items),
                "format_compliance": _mean(float(row.get("format_compliant", False)) for row in items),
                "unextractable": int(sum(1 for row in items if row["unextractable"])),
                "truncated": int(sum(1 for row in items if row["truncated"])),
                "mean_generated_tokens": _mean(row.get("n_tokens") for row in items),
                "degenerate_repetition": int(sum(1 for row in items if row.get("degenerate_repetition"))),
            }
            if name != "original" and reference:
                matched = [row for row in items if row["source_id"] in reference]
                if matched:
                    cell["accuracy_minus_original"] = bootstrap_paired_mean_diff(
                        np.asarray([float(row["correct"]) for row in matched]),
                        np.asarray([float(reference[row["source_id"]]["correct"]) for row in matched]),
                        seed=6304,
                    )
            entry[domain] = cell
        out[name] = entry
    return out


def training_summary(logs: dict) -> dict:
    """Costs, steps and final loss per run, plus seed dispersion per method."""
    runs = {}
    for name, log in logs.items():
        updates = log.get("updates", [])
        tail = [update["loss"] for update in updates[-10:]]
        runs[name] = {
            "optimizer_steps": log.get("optimizer_steps"),
            "skipped_or_overflowed_steps": log.get("skipped_or_overflowed_steps"),
            "teacher_scored_tokens": log.get("teacher_scored_tokens_total", (log.get("math_scored_tokens", 0) + log.get("short_scored_tokens", 0))),
            "first_loss": updates[0]["loss"] if updates else None,
            "final_loss_mean_last10": _mean(tail),
            "end_to_end_cost_s": log.get("end_to_end_cost_s", log.get("wall_time_s")),
            "phase_seconds": log.get("phase_seconds"),
            "peak_memory_bytes": log.get("peak_memory_bytes"),
            "parameter_counts": log.get("parameter_counts", log.get("parameters")),
        }
    methods: dict[str, list[str]] = defaultdict(list)
    for name in runs:
        methods[name.rsplit("_seed", 1)[0]].append(name)
    dispersion = {}
    for method, names in methods.items():
        dispersion[method] = {}
        for key in ("final_loss_mean_last10", "end_to_end_cost_s", "optimizer_steps"):
            values = {name: runs[name][key] for name in names if runs[name][key] is not None}
            if values:
                dispersion[method][key] = {"per_seed": values, "range": max(values.values()) - min(values.values())}
    return {"runs": runs, "seed_dispersion": dispersion}


def deployment_break_even(logs: dict, merges: dict, benchmarks: dict) -> dict:
    """Upfront distillation + merge cost against measured per-request savings.

    Two baselines per gamma: target-only decoding (adjacent paired runs), and
    speculative decoding with the original undistilled draft (same prompts and
    seeds). A non-positive saving means the cost is never recovered.
    """
    out = {}
    original = benchmarks.get("original", {}).get("rows", [])
    for name, log in logs.items():
        merge = merges.get(name, {})
        upfront = float(log.get("end_to_end_cost_s", log.get("wall_time_s", 0.0))) + float(merge.get("merge_cost_s", 0.0))
        entry = {
            "upfront_seconds": upfront,
            "components_s": {
                "offline_teacher_response_preparation": log.get("offline_teacher_response_preparation_s"),
                "student_rollout": log.get("student_rollout_s"),
                "teacher_scoring": log.get("teacher_scoring_s"),
                "student_forward_and_loss": log.get("student_forward_and_loss_s"),
                "backward": log.get("backward_s"),
                "optimizer": log.get("optimizer_s"),
                "model_load_and_other": None,
                "adapter_merge": merge.get("merge_cost_s"),
            },
            "per_gamma": {},
        }
        known = sum(value for key, value in entry["components_s"].items() if value is not None and key != "adapter_merge")
        entry["components_s"]["model_load_and_other"] = float(log.get("end_to_end_cost_s", 0.0)) - known
        rows = benchmarks.get(name, {}).get("rows", [])
        for gamma in sorted({int(row["gamma"]) for row in rows}):
            mine = _per_prompt([row for row in rows if int(row["gamma"]) == gamma])
            theirs = {item["prompt_id"]: item for item in _per_prompt([row for row in original if int(row["gamma"]) == gamma])}
            vs_target = _mean(item["target_latency_s"] - item["speculative_latency_s"] for item in mine) or 0.0
            matched = [item for item in mine if item["prompt_id"] in theirs]
            vs_original = _mean(theirs[item["prompt_id"]]["speculative_latency_s"] - item["speculative_latency_s"] for item in matched)
            entry["per_gamma"][str(gamma)] = {
                "seconds_saved_per_request_vs_target_only": vs_target,
                "requests_to_amortize_vs_target_only": requests_to_amortize(upfront, vs_target),
                "seconds_saved_per_request_vs_original_draft": vs_original,
                "requests_to_amortize_vs_original_draft": None if vs_original is None else requests_to_amortize(upfront, vs_original),
            }
        out[name] = entry
    return out


def wrapper_overhead_summary(rows: Sequence[dict]) -> dict:
    """Unmerged PEFT draft vs its merged copy, same prompts/seeds/gamma, alternating order."""
    out = {}
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_gamma[int(row["gamma"])].append(row)
    for gamma, items in sorted(by_gamma.items()):
        merged = [row["merged"] for row in items]
        unmerged = [row["unmerged"] for row in items]

        def per_forward(runs: list[dict]) -> float | None:
            return _ratio(sum(run["draft_proposal_s"] for run in runs), sum(run["draft_proposal_forwards"] for run in runs))

        merged_step = per_forward(merged)
        unmerged_step = per_forward(unmerged)
        out[str(gamma)] = {
            "n_rows": len(items),
            "merged_draft_seconds_per_forward": merged_step,
            "unmerged_draft_seconds_per_forward": unmerged_step,
            "draft_forward_overhead_ratio": None if not merged_step or unmerged_step is None else unmerged_step / merged_step,
            "mean_tokens_per_second": {"merged": _mean(run["tokens_per_second"] for run in merged), "unmerged": _mean(run["tokens_per_second"] for run in unmerged)},
            "mean_latency_s": {"merged": _mean(run["elapsed_s"] for run in merged), "unmerged": _mean(run["elapsed_s"] for run in unmerged)},
            "same_tokens_fraction": _mean(float(row["merged"].get("tokens") == row["unmerged"].get("tokens")) for row in items),
            "latency_unmerged_minus_merged": bootstrap_paired_mean_diff(
                np.asarray([run["elapsed_s"] for run in unmerged]), np.asarray([run["elapsed_s"] for run in merged]), seed=6304
            ),
        }
    return out
