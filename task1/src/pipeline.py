"""GPU stages. Test files open only after P3's tau and Q3's bit map are frozen."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from shared.paths import DATA_REPO_COMMIT, DATA_REPO_URL
from shared.seed import DEFAULT_SEED, set_seed
from task1.src.allocation import (
    Q3_LADDER_RULE,
    Q3M_RULE,
    ProjectionSize,
    allocate_p3_zeros,
    allocate_q3_bits,
    allocate_q3_knapsack,
)
from task1.src.apply import (
    average_bits,
    p3_matrix,
    projection_scale_bytes,
    prune_matrix,
    quantize_awq,
    quantize_uniform,
)
from task1.src.constants import (
    AWQ_ALPHAS,
    FULL_HELLASWAG_CONDITIONS,
    GROUP_SIZE,
    HELLASWAG_DIAGNOSTIC,
    MODEL_ID,
    P3_SPARSITIES,
    P3_TAUS,
    PRUNE_SPARSITIES,
    QUANT_BITS,
)
from task1.src import run_profile
from task1.src.data import (
    TestFileGate,
    load_locked,
    require_locked,
    require_present,
    token_rows,
    write_json,
    write_jsonl,
)
from task1.src.evaluate import development_nll, evaluate_hellaswag, evaluate_windows
from task1.src.model_ops import (
    collect_activation_stats,
    grouped_modules,
    layer_index,
    list_eligible,
    load_qwen,
    measure_reconstruction,
    norm_name_for,
    parameter_census,
    projection_kind,
    read_weight,
    restore_state,
    scale_norm_weight,
    snapshot_state,
    validate_layout,
    write_weight,
)
from task1.src.quantize import channel_scales, compensation_plan
from task1.src.resources import measure_latencies
from task1.src.storage import quantized_model_bytes, sparse_storage_bytes


HELLASWAG_PROTOCOL = (
    "Locked Rowan/hellaswag validation examples. "
    "The query is preprocess(activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize()), "
    "and each choice is preprocess(ending), using the lm-evaluation-harness bracket cleanup. "
    "The score is the conditional token log-likelihood of the choice given the query. "
    "acc_norm divides by the character length of the preprocessed choice, not by token count."
)


def _runtime(model) -> dict:
    import torch

    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    parameter = next(model.parameters())
    return {
        "model_id": MODEL_ID,
        "data_repo": DATA_REPO_URL,
        "data_repo_commit": DATA_REPO_COMMIT,
        "seed": DEFAULT_SEED,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "device": parameter.device.type,
        "dtype": str(parameter.dtype).replace("torch.", ""),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "vocab_size": int(model.config.vocab_size),
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "group_size": GROUP_SIZE,
        "execution": "dense_fp16",
        "packed_kernel": False,
        "smoke": run_profile.current().smoke,
    }


def _peak_now() -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _start(output: Path) -> None:
    """Cheap checks that must pass before any weight is loaded."""
    run_profile.check_output_dir(output, run_profile.current())
    output.mkdir(parents=True, exist_ok=True)


def _prepare(output: Path):
    _start(output)
    set_seed(DEFAULT_SEED)
    model, tokenizer = load_qwen()
    modules = list_eligible(model)
    census = parameter_census(model)
    validate_layout(model, census)
    profile = run_profile.current()
    if profile.last_layer_only:
        # Smoke: compress the last block only; the rest stay FP16 and are counted as excluded.
        modules = run_profile.select_modules(modules, int(model.config.num_hidden_layers), profile)
        census = parameter_census(model, modules)
        print(f"SMOKE profile: {len(modules)} eligible projections, sizes {profile.as_dict()}", flush=True)
    snapshot = snapshot_state(model)
    return model, tokenizer, modules, census, snapshot


def _load_calibration(gate: TestFileGate):
    profile = run_profile.current()
    calibration = token_rows(load_locked("calibration", gate))
    development = token_rows(load_locked("development", gate))
    run_profile.check_split_sizes(calibration, development, profile)
    calibration = run_profile.shrink_rows(calibration, profile.calibration_sequences, profile.sequence_length)
    development = run_profile.shrink_rows(development, profile.development_sequences, profile.sequence_length)
    return calibration, development


def _load_tests(gate: TestFileGate):
    if not gate.frozen:
        raise RuntimeError("test files were requested before the allocation freeze")
    profile = run_profile.current()
    wiki = load_locked("wikitext", gate)
    c4 = load_locked("c4_validation", gate)
    hellaswag = load_locked("hellaswag", gate)
    hellaswag = sorted(hellaswag, key=lambda row: int(row["order"]))
    if profile.eval_windows is not None:
        wiki = wiki[: profile.eval_windows]
        c4 = c4[: profile.eval_windows]
    if profile.hellaswag_full is not None:
        hellaswag = hellaswag[: profile.hellaswag_full]
    if not wiki or not c4 or not hellaswag:
        raise SystemExit("a locked test split is empty")
    return wiki, c4, hellaswag


def _hellaswag_slice(records: list[dict], condition: str) -> tuple[list[dict], str]:
    if condition in FULL_HELLASWAG_CONDITIONS:
        return records, "full"
    limit = run_profile.current().hellaswag_diagnostic
    return records[:limit], "diagnostic_512" if limit == HELLASWAG_DIAGNOSTIC else f"diagnostic_{limit}"


def _json_record(name: str, module, payload: dict, extra: dict | None = None) -> dict:
    rows, cols = module.weight.shape
    record = {
        "module": name,
        "layer": layer_index(name),
        "projection": projection_kind(name),
        "rows": int(rows),
        "cols": int(cols),
        "numel": int(rows * cols),
    }
    for key, value in payload.items():
        if key in {"values", "scores", "mask", "scales", "values_original_space", "values_scaled_space"}:
            continue
        if isinstance(value, (np.floating, float)):
            record[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            record[key] = int(value)
        else:
            record[key] = value
    if extra:
        record.update(extra)
    return record


def _commit(model, modules, values: dict[str, np.ndarray], folds: list[tuple[str, np.ndarray]]) -> None:
    seen = set()
    for norm_name, scales in folds:
        if norm_name in seen:
            continue
        scale_norm_weight(model, norm_name, scales)
        seen.add(norm_name)
    for name, module in modules:
        write_weight(module, values[name])


def _save_condition(output: Path, summary: dict, wiki: dict | None, c4: dict | None, hellaswag_rows: list[dict] | None) -> None:
    folder = output / "conditions" / summary["condition"]
    folder.mkdir(parents=True, exist_ok=True)
    if wiki is not None:
        np.savez_compressed(folder / "window_nll_wikitext.npz", nll_sums=wiki["nll_sums"], counts=wiki["counts"])
        summary["quality"]["wikitext"] = {
            key: wiki[key] for key in ("mean_nll", "perplexity", "n_tokens", "n_windows", "nll_sum")
        }
    if c4 is not None:
        np.savez_compressed(folder / "window_nll_c4.npz", nll_sums=c4["nll_sums"], counts=c4["counts"])
        summary["quality"]["c4_validation"] = {
            key: c4[key] for key in ("mean_nll", "perplexity", "n_tokens", "n_windows", "nll_sum")
        }
    if hellaswag_rows is not None:
        write_jsonl(folder / "hellaswag.jsonl", hellaswag_rows)
    profile = run_profile.current()
    summary["smoke"] = profile.smoke
    if profile.smoke:
        summary["smoke_profile"] = profile.as_dict()
        summary["smoke_note"] = "SMOKE RUN: tiny sizes and two layers. Not a reportable result."
    # summary.json is written last (atomically); _condition_done keys on it.
    write_json(folder / "summary.json", summary)


def _evaluate(model, tokenizer, wiki, c4, hellaswag, condition: str):
    wiki_metrics = evaluate_windows(model, wiki)
    c4_metrics = evaluate_windows(model, c4)
    chosen, subset = _hellaswag_slice(hellaswag, condition)
    hs_summary, hs_rows = evaluate_hellaswag(model, tokenizer, chosen)
    hs_summary = dict(hs_summary)
    hs_summary["subset"] = subset
    return wiki_metrics, c4_metrics, hs_summary, hs_rows


def _latency(model, skip: bool):
    if skip:
        return None
    profile = run_profile.current()
    return measure_latencies(
        model,
        int(model.config.vocab_size),
        decode_tokens=profile.decode_tokens,
        warmup=profile.timing_warmup,
        repeats=profile.timing_repeats,
    )


def run_baseline(output: Path, microbatch: int = 4, skip_latency: bool = False) -> None:
    gate = TestFileGate()
    gate.freeze("fp16 baseline has no sparsity or bit allocation")
    _start(output)
    if _condition_done(output, "fp16"):
        print("baseline: fp16 already written, skipping", flush=True)
        _write_index(output, ["fp16"])
        return
    require_locked(["wikitext", "c4_validation", "hellaswag"], gate)
    model, tokenizer, modules, census, _snapshot = _prepare(output)
    wiki, c4, hellaswag = _load_tests(gate)
    print("baseline: scoring FP16", flush=True)
    wiki_metrics, c4_metrics, hs_summary, hs_rows = _evaluate(model, tokenizer, wiki, c4, hellaswag, "fp16")
    latency = _latency(model, skip_latency)
    summary = {
        "condition": "fp16",
        "family": "baseline",
        "method": "fp16",
        "achieved_nonzero_fraction": 1.0,
        "nominal_bits": 16,
        "packed_bytes": census["n_unique"] * 2,
        "dense_fp16_bytes": census["n_unique"] * 2,
        "unstructured_sparse_bytes": sparse_storage_bytes(census["n_eligible"], census["n_eligible"], census["n_excluded"])[
            "unstructured_sparse_bytes"
        ],
        "storage_note": "Execution is dense FP16. unstructured_sparse_bytes is the mask-plus-value estimate with every eligible weight nonzero.",
        "calibration_seconds": 0.0,
        "census": {key: census[key] for key in ("n_eligible", "n_excluded", "n_unique", "n_eligible_matrices", "tied")},
        "quality": {"hellaswag": hs_summary, "protocol": HELLASWAG_PROTOCOL},
        "latency": latency,
        "runtime": _runtime(model),
        "recipe": {"family": "baseline"},
        "projections": [],
    }
    _ = modules
    _save_condition(output, summary, wiki_metrics, c4_metrics, hs_rows)
    # Merge, never overwrite: a rerun must not drop conditions other stages already listed.
    _write_index(output, ["fp16"])
    print("baseline: wrote", output / "conditions" / "fp16" / "summary.json", flush=True)


def _stats(model, modules, calibration, microbatch: int) -> tuple[dict, float, int | None]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stats = collect_activation_stats(model, modules, calibration, microbatch)
    seconds = time.perf_counter() - started
    return stats, seconds, _peak_now()


def _wanda_relative(model, modules, calibration, microbatch, stats, sparsity: float) -> dict[str, float]:
    recon = {}
    for name, module in modules:
        built = prune_matrix(read_weight(module), stats[name]["column_l2"], "wanda", sparsity)
        recon[name] = _as_fp16(built["values"])
        del built
    measured = measure_reconstruction(model, modules, calibration, microbatch, recon)
    del recon
    return {name: measured[name]["relative_mse"] for name, _module in modules}


def _apply_prune_method(modules, stats, method: str, sparsity: float, zero_counts: dict[str, int] | None = None):
    values = {}
    records = []
    nonzero = 0
    for name, module in modules:
        weight = read_weight(module)
        if method == "p3":
            if zero_counts is None:
                raise ValueError("p3 needs repaired zero counts")
            built = p3_matrix(weight, stats[name]["column_l2"], zero_counts[name])
        else:
            built = prune_matrix(weight, stats[name]["column_l2"], method, sparsity)
        values[name] = _as_fp16(built["values"])
        nonzero += int(built["nonzero"])
        records.append(_json_record(name, module, built))
        del built, weight
    return values, records, nonzero


def _p3_counts(modules, sensitivities: dict[str, float], sparsity: float, tau: float) -> dict[str, int]:
    sizes = []
    sens = []
    for name, module in modules:
        rows, cols = module.weight.shape
        sizes.append(ProjectionSize(name, int(rows), int(cols)))
        sens.append(float(sensitivities[name]))
    zeros = allocate_p3_zeros(sizes, sens, sparsity, tau)
    return {size.name: count for size, count in zip(sizes, zeros)}


TEST_KINDS = ["wikitext", "c4_validation", "hellaswag"]
PRUNE_CONDITIONS = ("p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p2_2to4", "p3_s50", "p3_s75")
QUANT_CONDITIONS = tuple(f"q{family}_b{bits}" for bits in QUANT_BITS for family in (1, 2)) + ("q3_b4", "q3_b3")


def _all_done(output: Path, stage: str, conditions) -> bool:
    """True (and the index refreshed) when a resumed stage has nothing left to evaluate."""
    if all(_condition_done(output, condition) for condition in conditions):
        print(f"{stage}: every condition already written, nothing to do", flush=True)
        _write_index(output, list(conditions))
        return True
    return False


def run_prune(output: Path, microbatch: int = 4, skip_latency: bool = False) -> None:
    gate = TestFileGate()
    _start(output)
    if _all_done(output, "prune", PRUNE_CONDITIONS):
        return
    require_locked(["calibration", "development"], gate)
    # Existence only (no test file is opened): a missing lock must fail now, not after the tau search.
    require_present(TEST_KINDS)
    calibration, development = _load_calibration(gate)
    model, tokenizer, modules, census, snapshot = _prepare(output)
    print("prune: collecting calibration activations", flush=True)
    stats, stat_seconds, stat_peak = _stats(model, modules, calibration, microbatch)
    cached = _load_p3_cache(output)
    if cached is not None:
        sensitivity, selection = cached
        winner = selection["winner"]
        print(f"prune: reusing frozen sensitivity and tau={winner} from {output / 'allocation'}", flush=True)
    else:
        sensitivity, selection = _select_p3_tau(model, modules, calibration, development, microbatch, stats, snapshot, output)
        winner = selection["winner"]
    gate.freeze(f"P3 tau frozen at {winner}")
    restore_state(model, snapshot)
    wiki, c4, hellaswag = _load_tests(gate)
    _prune_conditions(output, model, tokenizer, modules, census, snapshot, calibration, microbatch, skip_latency,
                      stats, stat_seconds, stat_peak, sensitivity, winner, wiki, c4, hellaswag)


def _load_p3_cache(output: Path):
    import json

    tau_path = output / "allocation" / "p3_tau.json"
    sens_path = output / "allocation" / "p3_sensitivity.json"
    if not (tau_path.is_file() and sens_path.is_file()):
        return None
    selection = json.loads(tau_path.read_text(encoding="utf-8"))
    raw = json.loads(sens_path.read_text(encoding="utf-8"))
    sensitivity = {float(key): {name: float(v) for name, v in table.items()} for key, table in raw.items()}
    return sensitivity, selection


def _select_p3_tau(model, modules, calibration, development, microbatch, stats, snapshot, output: Path):
    print("prune: Wanda sensitivity at 50% and 75%", flush=True)
    sensitivity = {}
    for sparsity in P3_SPARSITIES:
        sensitivity[sparsity] = _wanda_relative(model, modules, calibration, microbatch, stats, sparsity)
    write_json(
        output / "allocation" / "p3_sensitivity.json",
        {str(sparsity): table for sparsity, table in sensitivity.items()},
    )
    tau_report = {}
    for tau in P3_TAUS:
        per_sparsity = {}
        nll_sum = 0.0
        n_tokens = 0
        for sparsity in P3_SPARSITIES:
            restore_state(model, snapshot)
            counts = _p3_counts(modules, sensitivity[sparsity], sparsity, tau)
            values, _records, _nonzero = _apply_prune_method(modules, stats, "p3", sparsity, counts)
            _commit(model, modules, values, [])
            del values
            scored = development_nll(model, development)
            per_sparsity[str(sparsity)] = {
                "mean_nll": scored["mean_nll"],
                "n_tokens": scored["n_tokens"],
                "nll_sum": scored["nll_sum"],
            }
            nll_sum += float(scored["nll_sum"])
            n_tokens += int(scored["n_tokens"])
        tau_report[str(tau)] = {
            "mean_nll": nll_sum / n_tokens,
            "n_tokens": n_tokens,
            "per_sparsity": per_sparsity,
        }
        print(f"prune: tau={tau} development mean NLL={tau_report[str(tau)]['mean_nll']:.6f}", flush=True)
    winner = min(P3_TAUS, key=lambda tau: (_nan_last(tau_report[str(tau)]["mean_nll"]), tau))
    selection = {
        "taus": list(P3_TAUS),
        "winner": winner,
        "tie_break": "smaller tau",
        "score": "pooled development token NLL of the 50% and 75% models",
        "development": tau_report,
        "sensitivity_definition": "relative output MSE of an isolated Wanda prune at the target global sparsity",
    }
    write_json(output / "allocation" / "p3_tau.json", selection)
    return sensitivity, selection


def _nan_last(value: float) -> float:
    """A NaN score (fp16 overflow in a broken model) ranks last instead of poisoning min()."""
    value = float(value)
    return float("inf") if value != value else value


def _prune_conditions(output, model, tokenizer, modules, census, snapshot, calibration, microbatch, skip_latency,
                      stats, stat_seconds, stat_peak, sensitivity, winner, wiki, c4, hellaswag) -> None:
    jobs = [("p1_s25", "magnitude", 0.25), ("p1_s50", "magnitude", 0.50), ("p1_s75", "magnitude", 0.75),
            ("p2_s25", "wanda", 0.25), ("p2_s50", "wanda", 0.50), ("p2_s75", "wanda", 0.75),
            ("p2_2to4", "wanda_2to4", 0.50), ("p3_s50", "p3", 0.50), ("p3_s75", "p3", 0.75)]
    written = []
    for condition, method, sparsity in jobs:
        if _condition_done(output, condition):
            print(f"prune: {condition} already written, skipping", flush=True)
            _write_index(output, [condition])
            continue
        print(f"prune: {condition}", flush=True)
        restore_state(model, snapshot)
        counts = None
        if method == "p3":
            counts = _p3_counts(modules, sensitivity[sparsity], sparsity, winner)
        started = time.perf_counter()
        values, records, nonzero = _apply_prune_method(modules, stats, method, sparsity, counts)
        if method == "p3":
            for record in records:
                record["sensitivity"] = float(sensitivity[sparsity][record["module"]])
                record["sparsity"] = record["zeros"] / record["numel"]
        elif method == "wanda":
            for record in records:
                record["sensitivity"] = float(sensitivity[sparsity][record["module"]]) if sparsity in sensitivity else None
        measured = measure_reconstruction(model, modules, calibration, microbatch, values)
        for record in records:
            detail = measured[record["module"]]
            record["relative_mse"] = detail["relative_mse"]
            record["mse"] = detail["mse"]
        _commit(model, modules, values, [])
        del values, measured
        unique_seconds = time.perf_counter() - started
        wiki_metrics, c4_metrics, hs_summary, hs_rows = _evaluate(model, tokenizer, wiki, c4, hellaswag, condition)
        storage = sparse_storage_bytes(census["n_eligible"], nonzero, census["n_excluded"])
        summary = {
            "condition": condition,
            "family": "prune",
            "method": method,
            "sparsity_target": sparsity,
            "tau": winner if method == "p3" else None,
            "achieved_nonzero_fraction": nonzero / census["n_eligible"],
            "achieved_sparsity": 1.0 - (nonzero / census["n_eligible"]),
            "nominal_bits": None,
            "packed_bytes": storage["unstructured_sparse_bytes"],
            "dense_fp16_bytes": storage["dense_fp16_bytes"],
            "unstructured_sparse_bytes": storage["unstructured_sparse_bytes"],
            "storage": storage,
            "storage_note": "packed_bytes is the unstructured mask-bit plus 16-bit-nonzero estimate. Latency is a dense FP16 forward of the explicit zeros.",
            "calibration_seconds": stat_seconds + unique_seconds,
            "shared_activation_seconds": stat_seconds,
            "unique_calibration_seconds": unique_seconds,
            "calibration_peak_allocated_bytes": stat_peak,
            "calibration_time_note": "shared_activation_seconds is the one Wanda/AWQ activation pass for the whole prune stage, repeated on every condition so it is not added up.",
            "quality": {"hellaswag": hs_summary, "protocol": HELLASWAG_PROTOCOL},
            "latency": _latency(model, skip_latency),
            "runtime": _runtime(model),
            "recipe": {"family": "prune", "method": method, "sparsity": sparsity, "tau": winner if method == "p3" else None},
            "projections": records,
        }
        _save_condition(output, summary, wiki_metrics, c4_metrics, hs_rows)
        written.append(condition)
        _write_index(output, [condition])
    print("prune: done", flush=True)


def _as_fp16(values: np.ndarray) -> np.ndarray:
    """Hold an FP16-rounded float64 matrix as float16: exact, and a quarter of the host RAM.

    Every consumer (write_weight, measure_reconstruction) casts to float32 anyway, so results
    are bit-identical. Holding all 196 projections as float64 at once needs ~12 GB per copy
    and ran the Kaggle host out of memory (exit -9) after the Q3 freeze.
    """
    half = np.asarray(values, dtype=np.float16)
    if not np.array_equal(half.astype(np.float64), np.asarray(values, dtype=np.float64)):
        raise RuntimeError("quantized values are not exactly representable in FP16")
    return half


def _search_alphas(model, modules, calibration, microbatch, stats, bits: int):
    mse_table = {name: {} for name, _module in modules}
    for alpha in AWQ_ALPHAS:
        recon = {}
        for name, module in modules:
            built = quantize_awq(read_weight(module), stats[name]["mean_abs"], float(alpha), bits)
            recon[name] = _as_fp16(built["values_original_space"])
            del built
        measured = measure_reconstruction(model, modules, calibration, microbatch, recon)
        for name in recon:
            mse_table[name][float(alpha)] = measured[name]["mse"]
        print(f"quantize: alpha {alpha} at INT{bits} searched", flush=True)
    chosen = {}
    for name, scores in mse_table.items():
        chosen[name] = min(scores, key=lambda alpha: (scores[alpha], alpha))
    return chosen, mse_table


def _scale_plan(modules, stats, alphas: dict[str, float]) -> dict[str, str]:
    """Fold/store decision from channel scales. It does not depend on bit width."""
    modes = {}
    for _prefix, members in grouped_modules(modules).items():
        scales = {}
        names = {}
        for name, _module in members:
            kind = projection_kind(name)
            scales[kind] = channel_scales(stats[name]["mean_abs"], alphas[name])
            names[kind] = name
        plan = compensation_plan(scales)
        for kind, mode in plan.items():
            modes[names[kind]] = mode
    return modes


def _build_quant(modules, stats, method: str, bits: int | None, alphas: dict[str, float] | None, bit_map: dict[str, int] | None, modes: dict[str, str] | None):
    values = {}
    recon = {}
    folds = []
    records = []
    folded = set()
    for name, module in modules:
        weight = read_weight(module)
        assigned = int(bits if bit_map is None else bit_map[name])
        if method == "q1":
            built = quantize_uniform(weight, assigned)
            values[name] = _as_fp16(built["values_original_space"])
            recon[name] = values[name]
            mode = "none"
            scale_bytes = 0
            alpha = None
        else:
            if alphas is None or modes is None:
                raise ValueError("AWQ quantization needs frozen alphas and a compensation plan")
            alpha = float(alphas[name])
            built = quantize_awq(weight, stats[name]["mean_abs"], alpha, assigned)
            mode = modes[name]
            scale_bytes = projection_scale_bytes(int(module.weight.shape[1]), mode)
            recon[name] = _as_fp16(built["values_original_space"])
            if mode == "fold":
                norm = norm_name_for(name)
                if norm is None:
                    raise RuntimeError(f"{name} was marked fold but has no RMSNorm")
                if norm not in folded:
                    folds.append((norm, built["scales"]))
                    folded.add(norm)
                values[name] = _as_fp16(built["values_scaled_space"])
            else:
                values[name] = recon[name]
        records.append(
            _json_record(
                name,
                module,
                built,
                {"compensation": mode, "scale_bytes": scale_bytes, "alpha": alpha, "bits": assigned},
            )
        )
    return values, recon, folds, records


def _quant_bytes(census, records) -> dict:
    projections = [
        {"rows": row["rows"], "cols": row["cols"], "bits": row["bits"], "scale_bytes": row["scale_bytes"]}
        for row in records
    ]
    packed = quantized_model_bytes(projections, census["n_excluded"])
    nonzero = sum(int(row["nonzero"]) for row in records)
    sparse = sparse_storage_bytes(census["n_eligible"], nonzero, census["n_excluded"])
    packed["dense_fp16_bytes"] = sparse["dense_fp16_bytes"]
    packed["unstructured_sparse_bytes"] = sparse["unstructured_sparse_bytes"]
    packed["nominal_bits"] = average_bits(records)
    return packed


def _packed_bytes_for(modules, census, bit_of: dict[str, int], mode_of: dict[str, str]) -> int:
    """Packed bytes of one bit map: payload, group metadata, serialized scales, excluded FP16."""
    projections = []
    for name, module in modules:
        rows, cols = module.weight.shape
        projections.append(
            {
                "rows": int(rows),
                "cols": int(cols),
                "bits": int(bit_of[name]),
                "scale_bytes": projection_scale_bytes(int(cols), mode_of[name]) if mode_of else 0,
            }
        )
    return quantized_model_bytes(projections, census["n_excluded"])["packed_bytes"]


ALPHA_SEARCH_FILE = "q2_alpha_search.json"


def _save_alpha_search(output: Path, alphas: dict[int, dict], mse_tables: dict[int, dict]) -> None:
    write_json(
        output / "allocation" / ALPHA_SEARCH_FILE,
        {
            "grid": list(AWQ_ALPHAS),
            "selection": "smallest calibration output MSE per projection; ties keep the smaller alpha",
            "selection_data": "calibration only; written before any test file is opened",
            "bits": {
                str(bits): {
                    "alphas": {name: float(value) for name, value in alphas[bits].items()},
                    "mse_by_alpha": {
                        name: {str(alpha): float(mse) for alpha, mse in table.items()}
                        for name, table in mse_tables[bits].items()
                    },
                }
                for bits in alphas
            },
        },
    )


def _load_alpha_search(output: Path, names: list[str]):
    """Frozen alphas and MSE tables, or None. A file that does not cover ``names`` is ignored."""
    path = output / "allocation" / ALPHA_SEARCH_FILE
    if not path.is_file():
        return None
    raw = _read_json(path)
    alphas = {}
    mse_tables = {}
    for bits_key, entry in raw.get("bits", {}).items():
        bits = int(bits_key)
        alphas[bits] = {name: float(value) for name, value in entry["alphas"].items()}
        mse_tables[bits] = {
            name: {float(alpha): float(mse) for alpha, mse in table.items()}
            for name, table in entry["mse_by_alpha"].items()
        }
    if set(alphas) != set(QUANT_BITS) or any(set(alphas[bits]) != set(names) for bits in alphas):
        print(f"quantize: {path} does not cover this run's projections; searching again", flush=True)
        return None
    return alphas, mse_tables


def run_quantize(output: Path, microbatch: int = 4, skip_latency: bool = False) -> None:
    gate = TestFileGate()
    _start(output)
    if _all_done(output, "quantize", QUANT_CONDITIONS):
        return
    require_locked(["calibration", "development"], gate)
    require_present(TEST_KINDS)
    calibration, development = _load_calibration(gate)
    model, tokenizer, modules, census, snapshot = _prepare(output)
    print("quantize: collecting calibration activations", flush=True)
    stats, stat_seconds, stat_peak = _stats(model, modules, calibration, microbatch)
    cached = _load_alpha_search(output, [name for name, _module in modules])
    if cached is not None:
        alphas, mse_tables = cached
        print(f"quantize: reusing frozen alpha search from {output / 'allocation' / ALPHA_SEARCH_FILE}", flush=True)
    else:
        alphas = {}
        mse_tables = {}
        for bits in QUANT_BITS:
            restore_state(model, snapshot)
            chosen, table = _search_alphas(model, modules, calibration, microbatch, stats, bits)
            alphas[bits] = chosen
            mse_tables[bits] = table
        _save_alpha_search(output, alphas, mse_tables)
    modes = _scale_plan(modules, stats, alphas[4])

    # Budgets use the same byte function as the Q3 trial maps.
    def bytes_for(bit_of: dict[str, int], mode_of: dict[str, str]) -> int:
        return _packed_bytes_for(modules, census, bit_of, mode_of)

    q2_budgets = {}
    for bits in (4, 3):
        mode_of = _scale_plan(modules, stats, alphas[bits])
        q2_budgets[bits] = bytes_for({name: bits for name, _module in modules}, mode_of)
    q3_modes = modes
    sensitivities = {name: float(mse_tables[4][name][alphas[4][name]]) for name, _module in modules}

    def q3_byte_fn(bit_of: dict[str, int]) -> int:
        return bytes_for(bit_of, q3_modes)

    allocations = {}
    for bits, label in ((4, "q3_b4"), (3, "q3_b3")):
        allocation = allocate_q3_bits(
            [name for name, _module in modules],
            [sensitivities[name] for name, _module in modules],
            q3_byte_fn,
            q2_budgets[bits],
        )
        allocations[label] = allocation
        write_json(
            output / "allocation" / f"{label}.json",
            {
                "budget_condition": f"q2_b{bits}",
                "budget_bytes": allocation.budget_bytes,
                "final_bytes": allocation.final_bytes,
                "within_budget": allocation.within_budget,
                "bits": allocation.bits,
                "steps": [dict(step) for step in allocation.steps],
                "sensitivity": sensitivities,
                "sensitivity_definition": "Q2 calibration output MSE at INT4, original-space reconstruction",
                "scales": "INT4 AWQ channel scales; only the bit width changes",
                "rule": Q3_LADDER_RULE,
            },
        )
        print(f"quantize: froze {label} at {allocation.final_bytes} bytes (budget {allocation.budget_bytes})", flush=True)
    gate.freeze("Q3 bit maps written")
    # Development diagnostic after the freeze. It is not allowed to edit the maps.
    frozen_maps = {label: dict(allocation.bits) for label, allocation in allocations.items()}
    diag_path = output / "allocation" / "q3_development_diagnostic.json"
    dev_diag = _read_json(diag_path) if diag_path.is_file() else None
    if dev_diag is None or set(dev_diag) != set(allocations):
        dev_diag = {}
        for label, allocation in allocations.items():
            restore_state(model, snapshot)
            values, recon, folds, _records = _build_quant(modules, stats, "q3", None, alphas[4], allocation.bits, q3_modes)
            measured = measure_reconstruction(model, modules, development, microbatch, recon)
            del recon
            _commit(model, modules, values, folds)
            del values
            scored = development_nll(model, development)
            dev_diag[label] = {
                "mean_nll": scored["mean_nll"],
                "n_tokens": scored["n_tokens"],
                "mean_projection_mse": float(np.mean([measured[name]["mse"] for name in measured])),
            }
            if allocation.bits != frozen_maps[label]:
                raise RuntimeError("development diagnostic changed a frozen Q3 bit map")
        write_json(diag_path, dev_diag)
    restore_state(model, snapshot)
    wiki, c4, hellaswag = _load_tests(gate)
    jobs = []
    for bits in QUANT_BITS:
        jobs.append((f"q1_b{bits}", "q1", bits, None, None, None))
        jobs.append((f"q2_b{bits}", "q2", bits, alphas[bits], _scale_plan(modules, stats, alphas[bits]), None))
    for label, allocation in allocations.items():
        jobs.append((label, "q3", None, alphas[4], q3_modes, allocation.bits))
    written = []
    for condition, method, bits, alpha_map, mode_map, bit_map in jobs:
        if _condition_done(output, condition):
            print(f"quantize: {condition} already written, skipping", flush=True)
            _write_index(output, [condition])
            continue
        print(f"quantize: {condition}", flush=True)
        restore_state(model, snapshot)
        started = time.perf_counter()
        values, recon, folds, records = _build_quant(modules, stats, method, bits, alpha_map, bit_map, mode_map)
        if method == "q2":
            for record in records:
                record["mse"] = float(mse_tables[bits][record["module"]][record["alpha"]])
                record["mse_by_alpha"] = {str(key): float(value) for key, value in mse_tables[bits][record["module"]].items()}
        elif method == "q3":
            for record in records:
                record["sensitivity"] = sensitivities[record["module"]]
            measured = measure_reconstruction(model, modules, calibration, microbatch, recon)
            for record in records:
                record["mse"] = measured[record["module"]]["mse"]
        else:
            measured = measure_reconstruction(model, modules, calibration, microbatch, recon)
            for record in records:
                record["mse"] = measured[record["module"]]["mse"]
        _commit(model, modules, values, folds)
        del values, recon, folds
        unique_seconds = time.perf_counter() - started
        storage = _quant_bytes(census, records)
        wiki_metrics, c4_metrics, hs_summary, hs_rows = _evaluate(model, tokenizer, wiki, c4, hellaswag, condition)
        summary = {
            "condition": condition,
            "family": "quantize",
            "method": method,
            "bits": bits,
            "nominal_bits": storage["nominal_bits"],
            "achieved_nonzero_fraction": sum(row["nonzero"] for row in records) / census["n_eligible"],
            "packed_bytes": storage["packed_bytes"],
            "dense_fp16_bytes": storage["dense_fp16_bytes"],
            "unstructured_sparse_bytes": storage["unstructured_sparse_bytes"],
            "storage": storage,
            "storage_note": "packed_bytes counts the bit payload, 4 metadata bytes per group, serialized FP16 scales, and excluded FP16 tensors with tied weights once. Latency is dense FP16 of the reconstructed weights, not a low-bit kernel.",
            "calibration_seconds": stat_seconds + unique_seconds,
            "shared_activation_seconds": stat_seconds,
            "unique_calibration_seconds": unique_seconds,
            "calibration_peak_allocated_bytes": stat_peak,
            "quality": {"hellaswag": hs_summary, "protocol": HELLASWAG_PROTOCOL},
            "latency": _latency(model, skip_latency),
            "runtime": _runtime(model),
            "recipe": {
                "family": "quantize",
                "method": method,
                "bits": bits,
                "alphas": alpha_map,
                "bit_map": bit_map,
            },
            "projections": records,
            "q3_development_diagnostic": dev_diag.get(condition),
        }
        _save_condition(output, summary, wiki_metrics, c4_metrics, hs_rows)
        written.append(condition)
        _write_index(output, [condition])
    print("quantize: done", flush=True)


Q3M_LABELS = (("q3m_b4", 4), ("q3m_b3", 3))


def _read_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary(output: Path, condition: str) -> dict:
    path = output / "conditions" / condition / "summary.json"
    if not path.is_file():
        raise SystemExit(f"quantize_q3m needs {path}. Run the quantize stage first.")
    return _read_json(path)


def _awq_mse_at(model, modules, calibration, microbatch, stats, alphas: dict[str, float], bits: int) -> dict[str, float]:
    """Calibration output MSE of every projection quantized at ``bits`` with frozen alphas.

    The reconstructions are held as exact FP16, one copy of all projections at a time.
    """
    recon = {}
    for name, module in modules:
        built = quantize_awq(read_weight(module), stats[name]["mean_abs"], float(alphas[name]), bits)
        recon[name] = _as_fp16(built["values_original_space"])
        del built
    measured = measure_reconstruction(model, modules, calibration, microbatch, recon)
    del recon
    return {name: float(measured[name]["mse"]) for name, _module in modules}


def _q3m_check_alphas(modules, alphas: dict[str, float], q2_b4: dict) -> None:
    """Fail loudly unless the INT4 alphas equal the ones Q2 INT4 recorded."""
    recorded = {row["module"]: float(row["alpha"]) for row in q2_b4["projections"]}
    names = [name for name, _module in modules]
    if set(recorded) != set(names) or set(alphas) != set(names):
        raise RuntimeError("q2_b4 projection records do not cover the same eligible projections")
    wrong = [name for name in names if float(alphas[name]) != recorded[name]]
    if wrong:
        raise RuntimeError(
            f"recomputed INT4 alphas differ from conditions/q2_b4 on {len(wrong)} projections, e.g. "
            + ", ".join(f"{name}: {alphas[name]} vs {recorded[name]}" for name in wrong[:5])
        )


def _q3m_budgets(modules, census, modes4: dict[str, str], q2: dict[int, dict]) -> dict[int, int]:
    """Q2 uniform budgets, recomputed with the Q3 byte function and checked against the Q2 summaries."""
    names = [name for name, _module in modules]
    recorded4 = {row["module"]: row["compensation"] for row in q2[4]["projections"]}
    if any(recorded4[name] != modes4[name] for name in names):
        raise RuntimeError("INT4 fold/store plan differs from the one recorded in conditions/q2_b4")
    budgets = {}
    for bits in (4, 3):
        # Q2 at INT3 used its own alphas' fold/store plan, and its records keep that plan,
        # so the INT3 budget is reproduced without an INT3 alpha search.
        mode_of = {row["module"]: row["compensation"] for row in q2[bits]["projections"]}
        recomputed = _packed_bytes_for(modules, census, {name: bits for name in names}, mode_of)
        if recomputed != int(q2[bits]["packed_bytes"]):
            raise RuntimeError(
                f"recomputed Q2 INT{bits} budget {recomputed} != conditions/q2_b{bits} "
                f"packed_bytes {q2[bits]['packed_bytes']}"
            )
        budgets[bits] = recomputed
    return budgets


def _q3m_int4_alphas(output: Path, modules, q2_b4: dict) -> tuple[dict[str, float], dict[str, float], str]:
    """Frozen Q2 INT4 alphas and their calibration MSE, without a second alpha search.

    The quantize stage saved its search in allocation/q2_alpha_search.json before any test file
    was opened. Re-searching would cost three calibration passes and could flip a near-tied alpha
    if GPU arithmetic differed between sessions, so the frozen choice is reused and checked
    against the alphas and MSE recorded in conditions/q2_b4.
    """
    names = [name for name, _module in modules]
    recorded_mse = {row["module"]: float(row["mse"]) for row in q2_b4["projections"]}
    cached = _load_alpha_search(output, names)
    if cached is not None:
        alphas = dict(cached[0][4])
        mse4 = {name: float(cached[1][4][name][alphas[name]]) for name in names}
        source = f"allocation/{ALPHA_SEARCH_FILE}"
    else:
        alphas = {row["module"]: float(row["alpha"]) for row in q2_b4["projections"]}
        mse4 = dict(recorded_mse)
        source = "conditions/q2_b4/summary.json projection records"
    _q3m_check_alphas(modules, alphas, q2_b4)
    wrong = [name for name in names if mse4[name] != recorded_mse.get(name)]
    if wrong:
        raise RuntimeError(f"INT4 MSE in {source} differs from conditions/q2_b4 on {len(wrong)} projections")
    return alphas, mse4, source


def _q3m_select(output, model, modules, census, calibration, microbatch, stats, q2):
    """Sensitivity passes and both knapsack maps. Reads calibration data only."""
    started = time.perf_counter()
    alphas, mse4, source = _q3m_int4_alphas(output, modules, q2[4])
    print(f"quantize-q3m: INT4 alphas and INT4 MSE from {source} (match conditions/q2_b4)", flush=True)
    modes = _scale_plan(modules, stats, alphas)
    budgets = _q3m_budgets(modules, census, modes, q2)
    print(f"quantize-q3m: budgets INT4={budgets[4]} INT3={budgets[3]} match the Q2 summaries", flush=True)
    names = [name for name, _module in modules]
    mse = {name: {4: mse4[name]} for name in names}
    for bits in (2, 3, 8):
        measured = _awq_mse_at(model, modules, calibration, microbatch, stats, alphas, bits)
        for name in names:
            mse[name][bits] = measured[name]
        print(f"quantize-q3m: calibration MSE at INT{bits} measured", flush=True)

    def byte_fn(bit_of: dict[str, int]) -> int:
        return _packed_bytes_for(modules, census, bit_of, modes)

    v1_bits = {}
    for bits in (4, 3):
        path = output / "allocation" / f"q3_b{bits}.json"
        if path.is_file():
            v1_bits[bits] = {key: int(value) for key, value in _read_json(path)["bits"].items()}
    allocations = {}
    for label, bits in Q3M_LABELS:
        allocation = allocate_q3_knapsack(names, mse, byte_fn, budgets[bits])
        if not allocation.within_budget or allocation.final_bytes != byte_fn(allocation.bits):
            raise RuntimeError(f"{label} is over the Q2 INT{bits} budget")
        v1 = v1_bits.get(bits)
        payload = {
            "budget_condition": f"q2_b{bits}",
            "budget_bytes": allocation.budget_bytes,
            "final_bytes": allocation.final_bytes,
            "start_bytes": allocation.start_bytes,
            "within_budget": allocation.within_budget,
            "bits": allocation.bits,
            "bit_histogram": {str(b): sum(1 for v in allocation.bits.values() if v == b) for b in (2, 3, 4, 8)},
            "steps": [dict(step) for step in allocation.steps],
            "mse_by_bits": {name: {str(b): mse[name][b] for b in (2, 3, 4, 8)} for name in names},
            "alphas": {name: float(alphas[name]) for name in names},
            "compensation": modes,
            "total_calibration_mse": allocation.total_mse,
            "v1_total_calibration_mse": None if v1 is None else float(sum(mse[name][v1[name]] for name in names)),
            "q2_uniform_total_calibration_mse": float(sum(float(row["mse"]) for row in q2[bits]["projections"])),
            "sensitivity_definition": (
                "calibration output MSE of the original-space AWQ reconstruction of each projection at each "
                "b in {2, 3, 4, 8}, with that projection's frozen Q2 INT4 alpha; the INT4 entry is the "
                "alpha-search optimum"
            ),
            "scales": "Q2 INT4 alphas and INT4 fold/store plan at every bit width; only the bit width changes",
            "rule": Q3M_RULE,
            "selection_data": "calibration only; no test file was open when this map was written",
        }
        write_json(output / "allocation" / f"{label}.json", payload)
        allocations[label] = allocation.bits
        print(
            f"quantize-q3m: froze {label} at {allocation.final_bytes} bytes (budget {allocation.budget_bytes}), "
            f"total calibration MSE {allocation.total_mse:.6g}, bits {payload['bit_histogram']}",
            flush=True,
        )
    return allocations, alphas, mse, time.perf_counter() - started


def _q3m_load_frozen(output: Path):
    allocations = {}
    alphas = None
    mse = None
    for label, _bits in Q3M_LABELS:
        frozen = _read_json(output / "allocation" / f"{label}.json")
        allocations[label] = {key: int(value) for key, value in frozen["bits"].items()}
        alphas = {key: float(value) for key, value in frozen["alphas"].items()}
        mse = {key: {int(b): float(v) for b, v in table.items()} for key, table in frozen["mse_by_bits"].items()}
    return allocations, alphas, mse


def run_quantize_q3m(output: Path, microbatch: int = 4, skip_latency: bool = False) -> None:
    """Q3 v2 (q3m): knapsack bit maps from per-bit calibration MSE, then the usual quantize evaluation."""
    labels = [label for label, _bits in Q3M_LABELS]
    diag_path = output / "allocation" / "q3m_development_diagnostic.json"
    if all(_condition_done(output, label) for label in labels) and diag_path.is_file():
        print("quantize-q3m: q3m_b4 and q3m_b3 already written, nothing to do", flush=True)
        _write_index(output, labels)
        return
    q2 = {4: _read_summary(output, "q2_b4"), 3: _read_summary(output, "q2_b3")}
    gate = TestFileGate()
    _start(output)
    require_locked(["calibration", "development"], gate)
    require_present(TEST_KINDS)
    calibration, development = _load_calibration(gate)
    model, tokenizer, modules, census, snapshot = _prepare(output)
    print("quantize-q3m: collecting calibration activations", flush=True)
    stats, stat_seconds, stat_peak = _stats(model, modules, calibration, microbatch)
    selection_seconds = None
    if all((output / "allocation" / f"{label}.json").is_file() for label in labels):
        allocations, alphas, mse = _q3m_load_frozen(output)
        _q3m_check_alphas(modules, alphas, q2[4])
        print("quantize-q3m: reusing frozen q3m maps from", output / "allocation", flush=True)
    else:
        restore_state(model, snapshot)
        allocations, alphas, mse, selection_seconds = _q3m_select(
            output, model, modules, census, calibration, microbatch, stats, q2
        )
    gate.freeze("Q3m bit maps written")
    modes = _scale_plan(modules, stats, alphas)
    frozen_maps = {label: dict(bits) for label, bits in allocations.items()}
    if diag_path.is_file():
        dev_diag = _read_json(diag_path)
    else:
        # Development diagnostic after the freeze. It is not allowed to edit the maps.
        dev_diag = {}
        for label in labels:
            restore_state(model, snapshot)
            values, recon, folds, _records = _build_quant(modules, stats, "q3m", None, alphas, allocations[label], modes)
            measured = measure_reconstruction(model, modules, development, microbatch, recon)
            del recon
            _commit(model, modules, values, folds)
            del values
            scored = development_nll(model, development)
            dev_diag[label] = {
                "mean_nll": scored["mean_nll"],
                "n_tokens": scored["n_tokens"],
                "mean_projection_mse": float(np.mean([measured[name]["mse"] for name in measured])),
            }
            print(f"quantize-q3m: {label} development mean NLL={scored['mean_nll']:.6f}", flush=True)
        write_json(diag_path, dev_diag)
    if allocations != frozen_maps:
        raise RuntimeError("development diagnostic changed a frozen Q3m bit map")
    restore_state(model, snapshot)
    wiki, c4, hellaswag = _load_tests(gate)
    for label in labels:
        if _condition_done(output, label):
            print(f"quantize-q3m: {label} already written, skipping", flush=True)
            _write_index(output, [label])
            continue
        print(f"quantize-q3m: {label}", flush=True)
        restore_state(model, snapshot)
        started = time.perf_counter()
        bit_map = allocations[label]
        values, recon, folds, records = _build_quant(modules, stats, "q3m", None, alphas, bit_map, modes)
        del recon
        for record in records:
            name = record["module"]
            # The calibration MSE at the assigned width is the frozen table entry (same data, same reconstruction).
            record["mse"] = mse[name][record["bits"]]
            record["mse_by_bits"] = {str(b): mse[name][b] for b in (2, 3, 4, 8)}
            record["sensitivity"] = mse[name][4]
        _commit(model, modules, values, folds)
        del values, folds
        unique_seconds = time.perf_counter() - started
        storage = _quant_bytes(census, records)
        wiki_metrics, c4_metrics, hs_summary, hs_rows = _evaluate(model, tokenizer, wiki, c4, hellaswag, label)
        summary = {
            "condition": label,
            "family": "quantize",
            "method": "q3m",
            "bits": None,
            "nominal_bits": storage["nominal_bits"],
            "achieved_nonzero_fraction": sum(row["nonzero"] for row in records) / census["n_eligible"],
            "packed_bytes": storage["packed_bytes"],
            "dense_fp16_bytes": storage["dense_fp16_bytes"],
            "unstructured_sparse_bytes": storage["unstructured_sparse_bytes"],
            "storage": storage,
            "storage_note": "packed_bytes counts the bit payload, 4 metadata bytes per group, serialized FP16 scales, and excluded FP16 tensors with tied weights once. Latency is dense FP16 of the reconstructed weights, not a low-bit kernel.",
            "calibration_seconds": stat_seconds + unique_seconds,
            "shared_activation_seconds": stat_seconds,
            "unique_calibration_seconds": unique_seconds,
            "selection_seconds": selection_seconds,
            "selection_note": "INT2/3/8 calibration MSE passes and both knapsacks (INT4 alphas and MSE reused from the frozen Q2 search); null when frozen maps were reused.",
            "calibration_peak_allocated_bytes": stat_peak,
            "quality": {"hellaswag": hs_summary, "protocol": HELLASWAG_PROTOCOL},
            "latency": _latency(model, skip_latency),
            "runtime": _runtime(model),
            "recipe": {
                "family": "quantize",
                "method": "q3m",
                "bits": None,
                "alphas": alphas,
                "bit_map": bit_map,
            },
            "projections": records,
            "q3_development_diagnostic": dev_diag.get(label),
        }
        _save_condition(output, summary, wiki_metrics, c4_metrics, hs_rows)
        _write_index(output, [label])
    print("quantize-q3m: done", flush=True)


def _condition_done(output: Path, condition: str) -> bool:
    """A condition is finished once its summary is on disk (summary.json is written last).

    A summary from the other profile (smoke versus real) is never accepted as done.
    """
    path = output / "conditions" / condition / "summary.json"
    if not path.is_file():
        return False
    try:
        smoke = run_profile.summary_is_smoke(path)
    except ValueError as exc:
        raise SystemExit(f"{path} is not valid JSON ({exc}); delete that condition folder and rerun") from exc
    if smoke != run_profile.current().smoke:
        raise SystemExit(f"{path} has smoke={smoke}, but this run has smoke={run_profile.current().smoke}")
    return True


def _write_index(output: Path, new_conditions: list[str]) -> None:
    index_path = output / "index.json"
    existing: list[str] = []
    if index_path.is_file():
        import json

        existing = list(json.loads(index_path.read_text(encoding="utf-8")).get("conditions", []))
    for condition in new_conditions:
        if condition not in existing:
            existing.append(condition)
    write_json(index_path, {"conditions": existing})


def _resource_conditions(output: Path) -> list[str]:
    """Conditions in index.json plus any finished summary on disk, in index order."""
    import json

    names: list[str] = []
    index_path = output / "index.json"
    if index_path.is_file():
        names = list(json.loads(index_path.read_text(encoding="utf-8")).get("conditions", []))
    for path in sorted(output.glob("conditions/*/summary.json")):
        if path.parent.name not in names:
            names.append(path.parent.name)
    return [name for name in names if (output / "conditions" / name / "summary.json").is_file()]


def _latency_ok(latency) -> bool:
    """Recorded when the shortest prefill was timed (a 2048-token OOM is a recorded outcome)."""
    prefill = (latency or {}).get("prefill") or {}
    if not prefill:
        return False
    shortest = min(prefill, key=lambda key: int(key))
    return prefill[shortest].get("status") == "ok"


def run_resources(output: Path, microbatch: int = 4, remeasure: bool = False, only: list[str] | None = None) -> None:
    """Fill dense-FP16 latency for conditions whose summaries do not have it yet.

    ``remeasure`` times them again; ``only`` restricts the stage to the named conditions.
    """
    import json

    _start(output)
    names = _resource_conditions(output)
    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise SystemExit(f"resources --only names conditions with no summary: {', '.join(unknown)}")
        names = [name for name in names if name in set(only)]
    if not names:
        raise SystemExit(f"No condition summaries under {output}. Run baseline, prune, and quantize before resources.")
    pending = []
    for name in names:
        summary_path = output / "conditions" / name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if bool(summary.get("smoke", False)) != run_profile.current().smoke:
            raise SystemExit(f"{summary_path} does not match this run's smoke setting")
        if remeasure or not _latency_ok(summary.get("latency")):
            recipe = summary.get("recipe") or {}
            if "family" not in recipe:
                raise SystemExit(f"{summary_path} has no recipe; rerun that stage")
            pending.append((name, summary))
    if not pending:
        print("resources: latency already recorded for every condition", flush=True)
        return
    needs_stats = any(item[1]["recipe"]["family"] != "baseline" for item in pending)
    gate = TestFileGate()
    calibration = None
    if needs_stats:
        require_locked(["calibration", "development"], gate)
        calibration, _development = _load_calibration(gate)
    model, _tokenizer, modules, _census, snapshot = _prepare(output)
    stats = None
    if needs_stats:
        stats, _seconds, _peak = _stats(model, modules, calibration, microbatch)
    for name, summary in pending:
        print(f"resources: {name}", flush=True)
        restore_state(model, snapshot)
        recipe = summary["recipe"]
        if recipe["family"] == "prune":
            if stats is None:
                raise RuntimeError("prune resources need calibration activations")
            counts = None
            if recipe["method"] == "p3":
                sensitivity = {
                    record["module"]: float(record["sensitivity"]) for record in summary["projections"]
                }
                counts = _p3_counts(modules, sensitivity, float(recipe["sparsity"]), float(recipe["tau"]))
            values, _records, _nonzero = _apply_prune_method(
                modules, stats, recipe["method"], float(recipe["sparsity"]), counts
            )
            _commit(model, modules, values, [])
            del values
        elif recipe["family"] == "quantize" and recipe["method"] != "q1":
            if stats is None or recipe.get("alphas") is None:
                raise SystemExit(f"{name} is missing frozen AWQ alphas; rerun quantize.")
            alphas = {key: float(value) for key, value in recipe["alphas"].items()}
            modes = _scale_plan(modules, stats, alphas)
            bit_map = recipe.get("bit_map")
            if bit_map is not None:
                bit_map = {key: int(value) for key, value in bit_map.items()}
            values, _recon, folds, _records = _build_quant(
                modules,
                stats,
                recipe["method"],
                None if recipe["bits"] is None else int(recipe["bits"]),
                alphas,
                bit_map,
                modes,
            )
            _commit(model, modules, values, folds)
            del values, _recon
        elif recipe["family"] == "quantize":
            values, _recon, folds, _records = _build_quant(
                modules, stats, "q1", int(recipe["bits"]), None, None, None
            )
            _commit(model, modules, values, folds)
            del values, _recon
        summary["latency"] = _latency(model, False)
        write_json(output / "conditions" / name / "summary.json", summary)
    print("resources: done", flush=True)


def run_analyze(output: Path) -> None:
    from task1.src.analyze import analyze

    analyze(output)
