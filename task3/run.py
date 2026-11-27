"""Task 3 stages. Unit tests stay on CPU. Every stage that would download Qwen refuses without CUDA.

Order: unit, smoke, baseline, distill, merge, align, eval, analyze. `align`
does forward passes and frozen-rollout generation only (no speed numbers), so
it may run on a second GPU (CUDA_VISIBLE_DEVICES=1) while `eval` times on the
first. Every expensive piece is cached under results/task3/ as soon as it
finishes; with PA4_DEADLINE (unix seconds) set, a stage exits with code 75
before starting a piece that might not fit, and a rerun resumes from the cache.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task3.src.analysis import (
    attach_speed,
    bank_vs_measured,
    deployment_break_even,
    standalone_summary,
    training_summary,
    workload_breakdown,
    wrapper_overhead_summary,
)
from task3.src.distill import (
    DRAFT_MODEL,
    METHODS,
    TARGET_MODEL,
    TRAIN_SEEDS,
    assert_same_token_ids,
    default_data_paths,
    load_eval_pool,
    merge_adapter_into_fresh_backbone,
    sample_response,
    train_method,
)
from task3.src.evaluate import (
    DIAGNOSTIC_MAX_NEW,
    GAMMAS,
    PAIRED_REPETITIONS,
    SPEED_MAX_NEW,
    SpeculativeDecoder,
    _dump,
    _prompt_seed,
    answer_cap,
    benchmark_prompt,
    checkpoint_dir,
    decode_text,
    diagnostic_math_prompts,
    estimate_speed_model,
    render_chat,
    repeated_ngram_fraction,
    repetition_order,
    results_dir,
    score_standalone,
    timed_speculative,
    warm_setup,
)

OUT_OF_BUDGET = 75
PARTS_VERSION = "task3-parts-v2"
WRAPPER_RUN = ("off_policy", 6304)
WRAPPER_PROMPTS_PER_DOMAIN = 8
CONTINUATION_PROMPTS_PER_DOMAIN = 2


def require_cuda(stage: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{stage} requires CUDA and will not download weights or run on CPU.")


def _check_deadline(what: str) -> None:
    """Stop cleanly before Kaggle's 12h session limit. Finished pieces stay on disk."""
    deadline = os.environ.get("PA4_DEADLINE")
    if deadline and time.time() > float(deadline):
        print(f"out of session budget before {what}; rerun the stage to resume", flush=True)
        raise SystemExit(OUT_OF_BUDGET)


def _part(subdir: str, name: str, compute, *, gz: bool = False):
    """Cache one deterministic piece of a stage so a timed-out run resumes.

    Parts written by an older code version (no or another PARTS_VERSION) are recomputed.
    """
    path = results_dir() / subdir / (f"{name}.json.gz" if gz else f"{name}.json")
    if path.is_file():
        if gz:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                stored = json.load(handle)
        else:
            stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("version") == PARTS_VERSION:
            print(f"{subdir}: reusing {name}", flush=True)
            return stored["value"]
        print(f"{subdir}: {name} is from another code version; recomputing", flush=True)
    _check_deadline(f"{subdir} part {name}")
    started = time.time()
    value = compute()
    payload = {"value": value, "version": PARTS_VERSION}
    if gz:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        tmp.replace(path)
    else:
        _dump(path, payload)
    print(f"{subdir}: {name} done in {time.time() - started:.0f}s", flush=True)
    return value


def _part_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("version") == PARTS_VERSION


def _eval_part(name: str, compute):
    return _part("eval_parts", name, compute)


def _free() -> None:
    """Release models the caller has already dropped its references to."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _eos(tokenizer) -> int | None:
    return None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)


def stage_unit() -> None:
    import unittest

    suite = unittest.defaultTestLoader.discover("task3/tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def stage_smoke() -> None:
    require_cuda("smoke")
    from task3.src.evaluate import run_smoke

    run_smoke()


def _export_existing(source: Path, target: Path) -> None:
    """Copy evidence written next to a checkpoint into results/ when only the checkpoint copy exists."""
    import shutil

    if source.is_file() and not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def stage_distill() -> None:
    require_cuda("distill")
    from task3.src.distill import distill_results_dir

    for method in METHODS:
        for seed in TRAIN_SEEDS:
            output = checkpoint_dir() / method / f"seed{seed}"
            if (output / "adapter").is_dir():
                _export_existing(output / "train_log.json", distill_results_dir(method, seed) / "train_log.json")
                print(f"distill {method} seed {seed}: adapter exists, skipping", flush=True)
                continue
            _check_deadline(f"distill {method} seed {seed}")
            print(f"distilling {method} seed {seed}", flush=True)
            train_method(method, seed, output)


def merge_report_path(method: str, seed: int) -> Path:
    return results_dir() / "merge" / f"{method}_seed{seed}" / "merge_error.json"


def stage_merge() -> None:
    require_cuda("merge")
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            adapter = checkpoint_dir() / method / f"seed{seed}" / "adapter"
            merged = checkpoint_dir() / method / f"seed{seed}" / "merged"
            if not adapter.is_dir():
                raise SystemExit(f"missing adapter {adapter}. Run the distill stage first.")
            report = merge_report_path(method, seed)
            if merged.is_dir() and any(merged.glob("*.safetensors")):
                _export_existing(merged / "merge_error.json", report)
                print(f"merge {method} seed {seed}: merged draft exists, skipping", flush=True)
                continue
            _check_deadline(f"merge {method} seed {seed}")
            print(f"merging {method} seed {seed}", flush=True)
            merge_adapter_into_fresh_backbone(adapter, merged, report_path=report)


def _load_target():
    from task3.src.distill import load_frozen_causal

    return load_frozen_causal(TARGET_MODEL)


def _draft_names() -> list[str]:
    return ["original"] + [f"{method}_seed{seed}" for method in METHODS for seed in TRAIN_SEEDS]


def _merged_dir(name: str) -> Path:
    method, seed = name.rsplit("_seed", 1)
    return checkpoint_dir() / method / f"seed{seed}" / "merged"


def _load_draft(name: str, tokenizer):
    """The original 0.5B draft, or a merged FP16 checkpoint. Token ids must match the target."""
    from task3.src.distill import load_frozen_causal

    source = DRAFT_MODEL if name == "original" else str(_merged_dir(name))
    if name != "original" and not _merged_dir(name).is_dir():
        raise SystemExit(f"missing merged draft {_merged_dir(name)}. Run the merge stage first.")
    model, draft_tokenizer = load_frozen_causal(source)
    assert_same_token_ids(tokenizer, draft_tokenizer)
    return model


def _benchmark(target, draft, tokenizer, examples, *, label: str, max_new: int, ignore_eos: bool, seed: int) -> dict:
    eos_id = _eos(tokenizer)
    decoder = SpeculativeDecoder(target, draft, np.random.default_rng(seed))
    setup = warm_setup(decoder, target, render_chat(tokenizer, examples[0].text), gamma=max(GAMMAS), eos_id=None if ignore_eos else eos_id)
    isolated = {str(gamma): estimate_speed_model(draft, target, gamma) for gamma in GAMMAS}
    rows = []
    for gamma in GAMMAS:
        for index, example in enumerate(examples):
            prompt_ids = render_chat(tokenizer, example.text)
            for repetition in range(PAIRED_REPETITIONS):
                record = benchmark_prompt(
                    decoder,
                    target,
                    prompt_ids,
                    gamma=gamma,
                    max_new_tokens=max_new,
                    eos_id=eos_id,
                    seed=_prompt_seed(seed, index, repetition) + 1000 * gamma,
                    repetition=repetition,
                    ignore_eos=ignore_eos,
                )
                record["prompt_id"] = example.source_id
                record["domain"] = example.domain
                record["prompt_tokens"] = len(prompt_ids)
                rows.append(record)
    report = {
        "label": label,
        "ignore_eos": ignore_eos,
        "max_new_tokens": max_new,
        "warm_setup": setup,
        "isolated": isolated,
        "rows": rows,
    }
    attach_speed(report)
    return report


def stage_baseline() -> None:
    require_cuda("baseline")
    target, tokenizer = _load_target()
    draft = _load_draft("original", tokenizer)
    examples = load_eval_pool()
    report = _part(
        "baseline_parts",
        "primary",
        lambda: _benchmark(target, draft, tokenizer, examples, label="original_draft", max_new=SPEED_MAX_NEW, ignore_eos=False, seed=6304),
    )
    diagnostic = _part(
        "baseline_parts",
        "math_diagnostic_64_ignore_eos",
        lambda: _benchmark(
            target, draft, tokenizer, diagnostic_math_prompts(examples), label="math_diagnostic_64",
            max_new=DIAGNOSTIC_MAX_NEW, ignore_eos=True, seed=6304,
        ),
    )
    payload = {
        "primary": report,
        "diagnostic_ignore_eos": diagnostic,
        "diagnostic_note": "24 math prompts, exactly 64 new tokens with EOS ignored, timing only; not an answer-quality run.",
        "target": TARGET_MODEL,
        "draft": DRAFT_MODEL,
    }
    _dump(results_dir() / "baseline.json", payload)
    print(f"wrote {results_dir() / 'baseline.json'}", flush=True)


def stage_align() -> None:
    """Common-prefix alignment of every draft on the shared and student banks (no timing)."""
    require_cuda("align")
    from task3.src import alignment as al

    bank_path = default_data_paths()["prefix_banks"]
    supplied = al.load_prefix_bank(bank_path)
    examples = load_eval_pool()
    eval_text = {example.source_id: example.text for example in examples}
    unknown = sorted({row["prompt_id"] for row in supplied} - set(eval_text))
    if unknown:
        raise SystemExit(f"prefix bank has prompt ids outside the evaluation pool: {unknown[:5]}")
    text_mismatch = sorted({row["prompt_id"] for row in supplied if row["prompt"] != eval_text[row["prompt_id"]]})

    target, tokenizer = _load_target()
    drafts = {name: _load_draft(name, tokenizer) for name in _draft_names()}
    banks: dict[str, list[dict]] = {}
    for row in supplied:
        banks.setdefault(row["bank"], []).append(row)
    banks[al.SPEC_BANK] = _part("align_parts", "bank_exact_spec_original_gamma4", lambda: al.exact_spec_bank(target, drafts["original"], tokenizer, examples), gz=True)
    for name in _draft_names()[1:]:
        seed = int(name.rsplit("_seed", 1)[1])
        banks[f"student_{name}"] = _part(
            "align_parts",
            f"bank_student_{name}",
            lambda name=name, seed=seed: al.student_bank(drafts[name], name, str(_merged_dir(name)), tokenizer, examples, seed=seed),
            gz=True,
        )
    shared = (*al.SUPPLIED_BANKS, al.SPEC_BANK)
    scores: dict[str, list[dict]] = {}
    for bank, rows in banks.items():
        scores[bank] = _part(
            "align_parts",
            f"scores_{bank}",
            lambda rows=rows, bank=bank: al.score_bank(target, drafts, tokenizer, rows, keep_top=bank in shared),
            gz=True,
        )
    all_rows = [row for rows in scores.values() for row in rows]
    shared_prefixes = [row for bank in shared for row in banks[bank]]
    shared_scores = [row for bank in shared for row in scores[bank]]
    payload = {
        "banks": {
            bank: {
                "n_prefixes": len(rows),
                "n_prompts": len({row["prompt_id"] for row in rows}),
                "producers": sorted({str(row.get("producer")) for row in rows}),
            }
            for bank, rows in banks.items()
        },
        "n_supplied_prefixes": len(supplied),
        "supplied_bank_path": str(bank_path),
        "supplied_prompt_text_mismatches": text_mismatch,
        "drafts": {name: ("Qwen/Qwen2.5-0.5B-Instruct" if name == "original" else str(_merged_dir(name))) for name in drafts},
        "summary": al.summarize_scores(all_rows),
        "target_vs_speculative_consistency": al.consistency_check(all_rows),
        "aligned_disagreements": al.decode_disagreements(shared_scores, shared_prefixes, tokenizer),
    }
    _dump(results_dir() / "alignment.json", payload)
    path = results_dir() / "alignment_rows.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in al.compact(all_rows):
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {results_dir() / 'alignment.json'} ({len(all_rows)} scored rows)", flush=True)


def _standalone(model, tokenizer, examples) -> list[dict]:
    """The draft as a model: answer-length caps, temperature 1, fixed per-prompt seeds."""
    eos_id = _eos(tokenizer)
    rows = []
    for index, example in enumerate(examples):
        prompt_ids = render_chat(tokenizer, example.text)
        tokens, reason = sample_response(
            model,
            prompt_ids,
            max_new_tokens=answer_cap(example.domain),
            eos_id=eos_id,
            rng=np.random.default_rng(6304 + 10007 * index),
        )
        text = decode_text(tokenizer, tokens)
        scored = score_standalone(example.domain, text, example.answer, length_capped=reason == "length")
        repeat = repeated_ngram_fraction(tokens)
        scored.update(
            {
                "source_id": example.source_id,
                "domain": example.domain,
                "termination_reason": reason,
                "n_tokens": len(tokens),
                "answer_cap": answer_cap(example.domain),
                "repeated_4gram_fraction": repeat,
                "degenerate_repetition": bool(len(tokens) >= 64 and repeat > 0.5),
                "text": text,
            }
        )
        rows.append(scored)
    return rows


def _wrapper_subset(examples) -> list:
    chosen = []
    for domain in ("math", "familiar", "shifted"):
        chosen.extend([example for example in examples if example.domain == domain][:WRAPPER_PROMPTS_PER_DOMAIN])
    return chosen


def _wrapper_overhead(target, merged, tokenizer, examples) -> dict:
    """Unmerged PEFT adapter vs its merged copy: same prompts, seeds, gammas, 64-token EOS-aware cap.

    Both drafts are resident; per prompt and repetition the two runs alternate order.
    """
    from peft import PeftModel

    from task3.src.distill import load_frozen_causal

    method, seed = WRAPPER_RUN
    adapter = checkpoint_dir() / method / f"seed{seed}" / "adapter"
    base, _ = load_frozen_causal(DRAFT_MODEL)
    unmerged = PeftModel.from_pretrained(base, adapter).eval()
    eos_id = _eos(tokenizer)
    decoders = {
        "merged": SpeculativeDecoder(target, merged, np.random.default_rng(0)),
        "unmerged": SpeculativeDecoder(target, unmerged, np.random.default_rng(0)),
    }
    subset = _wrapper_subset(examples)
    for decoder in decoders.values():
        warm_setup(decoder, target, render_chat(tokenizer, subset[0].text), gamma=max(GAMMAS), eos_id=eos_id)
    rows = []
    for gamma in GAMMAS:
        for index, example in enumerate(subset):
            prompt_ids = render_chat(tokenizer, example.text)
            for repetition in range(PAIRED_REPETITIONS):
                order = ("merged", "unmerged") if repetition_order(repetition)[0] == "target_only" else ("unmerged", "merged")
                seed_value = _prompt_seed(6304, index, repetition) + 1000 * gamma
                row = {"prompt_id": example.source_id, "domain": example.domain, "gamma": gamma, "repetition": repetition, "order": list(order)}
                for kind in order:
                    row[kind] = timed_speculative(
                        decoders[kind], prompt_ids, gamma=gamma, max_new_tokens=SPEED_MAX_NEW, eos_id=eos_id, seed=seed_value
                    )
                rows.append(row)
    del decoders, unmerged, base
    _free()
    return {"run": f"{method}_seed{seed}", "adapter": str(adapter), "n_prompts": len(subset), "rows": rows}


def _continuation_ids(examples) -> list[str]:
    """Fixed rule: the first two evaluation prompts of each domain in file order."""
    chosen = []
    for domain in ("math", "familiar", "shifted"):
        chosen.extend([example.source_id for example in examples if example.domain == domain][:CONTINUATION_PROMPTS_PER_DOMAIN])
    return chosen


def _continuations(tokenizer, examples, benchmarks: dict, accuracy: dict) -> dict:
    ids = _continuation_ids(examples)
    prompts = {example.source_id: example for example in examples}
    out = []
    for prompt_id in ids:
        entry = {"prompt_id": prompt_id, "domain": prompts[prompt_id].domain, "prompt": prompts[prompt_id].text, "gold": prompts[prompt_id].answer}
        entry["standalone"] = {
            name: {key: row[key] for key in ("text", "prediction", "correct", "termination_reason", "n_tokens")}
            for name, rows in accuracy.items()
            for row in rows
            if row["source_id"] == prompt_id
        }
        entry["speculative_gamma4_rep0_64"] = {}
        for name, report in benchmarks.items():
            for row in report["rows"]:
                if row["prompt_id"] == prompt_id and row["gamma"] == 4 and row["repetition"] == 0:
                    entry["speculative_gamma4_rep0_64"][name] = decode_text(tokenizer, row["speculative"]["tokens"])
                    entry.setdefault("target_only_gamma4_rep0_64", decode_text(tokenizer, row["target_only"]["tokens"]))
        out.append(entry)
    return {
        "rule": "First two evaluation prompts of each domain in file order; texts decoded with the shared tokenizer.",
        "note": "Speculative and target-only texts follow the target law; only the standalone texts describe the draft as a model.",
        "examples": out,
    }


def stage_eval() -> None:
    """Speed benchmarks for the original and four merged drafts, wrapper overhead, standalone accuracy.

    Drafts are loaded one at a time, so peak memory reflects the target plus
    one draft. Each piece is cached in results/task3/eval_parts/.
    """
    require_cuda("eval")
    target, tokenizer = _load_target()
    examples = load_eval_pool()
    benchmarks: dict = {}
    accuracy: dict = {}
    diagnostics: dict = {}
    wrapper = None
    for name in _draft_names():
        parts = [f"benchmark_{name}", f"diagnostic_{name}", f"standalone_{name}"]
        if name == f"{WRAPPER_RUN[0]}_seed{WRAPPER_RUN[1]}":
            parts.append("wrapper_overhead")
        cached = all(_part_is_current(results_dir() / "eval_parts" / f"{part}.json") for part in parts)
        draft = None if cached else _load_draft(name, tokenizer)
        benchmarks[name] = _eval_part(
            f"benchmark_{name}",
            lambda draft=draft, name=name: _benchmark(target, draft, tokenizer, examples, label=name, max_new=SPEED_MAX_NEW, ignore_eos=False, seed=6304),
        )
        diagnostics[name] = _eval_part(
            f"diagnostic_{name}",
            lambda draft=draft, name=name: _benchmark(
                target, draft, tokenizer, diagnostic_math_prompts(examples), label=f"{name}_math_diagnostic_64",
                max_new=DIAGNOSTIC_MAX_NEW, ignore_eos=True, seed=6304,
            ),
        )
        if "wrapper_overhead" in parts:
            wrapper = _eval_part("wrapper_overhead", lambda draft=draft: _wrapper_overhead(target, draft, tokenizer, examples))
        accuracy[name] = _eval_part(f"standalone_{name}", lambda draft=draft: _standalone(draft, tokenizer, examples))
        draft = None
        _free()
    payload = {
        "benchmarks": benchmarks,
        "diagnostic_ignore_eos": diagnostics,
        "standalone_accuracy": accuracy,
        "wrapper_overhead": wrapper,
        "continuations": _continuations(tokenizer, examples, benchmarks, accuracy),
        "standalone_note": "Standalone accuracy is the draft as a model (512 math / 32 short caps). Speculative outputs follow the target and are not scored.",
    }
    _dump(results_dir() / "eval.json", payload)
    print(f"wrote {results_dir() / 'eval.json'}", flush=True)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def stage_analyze() -> None:
    """Summarize saved GPU artifacts on CPU. Missing files are reported; nothing is trained here."""
    from task3.src.distill import distill_results_dir

    expected = {
        "smoke": results_dir() / "smoke.json",
        "baseline": results_dir() / "baseline.json",
        "align": results_dir() / "alignment.json",
        "eval": results_dir() / "eval.json",
    }
    logs, merges = {}, {}
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            name = f"{method}_seed{seed}"
            expected[f"distill_{name}"] = distill_results_dir(method, seed) / "train_log.json"
            expected[f"merge_{name}"] = merge_report_path(method, seed)
            if expected[f"distill_{name}"].is_file():
                logs[name] = _read(expected[f"distill_{name}"])
            if expected[f"merge_{name}"].is_file():
                merges[name] = _read(expected[f"merge_{name}"])
    missing = [name for name, path in expected.items() if not path.is_file()]
    report: dict = {"missing": missing, "gpu_stages_remaining": _remaining_stages(missing)}
    smoke = _read(expected["smoke"])
    if smoke:
        report["smoke"] = {key: smoke.get(key) for key in ("passed", "fp32_gate", "fp32_gate_failures", "fp32", "fp16_diagnostic")}
    baseline = _read(expected["baseline"])
    if baseline:
        report["baseline_speed"] = baseline["primary"].get("speed")
        report["baseline_workload"] = workload_breakdown(baseline["primary"]["rows"])
        report["baseline_warm_setup"] = baseline["primary"].get("warm_setup")
        report["baseline_isolated_costs"] = baseline["primary"].get("isolated")
        report["diagnostic_ignore_eos_speed"] = baseline["diagnostic_ignore_eos"].get("speed")
    if logs:
        report["training"] = training_summary(logs)
    if merges:
        report["merge"] = merges
    alignment = _read(expected["align"])
    if alignment:
        report["alignment_overall"] = {
            bank: {name: entry["overall"] for name, entry in table["drafts"].items()} for bank, table in alignment["summary"].items()
        }
        report["target_vs_speculative_consistency"] = {
            key: alignment["target_vs_speculative_consistency"][key] for key in ("n_comparisons", "n_ci_excluding_zero", "overall")
        }
    evaluation = _read(expected["eval"])
    if evaluation:
        benchmarks = evaluation["benchmarks"]
        report["eval_speed"] = {name: payload.get("speed") for name, payload in benchmarks.items()}
        report["eval_workload"] = {name: workload_breakdown(payload["rows"]) for name, payload in benchmarks.items()}
        report["eval_diagnostic_ignore_eos_speed"] = {name: payload.get("speed") for name, payload in evaluation.get("diagnostic_ignore_eos", {}).items()}
        report["standalone"] = standalone_summary(evaluation["standalone_accuracy"])
        if evaluation.get("wrapper_overhead"):
            report["wrapper_overhead"] = wrapper_overhead_summary(evaluation["wrapper_overhead"]["rows"])
            report["wrapper_overhead"]["run"] = evaluation["wrapper_overhead"]["run"]
        if alignment:
            report["bank_vs_measured_acceptance"] = bank_vs_measured(alignment["summary"], benchmarks)
        if logs:
            report["deployment_break_even"] = deployment_break_even(logs, merges, benchmarks)
    _dump(results_dir() / "analyze.json", report)
    print(json.dumps({"missing": missing, "gpu_stages_remaining": report["gpu_stages_remaining"]}, indent=2))


def _remaining_stages(missing: list[str]) -> list[str]:
    stages = []
    for stage, prefix in (("smoke", "smoke"), ("baseline", "baseline"), ("distill", "distill_"), ("merge", "merge_"), ("align", "align"), ("eval", "eval")):
        if any(name == prefix or name.startswith(prefix) for name in missing):
            stages.append(stage)
    if stages:
        stages.append("analyze")
    return stages


STAGES = {
    "unit": stage_unit,
    "smoke": stage_smoke,
    "baseline": stage_baseline,
    "distill": stage_distill,
    "merge": stage_merge,
    "align": stage_align,
    "eval": stage_eval,
    "analyze": stage_analyze,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ATML PA4 task 3 stages")
    parser.add_argument("stage", choices=list(STAGES))
    args = parser.parse_args(argv)
    STAGES[args.stage]()


if __name__ == "__main__":
    main(sys.argv[1:])
