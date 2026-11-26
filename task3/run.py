"""Task 3 stages. Unit tests stay on CPU. Every stage that would download Qwen refuses without CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task3.src.distill import (
    DRAFT_MODEL,
    METHODS,
    TARGET_MODEL,
    TRAIN_SEEDS,
    assert_same_token_ids,
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
    answer_cap,
    benchmark_prompt,
    checkpoint_dir,
    decode_text,
    diagnostic_math_prompts,
    estimate_speed_model,
    evaluate_draft_on_bank,
    load_prefix_bank,
    load_result,
    paired_bootstrap_report,
    prefix_rows_from_generation,
    project_speedup,
    render_chat,
    results_dir,
    score_standalone,
    summarize_alignment,
    _dump,
    _prompt_seed,
)
from task3.src.distill import default_data_paths


def require_cuda(stage: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{stage} requires CUDA and will not download weights or run on CPU.")


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


def stage_distill() -> None:
    require_cuda("distill")
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            output = checkpoint_dir() / method / f"seed{seed}"
            print(f"distilling {method} seed {seed}", flush=True)
            train_method(method, seed, output)


def _load_pair():
    from task3.src.distill import load_frozen_causal

    target, target_tokenizer = load_frozen_causal(TARGET_MODEL)
    draft, draft_tokenizer = load_frozen_causal(DRAFT_MODEL)
    assert_same_token_ids(target_tokenizer, draft_tokenizer)
    return target, target_tokenizer, draft, draft_tokenizer


def _benchmark(target, draft, tokenizer, examples, *, label: str, max_new: int, ignore_eos: bool, seed: int) -> dict:
    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    decoder = SpeculativeDecoder(target, draft, np.random.default_rng(seed))
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
                rows.append(record)
    return {"label": label, "ignore_eos": ignore_eos, "max_new_tokens": max_new, "isolated": isolated, "rows": rows}


def _acceptance(row: dict) -> float:
    value = row["speculative"].get("accepted_over_proposed")
    return 0.0 if value is None else float(value)


def _attach_speed(report: dict) -> None:
    grouped: dict[int, list[dict]] = {}
    for row in report["rows"]:
        grouped.setdefault(int(row["gamma"]), []).append(row)
    projections = {}
    for gamma, rows in grouped.items():
        ratios = report["isolated"][str(gamma)]
        per_prompt = []
        # Average the two repetitions of each prompt before the paired bootstrap.
        by_prompt: dict[str, list[dict]] = {}
        for row in rows:
            by_prompt.setdefault(row["prompt_id"], []).append(row)
        for prompt_rows in by_prompt.values():
            per_prompt.append(
                {
                    "speculative_tps": float(np.mean([item["speculative"]["tokens_per_second"] for item in prompt_rows])),
                    "target_tps": float(np.mean([item["target_only"]["tokens_per_second"] for item in prompt_rows])),
                    "acceptance": float(np.mean([_acceptance(item) for item in prompt_rows])),
                }
            )
        acceptance = float(np.mean([item["acceptance"] for item in per_prompt]))
        projections[str(gamma)] = {
            **project_speedup(acceptance, gamma, ratios["c"], ratios["v_gamma"]),
            "measured_tokens_per_second_speculative": float(np.mean([item["speculative_tps"] for item in per_prompt])),
            "measured_tokens_per_second_target": float(np.mean([item["target_tps"] for item in per_prompt])),
            "bootstrap": paired_bootstrap_report(per_prompt, seed=6304),
        }
        mean_spec = projections[str(gamma)]["measured_tokens_per_second_speculative"]
        mean_target = projections[str(gamma)]["measured_tokens_per_second_target"]
        projections[str(gamma)]["measured_speedup"] = None if mean_target <= 0 else mean_spec / mean_target
    report["speed"] = projections


def stage_baseline() -> None:
    require_cuda("baseline")
    target, tokenizer, draft, _draft_tokenizer = _load_pair()
    examples = load_eval_pool()
    report = _benchmark(target, draft, tokenizer, examples, label="original_draft", max_new=SPEED_MAX_NEW, ignore_eos=False, seed=6304)
    diagnostic = _benchmark(
        target,
        draft,
        tokenizer,
        diagnostic_math_prompts(examples),
        label="math_diagnostic_64",
        max_new=DIAGNOSTIC_MAX_NEW,
        ignore_eos=True,
        seed=6304,
    )
    _attach_speed(report)
    _attach_speed(diagnostic)
    payload = {"primary": report, "diagnostic_ignore_eos": diagnostic, "target": TARGET_MODEL, "draft": DRAFT_MODEL}
    _dump(results_dir() / "baseline.json", payload)
    print(f"wrote {results_dir() / 'baseline.json'}", flush=True)


def stage_merge() -> None:
    require_cuda("merge")
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            adapter = checkpoint_dir() / method / f"seed{seed}" / "adapter"
            merged = checkpoint_dir() / method / f"seed{seed}" / "merged"
            if not adapter.is_dir():
                raise SystemExit(f"missing adapter {adapter}. Run the distill stage first.")
            print(f"merging {method} seed {seed}", flush=True)
            merge_adapter_into_fresh_backbone(adapter, merged)


def _standalone(model, tokenizer, examples) -> list[dict]:
    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
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
        scored = score_standalone(example.domain, decode_text(tokenizer, tokens), example.answer, length_capped=reason == "length")
        scored["source_id"] = example.source_id
        scored["domain"] = example.domain
        scored["termination_reason"] = reason
        rows.append(scored)
    return rows


def _exact_spec_rows(target, draft, tokenizer, examples) -> list[dict]:
    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    decoder = SpeculativeDecoder(target, draft, np.random.default_rng(6304))
    rows = []
    for example in examples:
        prompt_ids = render_chat(tokenizer, example.text)
        decoder.prefill(prompt_ids)
        generated = decoder.generate(gamma=4, max_new_tokens=192, eos_id=eos_id)
        rows.extend(
            prefix_rows_from_generation(
                example.source_id,
                example.text,
                example.domain,
                generated.tokens,
                "exact_spec_original_gamma4",
                producer="original_draft_gamma4",
            )
        )
    return rows


def stage_eval() -> None:
    require_cuda("eval")
    from task3.src.distill import load_frozen_causal

    target, tokenizer, original_draft, _draft_tokenizer = _load_pair()
    examples = load_eval_pool()
    bank_path = default_data_paths()["prefix_banks"]
    supplied = load_prefix_bank(bank_path) if bank_path.is_file() else []
    exact_rows = _exact_spec_rows(target, original_draft, tokenizer, examples)
    drafts = [("original", original_draft)]
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            merged_dir = checkpoint_dir() / method / f"seed{seed}" / "merged"
            if not merged_dir.is_dir():
                raise SystemExit(f"missing merged draft {merged_dir}. Run the merge stage first.")
            model, model_tokenizer = load_frozen_causal(str(merged_dir))
            assert_same_token_ids(tokenizer, model_tokenizer)
            drafts.append((f"{method}_seed{seed}", model))
    alignment = {}
    shared_banks = supplied + exact_rows
    for name, model in drafts:
        scored = evaluate_draft_on_bank(target, model, tokenizer, shared_banks, name)
        alignment[name] = summarize_alignment(scored)
    # Each trained student's own prefixes, then both students of every method on
    # each frozen rollout bank. Comparisons stay inside one bank's prefix ids.
    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    student_banks: dict[str, list[dict]] = {}
    for name, model in drafts:
        if name == "original":
            continue
        seed = int(name.rsplit("seed", 1)[1])
        rows = []
        for index, example in enumerate(examples):
            prompt_ids = render_chat(tokenizer, example.text)
            tokens, _reason = sample_response(
                model,
                prompt_ids,
                max_new_tokens=192,
                eos_id=eos_id,
                rng=np.random.default_rng(seed + 10007 * index),
            )
            rows.extend(
                prefix_rows_from_generation(
                    example.source_id,
                    example.text,
                    example.domain,
                    tokens,
                    f"student_{name}",
                    producer=name,
                )
            )
        student_banks[name] = rows
        _dump(results_dir() / f"prefixes_{name}.json", {"rows": rows})
    cross = {}
    for name, model in drafts:
        if name == "original":
            continue
        for other_name, rows in student_banks.items():
            scored = evaluate_draft_on_bank(target, model, tokenizer, rows, name)
            cross[f"{name}_on_{other_name}"] = summarize_alignment(scored)
    benchmarks = {}
    accuracy = {}
    for name, model in drafts:
        benchmarks[name] = _benchmark(
            target,
            model,
            tokenizer,
            examples,
            label=name,
            max_new=SPEED_MAX_NEW,
            ignore_eos=False,
            seed=6304,
        )
        _attach_speed(benchmarks[name])
        # Standalone student generations use the answer caps. Speculative rows are not scored.
        accuracy[name] = _standalone(model, tokenizer, examples)
    payload = {
        "alignment": alignment,
        "cross_student_alignment": cross,
        "benchmarks": benchmarks,
        "standalone_accuracy": accuracy,
        "n_supplied_prefixes": len(supplied),
        "n_exact_spec_prefixes": len(exact_rows),
    }
    _dump(results_dir() / "eval.json", payload)
    print(f"wrote {results_dir() / 'eval.json'}", flush=True)


def stage_analyze() -> None:
    """Summarize saved GPU artifacts. Missing files are reported; nothing is trained here."""
    expected = {
        "smoke": results_dir() / "smoke.json",
        "baseline": results_dir() / "baseline.json",
        "eval": results_dir() / "eval.json",
    }
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            expected[f"distill_{method}_{seed}"] = checkpoint_dir() / method / f"seed{seed}" / "train_log.json"
            expected[f"merge_{method}_{seed}"] = checkpoint_dir() / method / f"seed{seed}" / "merged" / "merge_error.json"
    missing = [name for name, path in expected.items() if not path.is_file()]
    report: dict = {
        "missing": missing,
        "gpu_stages_remaining": _remaining_stages(missing),
    }
    if (results_dir() / "baseline.json").is_file():
        baseline = load_result("baseline.json")
        report["baseline_speed"] = baseline.get("primary", {}).get("speed")
    if (results_dir() / "eval.json").is_file():
        evaluation = load_result("eval.json")
        report["eval_speed"] = {name: payload.get("speed") for name, payload in evaluation.get("benchmarks", {}).items()}
    _dump(results_dir() / "analyze.json", report)
    print(json.dumps({"missing": missing, "gpu_stages_remaining": report["gpu_stages_remaining"]}, indent=2))


def _remaining_stages(missing: list[str]) -> list[str]:
    stages = []
    if "smoke" in missing:
        stages.append("smoke")
    if "baseline" in missing:
        stages.append("baseline")
    if any(name.startswith("distill_") for name in missing):
        stages.append("distill")
    if any(name.startswith("merge_") for name in missing):
        stages.append("merge")
    if "eval" in missing:
        stages.append("eval")
    if stages:
        stages.append("analyze")
    return stages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ATML PA4 task 3 stages")
    parser.add_argument("stage", choices=["unit", "smoke", "baseline", "distill", "merge", "eval", "analyze"])
    args = parser.parse_args(argv)
    {
        "unit": stage_unit,
        "smoke": stage_smoke,
        "baseline": stage_baseline,
        "distill": stage_distill,
        "merge": stage_merge,
        "eval": stage_eval,
        "analyze": stage_analyze,
    }[args.stage]()


if __name__ == "__main__":
    main(sys.argv[1:])
