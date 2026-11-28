"""Task 3 stages. Unit tests stay on CPU. Every stage that would download Qwen refuses without CUDA.

Order: unit, prefix_banks, smoke, baseline, distill, merge, align, eval,
analyze. `prefix_banks` must finish before `align`. `align` does forward
passes and frozen-rollout generation only (no speed numbers), so it may run on
a second GPU (CUDA_VISIBLE_DEVICES=1) while `eval` times on the first. Every
expensive piece is cached under results/task3/ as soon as it finishes; with
PA4_DEADLINE (unix seconds) set, a stage exits with code 75 before starting a
piece, and a rerun resumes from the cache. Each stage checks its inputs before
loading any model.

Smoke mode (`--smoke` or PA4_SMOKE=1) runs the same code on 4 evaluation
prompts with small token caps and quotas (see SMOKE_SETTINGS), writing only to
results/task3_smoke/ and checkpoints/task3_smoke/.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.paths import course_data_root
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
    assert_no_eval_leak,
    assert_same_token_ids,
    default_data_paths,
    load_eval_pool,
    load_training_pool,
    merge_adapter_into_fresh_backbone,
    render_chat,
    sample_response,
    train_method,
)
from task3.src.evaluate import (
    DIAGNOSTIC_MATH_PROMPTS,
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
    repeated_ngram_fraction,
    repetition_order,
    results_dir,
    score_standalone,
    timed_speculative,
    warm_setup,
)
from task3.src.runtime import SMOKE_ENV, SMOKE_MARKER, is_smoke, json_default, mark

OUT_OF_BUDGET = 75
PARTS_VERSION = "task3-parts-v3"
WRAPPER_RUN = ("off_policy", 6304)
WRAPPER_PROMPTS_PER_DOMAIN = 8
CONTINUATION_PROMPTS_PER_DOMAIN = 2
DIAGNOSTIC_PROMPTS = DIAGNOSTIC_MATH_PROMPTS
EXPECTED_EVAL_COUNTS = {"math": 60, "familiar": 30, "shifted": 30}
EXPECTED_TRAIN_COUNTS = {"math": 500, "short": 300}
BANK_EQUIVALENCE_TOL = 0.02

# Smoke mode shrinks sizes only. Every stage, model, bank and code path still runs.
SMOKE_EVAL_PER_DOMAIN = {"math": 2, "familiar": 1, "shifted": 1}
SMOKE_SETTINGS = {
    "eval_prompts_per_domain": SMOKE_EVAL_PER_DOMAIN,
    "prefix_bank_prompts": "the 4 smoke evaluation prompts, with their full-list seeds; 192-token cap kept",
    "run.SPEED_MAX_NEW": 24,
    "run.DIAGNOSTIC_MAX_NEW": 24,
    "run.DIAGNOSTIC_PROMPTS": 2,
    "run.WRAPPER_PROMPTS_PER_DOMAIN": 1,
    "evaluate.MATH_ANSWER_MAX_NEW": 64,
    "distill.MATH_QUOTA": 320,
    "distill.SHORT_QUOTA": 64,
    "unchanged": "gammas (2, 4), 2 paired repetitions, 192-token bank caps, 256-token updates, FP32 gate, all 5 drafts",
}


def smoke_overrides() -> list[tuple[object, str, object]]:
    """(module, attribute, value) for every size a smoke run shrinks."""
    from task3.src import distill, evaluate

    this = sys.modules[__name__]
    return [
        (this, "SPEED_MAX_NEW", SMOKE_SETTINGS["run.SPEED_MAX_NEW"]),
        (this, "DIAGNOSTIC_MAX_NEW", SMOKE_SETTINGS["run.DIAGNOSTIC_MAX_NEW"]),
        (this, "DIAGNOSTIC_PROMPTS", SMOKE_SETTINGS["run.DIAGNOSTIC_PROMPTS"]),
        (this, "WRAPPER_PROMPTS_PER_DOMAIN", SMOKE_SETTINGS["run.WRAPPER_PROMPTS_PER_DOMAIN"]),
        (evaluate, "MATH_ANSWER_MAX_NEW", SMOKE_SETTINGS["evaluate.MATH_ANSWER_MAX_NEW"]),
        (distill, "MATH_QUOTA", SMOKE_SETTINGS["distill.MATH_QUOTA"]),
        (distill, "SHORT_QUOTA", SMOKE_SETTINGS["distill.SHORT_QUOTA"]),
    ]


def configure_smoke() -> None:
    os.environ[SMOKE_ENV] = "1"
    for module, name, value in smoke_overrides():
        setattr(module, name, value)
    print(f"SMOKE MODE: writing to {results_dir()} and {checkpoint_dir()}; {SMOKE_SETTINGS}", flush=True)


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


def _read_part(path: Path):
    """The cached value, or None when the file is missing, unreadable, stale or from the other mode."""
    if not path.is_file():
        return None
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                stored = json.load(handle)
        else:
            stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, EOFError, ValueError) as error:
        print(f"{path.name} is unreadable ({error!r}); recomputing", flush=True)
        return None
    if not isinstance(stored, dict) or stored.get("version") != PARTS_VERSION or "value" not in stored:
        print(f"{path.name} is from another code version; recomputing", flush=True)
        return None
    if bool(stored.get(SMOKE_MARKER)) != is_smoke():
        print(f"{path.name} belongs to the other (smoke/full) mode; recomputing", flush=True)
        return None
    return stored


def _part(subdir: str, name: str, compute, *, gz: bool = False):
    """Cache one deterministic piece of a stage so a timed-out run resumes.

    Writes are atomic, so a killed process leaves either the old file or none.
    Parts that are unreadable or written by an older code version are recomputed.
    """
    path = results_dir() / subdir / (f"{name}.json.gz" if gz else f"{name}.json")
    stored = _read_part(path)
    if stored is not None:
        print(f"{subdir}: reusing {name}", flush=True)
        return stored["value"]
    _check_deadline(f"{subdir} part {name}")
    started = time.time()
    value = compute()
    payload = mark({"value": value, "version": PARTS_VERSION})
    if gz:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, default=json_default)
        os.replace(tmp, path)
    else:
        _dump(path, payload)
    print(f"{subdir}: {name} done in {time.time() - started:.0f}s", flush=True)
    return value


def _part_is_current(path: Path) -> bool:
    return _read_part(path) is not None


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


# ---------------------------------------------------------------- inputs


def _domain_counts(examples) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.domain] = counts.get(example.domain, 0) + 1
    return counts


def _validate_eval_pool(examples) -> None:
    counts = _domain_counts(examples)
    if counts != EXPECTED_EVAL_COUNTS:
        raise SystemExit(f"evaluation pool has {counts}, expected {EXPECTED_EVAL_COUNTS}")
    ids = [example.source_id for example in examples]
    if len(set(ids)) != len(ids) or any(not example.text.strip() for example in examples):
        raise SystemExit("evaluation pool has duplicate ids or empty prompts")


def _smoke_subset(examples) -> list:
    chosen = []
    for domain, count in SMOKE_EVAL_PER_DOMAIN.items():
        chosen.extend([example for example in examples if example.domain == domain][:count])
    order = {example.source_id: index for index, example in enumerate(examples)}
    return sorted(chosen, key=lambda example: order[example.source_id])


def _inputs(load):
    """Call a data loader; a missing or malformed input file stops the stage before any model load."""
    try:
        return load()
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"input check failed: {error!r}") from error


def _eval_examples() -> list:
    """The validated 120-prompt evaluation pool (the 4-prompt subset in smoke mode)."""
    examples = _inputs(load_eval_pool)
    _validate_eval_pool(examples)
    return _smoke_subset(examples) if is_smoke() else examples


def _adapter_complete(run_dir: Path) -> bool:
    adapter = run_dir / "adapter"
    return (run_dir / "train_log.json").is_file() and (adapter / "adapter_config.json").is_file() and any(adapter.glob("adapter_model.*"))


def _merged_complete(merged: Path) -> bool:
    return (merged / "merge_error.json").is_file() and (merged / "config.json").is_file() and any(merged.glob("*.safetensors"))


def _run_dir(method: str, seed: int) -> Path:
    return checkpoint_dir() / method / f"seed{seed}"


def _draft_names() -> list[str]:
    return ["original"] + [f"{method}_seed{seed}" for method in METHODS for seed in TRAIN_SEEDS]


def _merged_dir(name: str) -> Path:
    method, seed = name.rsplit("_seed", 1)
    return checkpoint_dir() / method / f"seed{seed}" / "merged"


def _require_merged_drafts() -> None:
    missing = [str(_merged_dir(name)) for name in _draft_names()[1:] if not _merged_complete(_merged_dir(name))]
    if missing:
        raise SystemExit(f"missing or incomplete merged drafts {missing}. Run the merge stage first.")


def prefix_bank_path() -> Path:
    """The supplied bank lives in the data repo; a smoke bank stays inside the smoke results."""
    from task3.src.prefix_banks import BANK_FILE

    if is_smoke():
        return results_dir() / "prefix_banks" / BANK_FILE
    return default_data_paths()["prefix_banks"]


def _validate_bank(path: Path, examples) -> list[dict]:
    """Both supplied banks must cover exactly the evaluation prompts, with identical prompt text."""
    from task3.src import alignment as al

    rows = _inputs(lambda: al.load_prefix_bank(path))
    text = {example.source_id: example.text for example in examples}
    for bank in al.SUPPLIED_BANKS:
        ids = {row["prompt_id"] for row in rows if row["bank"] == bank}
        if ids != set(text):
            raise SystemExit(
                f"{path}: bank {bank} covers {len(ids)} prompts, expected {len(text)} "
                f"(missing {sorted(set(text) - ids)[:5]}, extra {sorted(ids - set(text))[:5]})"
            )
    mismatch = sorted({row["prompt_id"] for row in rows if row["prompt"] != text[row["prompt_id"]]})
    if mismatch:
        raise SystemExit(f"{path}: prompt text differs from the evaluation pool for {mismatch[:5]}")
    return rows


# ---------------------------------------------------------------- stages


def stage_unit() -> None:
    import unittest

    suite = unittest.defaultTestLoader.discover("task3/tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def _bank_plan():
    """Revisions, the course prompt list (checked against the evaluation pool) and the prompts to build."""
    from task3.src import prefix_banks as pb

    root = course_data_root()
    revisions = _inputs(lambda: pb.load_revisions(root))
    prompts = _inputs(lambda: pb.bank_prompts(root))
    examples = _inputs(load_eval_pool)
    _validate_eval_pool(examples)
    if [(pid, text) for pid, text, _domain in prompts] != [(example.source_id, example.text) for example in examples]:
        raise SystemExit("course prefix-bank prompts differ from the evaluation pool (ids, order or text)")
    indexed = [(index, pid, text, domain) for index, (pid, text, domain) in enumerate(prompts)]
    if is_smoke():
        keep = {example.source_id for example in _smoke_subset(examples)}
        indexed = [item for item in indexed if item[1] in keep]
    return revisions, prompts, indexed


def _load_bank_tokenizers(revisions: dict):
    from transformers import AutoTokenizer

    from task3.src.prefix_banks import REPOS
    from task3.src.runtime import with_retries

    return tuple(
        with_retries(lambda key=key: AutoTokenizer.from_pretrained(REPOS[key], revision=revisions[key]), f"tokenizer {REPOS[key]}")
        for key in ("model_target", "model_draft")
    )


def _load_bank_model(key: str, revisions: dict):
    """Exactly the course load: pinned revision, FP16, SDPA, CUDA, eval mode."""
    import torch
    from transformers import AutoModelForCausalLM

    from task3.src.prefix_banks import REPOS
    from task3.src.runtime import with_retries

    model = with_retries(
        lambda: AutoModelForCausalLM.from_pretrained(
            REPOS[key], revision=revisions[key], torch_dtype=torch.float16, attn_implementation="sdpa"
        ),
        f"model {REPOS[key]}",
    )
    return model.to("cuda").eval()


def _build_one_bank(bank: str, key: str, revisions: dict, tokenizer, indexed) -> dict:
    from task3.src import prefix_banks as pb

    print(f"prefix_banks: loading {pb.REPOS[key]} for the {bank} bank", flush=True)
    model = _load_bank_model(key, revisions)
    settings = pb.sampling_settings(model, _eos(tokenizer))
    check = pb.generate_equivalence(model, pb.chat_ids(tokenizer, indexed[0][2]), settings)
    print(f"prefix_banks: {bank} settings {settings}; generate() equivalence {check['max_abs_prob_gap']}", flush=True)
    if check["max_abs_prob_gap"] is None or check["max_abs_prob_gap"] > BANK_EQUIVALENCE_TOL:
        raise RuntimeError(f"explicit sampler differs from generate() for {bank}: {check}")
    started = time.time()
    records = pb.generate_bank(model, tokenizer, indexed, bank, settings, progress=lambda text: print(text, flush=True))
    seconds = time.time() - started
    del model
    _free()
    return {"records": records, "settings": settings, "generate_equivalence": check, "generation_seconds": seconds}


def stage_prefix_banks() -> None:
    """Rebuild the supplied teacher / original-draft banks (course prefix_banks(), explicit sampler)."""
    import torch
    import transformers

    from task3.src import prefix_banks as pb

    revisions, prompts, indexed = _bank_plan()
    dest = prefix_bank_path()
    if dest.is_file():
        rows = _validate_bank(dest, [example for example in _eval_examples()])
        print(f"prefix_banks: {dest} exists with {len(rows)} valid records; skipping", flush=True)
        return
    require_cuda("prefix_banks")
    started = time.time()
    target_tokenizer, draft_tokenizer = _load_bank_tokenizers(revisions)
    pb.check_tokenizers(target_tokenizer, draft_tokenizer)
    template_mismatch = [pid for _index, pid, text, _domain in indexed if pb.chat_ids(target_tokenizer, text) != render_chat(target_tokenizer, text)]
    if template_mismatch:
        raise SystemExit(f"course chat template differs from the evaluation template for {template_mismatch[:5]}")
    built = {
        bank: _part("prefix_bank_parts", bank, lambda bank=bank, key=key: _build_one_bank(bank, key, revisions, target_tokenizer, indexed), gz=True)
        for bank, key, _offset in pb.BANKS
    }
    records = [record for bank, _key, _offset in pb.BANKS for record in built[bank]["records"]]
    pb.write_bank(dest, records)
    meta = {
        "bank_file": str(dest),
        "n_records": len(records),
        "n_prompts_built": len(indexed),
        "n_prompts_course_list": len(prompts),
        "prompts_sha256": pb.prompts_digest(prompts),
        "prompt_source": "data/public_locked/gsm8k_test_locked.jsonl.gz (sha256-checked GSM8K test rows of the course manifest) + data/short_form/short_test.jsonl",
        "course_function": pb.COURSE_SCRIPT,
        "deviation": pb.DEVIATION,
        "sampling": {bank: built[bank]["settings"] for bank in built},
        "generate_equivalence_check": {bank: built[bank]["generate_equivalence"] for bank in built},
        "generation_seconds": {bank: built[bank]["generation_seconds"] for bank in built},
        "summary": pb.bank_summary(records),
        "revisions": revisions,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "note": "Qwen2.5 generation_config keeps repetition_penalty=1.1 under generate(); the teacher bank therefore "
        "follows the penalized target law. align adds an unpenalized target bank as a consistency control.",
        "stage_seconds": time.time() - started,
    }
    _dump(dest.with_name(pb.META_FILE), meta)
    _dump(results_dir() / pb.META_FILE, meta)
    print(f"wrote {dest} ({len(records)} records)", flush=True)


def stage_smoke() -> None:
    require_cuda("smoke")
    from task3.src.evaluate import run_smoke

    run_smoke()


def _export_existing(source: Path, target: Path) -> None:
    """Copy evidence written next to a checkpoint into results/ when only the checkpoint copy exists."""
    if source.is_file() and not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def stage_distill() -> None:
    from task3.src.distill import distill_results_dir

    math_examples, short_examples = _inputs(load_training_pool)
    counts = {"math": len(math_examples), "short": len(short_examples)}
    if counts != EXPECTED_TRAIN_COUNTS:
        raise SystemExit(f"training pool has {counts}, expected {EXPECTED_TRAIN_COUNTS}")
    assert_no_eval_leak([*math_examples, *short_examples], _inputs(load_eval_pool))
    require_cuda("distill")
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            output = _run_dir(method, seed)
            if _adapter_complete(output):
                _export_existing(output / "train_log.json", distill_results_dir(method, seed) / "train_log.json")
                print(f"distill {method} seed {seed}: adapter exists, skipping", flush=True)
                continue
            if (output / "adapter").exists():
                print(f"distill {method} seed {seed}: removing an incomplete adapter", flush=True)
                shutil.rmtree(output / "adapter")
            _check_deadline(f"distill {method} seed {seed}")
            print(f"distilling {method} seed {seed}", flush=True)
            train_method(method, seed, output)


def merge_report_path(method: str, seed: int) -> Path:
    return results_dir() / "merge" / f"{method}_seed{seed}" / "merge_error.json"


def stage_merge() -> None:
    missing = [str(_run_dir(m, s)) for m in METHODS for s in TRAIN_SEEDS if not _adapter_complete(_run_dir(m, s))]
    if missing:
        raise SystemExit(f"missing or incomplete adapters {missing}. Run the distill stage first.")
    require_cuda("merge")
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            adapter = _run_dir(method, seed) / "adapter"
            merged = _run_dir(method, seed) / "merged"
            report = merge_report_path(method, seed)
            if _merged_complete(merged):
                _export_existing(merged / "merge_error.json", report)
                print(f"merge {method} seed {seed}: merged draft exists, skipping", flush=True)
                continue
            if merged.exists():
                print(f"merge {method} seed {seed}: removing an incomplete merged draft", flush=True)
                shutil.rmtree(merged)
            _check_deadline(f"merge {method} seed {seed}")
            print(f"merging {method} seed {seed}", flush=True)
            merge_adapter_into_fresh_backbone(adapter, merged, report_path=report)
            _free()


def _load_target():
    from task3.src.distill import load_frozen_causal

    return load_frozen_causal(TARGET_MODEL)


def _load_draft(name: str, tokenizer):
    """The original 0.5B draft, or a merged FP16 checkpoint. Token ids must match the target."""
    from task3.src.distill import load_frozen_causal

    if name != "original" and not _merged_complete(_merged_dir(name)):
        raise SystemExit(f"missing or incomplete merged draft {_merged_dir(name)}. Run the merge stage first.")
    source = DRAFT_MODEL if name == "original" else str(_merged_dir(name))
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


def _diagnostic_prompts(examples) -> list:
    try:
        return diagnostic_math_prompts(examples, DIAGNOSTIC_PROMPTS)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


def stage_baseline() -> None:
    examples = _eval_examples()
    _diagnostic_prompts(examples)
    require_cuda("baseline")
    target, tokenizer = _load_target()
    draft = _load_draft("original", tokenizer)
    report = _part(
        "baseline_parts",
        "primary",
        lambda: _benchmark(target, draft, tokenizer, examples, label="original_draft", max_new=SPEED_MAX_NEW, ignore_eos=False, seed=6304),
    )
    diagnostic = _part(
        "baseline_parts",
        "math_diagnostic_64_ignore_eos",
        lambda: _benchmark(
            target, draft, tokenizer, _diagnostic_prompts(examples), label="math_diagnostic_64",
            max_new=DIAGNOSTIC_MAX_NEW, ignore_eos=True, seed=6304,
        ),
    )
    payload = {
        "primary": report,
        "diagnostic_ignore_eos": diagnostic,
        "diagnostic_note": f"{DIAGNOSTIC_PROMPTS} math prompts, exactly {DIAGNOSTIC_MAX_NEW} new tokens with EOS ignored, timing only; not an answer-quality run.",
        "target": TARGET_MODEL,
        "draft": DRAFT_MODEL,
    }
    _dump(results_dir() / "baseline.json", payload)
    print(f"wrote {results_dir() / 'baseline.json'}", flush=True)


def stage_align() -> None:
    """Common-prefix alignment of every draft on the shared and student banks (no timing)."""
    from task3.src import alignment as al

    examples = _eval_examples()
    bank_path = prefix_bank_path()
    supplied = _validate_bank(bank_path, examples)
    _require_merged_drafts()
    require_cuda("align")

    target, tokenizer = _load_target()
    drafts = {name: _load_draft(name, tokenizer) for name in _draft_names()}
    banks: dict[str, list[dict]] = {}
    for row in supplied:
        banks.setdefault(row["bank"], []).append(row)
    banks[al.SPEC_BANK] = _part("align_parts", "bank_exact_spec_original_gamma4", lambda: al.exact_spec_bank(target, drafts["original"], tokenizer, examples), gz=True)
    banks[al.TARGET_LAW_BANK] = _part("align_parts", f"bank_{al.TARGET_LAW_BANK}", lambda: al.target_law_bank(target, tokenizer, examples), gz=True)
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
        "drafts": {name: (DRAFT_MODEL if name == "original" else str(_merged_dir(name))) for name in drafts},
        "summary": al.summarize_scores(all_rows),
        "target_vs_speculative_consistency": al.consistency_check(all_rows),
        "target_law_control_consistency": al.consistency_check(all_rows, bank_a=al.TARGET_LAW_BANK),
        "consistency_note": al.CONSISTENCY_NOTE,
        "aligned_disagreements": al.decode_disagreements(shared_scores, shared_prefixes, tokenizer),
    }
    _dump(results_dir() / "alignment.json", payload)
    path = results_dir() / "alignment_rows.jsonl.gz"
    tmp = path.with_name(f".{path.name}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for row in al.compact(all_rows):
            handle.write(json.dumps(row, default=json_default) + "\n")
    os.replace(tmp, path)
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
    examples = _eval_examples()
    _diagnostic_prompts(examples)
    _require_merged_drafts()
    if not _adapter_complete(_run_dir(*WRAPPER_RUN)):
        raise SystemExit(f"missing adapter {_run_dir(*WRAPPER_RUN)} for the wrapper-overhead run. Run distill first.")
    require_cuda("eval")
    target, tokenizer = _load_target()
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
                target, draft, tokenizer, _diagnostic_prompts(examples), label=f"{name}_math_diagnostic_64",
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
    """A stage output, or None. Smoke outputs never enter a full-run analysis."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get(SMOKE_MARKER) and not is_smoke():
        raise SystemExit(f"{path} is a smoke-run output and cannot enter the full analysis")
    return payload


def _smoke_coverage(report: dict, outputs: dict) -> dict[str, bool]:
    """Every code path a full run needs, as booleans; a smoke run fails unless all are true."""
    from task3.src import alignment as al

    banks = outputs.get("prefix_banks") or {}
    baseline = outputs.get("baseline") or {}
    alignment = outputs.get("align") or {}
    evaluation = outputs.get("eval") or {}
    logs = outputs.get("logs", {})
    return {
        "prefix_banks_both_banks": set((banks.get("summary") or {})) == set(al.SUPPLIED_BANKS)
        and all(value["n_records"] > 0 for value in banks["summary"].values()),
        "prefix_banks_generate_equivalence": all(
            (value or {}).get("max_abs_prob_gap") is not None for value in (banks.get("generate_equivalence_check") or {"x": None}).values()
        ),
        "fp32_gate_passed": bool((outputs.get("smoke") or {}).get("passed")),
        "baseline_both_gammas": set((baseline.get("primary") or {}).get("speed", {})) == {str(g) for g in GAMMAS},
        "baseline_ignore_eos_diagnostic": bool((baseline.get("diagnostic_ignore_eos") or {}).get("ignore_eos")),
        "distill_four_runs": len(logs) == len(METHODS) * len(TRAIN_SEEDS),
        "distill_full_and_short_updates": bool(logs) and all(log.get("optimizer_steps", 0) >= 2 and log.get("short_final_update") for log in logs.values()),
        "distill_on_policy_rollouts": bool(logs) and all(log.get("student_rollout_s", 0) > 0 for name, log in logs.items() if name.startswith("on_policy")),
        "distill_off_policy_teacher_pool": bool(logs) and all(log.get("offline_teacher_response_preparation_s", 0) > 0 for name, log in logs.items() if name.startswith("off_policy")),
        "merge_reports": len(outputs.get("merges", {})) == len(METHODS) * len(TRAIN_SEEDS),
        "align_all_banks": {*al.SUPPLIED_BANKS, al.SPEC_BANK, al.TARGET_LAW_BANK, *(f"student_{n}" for n in _draft_names()[1:])} <= set(alignment.get("summary", {})),
        "align_consistency": (alignment.get("target_vs_speculative_consistency") or {}).get("n_comparisons", 0) > 0,
        "eval_all_drafts": set(evaluation.get("benchmarks", {})) == set(_draft_names()) and set(evaluation.get("standalone_accuracy", {})) == set(_draft_names()),
        "eval_wrapper_overhead": bool((evaluation.get("wrapper_overhead") or {}).get("rows")),
        "eval_continuations": bool((evaluation.get("continuations") or {}).get("examples")),
        "analyze_bootstrap": all("bootstrap" in (speed or {}).get(str(GAMMAS[0]), {}) for speed in report.get("eval_speed", {"x": None}).values()),
    }


def stage_analyze() -> None:
    """Summarize saved GPU artifacts on CPU. Missing files are reported; nothing is trained here."""
    from task3.src.distill import distill_results_dir
    from task3.src.prefix_banks import META_FILE

    expected = {
        "prefix_banks": results_dir() / META_FILE,
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
    banks = _read(expected["prefix_banks"])
    if banks:
        report["prefix_banks"] = {key: banks.get(key) for key in ("n_records", "summary", "sampling", "generate_equivalence_check", "deviation", "note")}
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
        for key in ("target_vs_speculative_consistency", "target_law_control_consistency"):
            if alignment.get(key):
                report[key] = {field: alignment[key].get(field) for field in ("bank_a", "bank_b", "n_comparisons", "n_ci_excluding_zero", "overall")}
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
    failed: list[str] = []
    if is_smoke():
        outputs = {"prefix_banks": banks, "smoke": smoke, "baseline": baseline, "align": alignment, "eval": evaluation, "logs": logs, "merges": merges}
        report["smoke_settings"] = SMOKE_SETTINGS
        report["smoke_coverage"] = _smoke_coverage(report, outputs)
        failed = [name for name, ok in report["smoke_coverage"].items() if not ok]
    _dump(results_dir() / "analyze.json", report)
    print(json.dumps({"missing": missing, "gpu_stages_remaining": report["gpu_stages_remaining"]}, indent=2))
    if is_smoke():
        if missing or failed:
            raise SystemExit(f"SMOKE FAILED: missing outputs {missing}, uncovered paths {failed}")
        if checkpoint_dir().is_dir():
            shutil.rmtree(checkpoint_dir())
            print(f"removed smoke checkpoints {checkpoint_dir()}", flush=True)
        print("SMOKE PASSED: every stage and code path ran on the real models", flush=True)


def _remaining_stages(missing: list[str]) -> list[str]:
    stages = []
    for stage, prefix in (
        ("prefix_banks", "prefix_banks"),
        ("smoke", "smoke"),
        ("baseline", "baseline"),
        ("distill", "distill_"),
        ("merge", "merge_"),
        ("align", "align"),
        ("eval", "eval"),
    ):
        if any(name == prefix or name.startswith(prefix) for name in missing):
            stages.append(stage)
    if stages:
        stages.append("analyze")
    return stages


STAGES = {
    "unit": stage_unit,
    "prefix_banks": stage_prefix_banks,
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
    parser.add_argument("--smoke", action="store_true", help="reduced sizes, separate results/checkpoints (also PA4_SMOKE=1)")
    args = parser.parse_args(argv)
    if args.smoke or is_smoke():
        configure_smoke()
    STAGES[args.stage]()


if __name__ == "__main__":
    main(sys.argv[1:])
