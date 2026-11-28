"""Speculative timing, smoke checks, and standalone answer checks (prefix metrics live in alignment.py).

Speculative samples follow the target distribution and are not used to score
the student. Standalone accuracy uses the answer-length caps and the extractors
below. Latency sections use shared.timing so CUDA work is synchronized.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from shared.timing import cuda_sync, time_call
from task3.src import runtime
from task3.src.distill import (
    DRAFT_MODEL,
    TARGET_MODEL,
    TOKEN_CHECK_PROMPTS,
    PromptExample,
    assert_same_token_ids,
    render_chat,
)
from task3.src.metrics import cost_ratios
from task3.src.speculative import (
    KVCacheState,
    SpeculativeDecoder,
    cache_length,
    logits_to_probs,
    sample_categorical,
)

SPEED_MAX_NEW = 64
MATH_ANSWER_MAX_NEW = 512
SHORT_ANSWER_MAX_NEW = 32
DIAGNOSTIC_MATH_PROMPTS = 24
DIAGNOSTIC_MAX_NEW = 64
GAMMAS = (2, 4)
PAIRED_REPETITIONS = 2

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_GSM_MARK = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def results_dir() -> Path:
    """results/task3, or results/task3_smoke in a smoke run."""
    return runtime.results_dir()


def checkpoint_dir() -> Path:
    return runtime.checkpoint_dir()


def repetition_order(repetition: int) -> tuple[str, str]:
    """Counterbalance speculative and target-only runs across paired repetitions."""
    if repetition % 2 == 0:
        return ("target_only", "speculative")
    return ("speculative", "target_only")


def gsm8k_extraction(text: str | None) -> tuple[str | None, str]:
    """Final number and how it was found: boxed, #### marker, last number, or none."""
    if text is None or not str(text).strip():
        return None, "none"
    for content in reversed(_BOXED.findall(text)):
        found = _NUMBER.findall(content)
        if found:
            return found[-1].replace(",", ""), "boxed"
    marked = _GSM_MARK.findall(text)
    if marked:
        return marked[-1].replace(",", ""), "marker"
    found = _NUMBER.findall(text)
    if found:
        return found[-1].replace(",", ""), "last_number"
    return None, "none"


def extract_gsm8k_number(text: str | None) -> str | None:
    """Final number: a \\boxed{} answer, else a GSM8K `####` marker, else the last number."""
    return gsm8k_extraction(text)[0]


def numbers_equal(predicted: str | None, gold: str | None) -> bool:
    if predicted is None or gold is None:
        return False
    try:
        left = float(predicted)
        right = float(gold)
    except ValueError:
        return False
    return abs(left - right) <= 1e-6 * max(1.0, abs(right))


def _clean_line(line: str) -> str:
    line = line.strip().strip("`\"'* ")
    head, sep, tail = line.partition(":")
    # A short label such as "Answer:" or "Destination code:" before the value.
    if sep and tail.strip() and len(head.split()) <= 4:
        line = tail.strip().strip("`\"'* ")
    return line.rstrip(".").strip()


def extract_short_form(text: str | None) -> str | None:
    """First non-empty line, with a short leading label and a trailing period removed."""
    if text is None or not str(text).strip():
        return None
    line = _clean_line(str(text).strip().splitlines()[0])
    return line or None


def short_form_match(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    return predicted.casefold() == gold.strip().casefold()


def short_form_contains(text: str | None, gold: str) -> bool:
    """Lenient check: the gold answer appears as a whole word anywhere in the response."""
    if not text:
        return False
    return re.search(rf"(?<![\w]){re.escape(gold.strip())}(?![\w])", text, flags=re.IGNORECASE) is not None


def repeated_ngram_fraction(token_ids: Sequence[int], n: int = 4) -> float:
    """Share of n-grams that already occurred earlier in the response (a degeneration signal)."""
    grams = [tuple(token_ids[i : i + n]) for i in range(max(0, len(token_ids) - n + 1))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def score_standalone(domain: str, generated_text: str | None, gold: str, *, length_capped: bool) -> dict:
    """Score a student generation. Do not pass speculative target samples here.

    Math: the extracted final number must equal the GSM8K answer; format
    compliance means an explicit final-answer marker (boxed or ####). Short
    form: `correct` is a strict match of the cleaned first line; format
    compliance means a single non-empty line of at most three words;
    `correct_lenient` accepts the gold answer anywhere as a whole word.
    """
    if domain == "math":
        predicted, method = gsm8k_extraction(generated_text)
        correct = numbers_equal(predicted, extract_gsm8k_number(gold))
        compliant = method in ("boxed", "marker")
        lenient = correct
    else:
        predicted = extract_short_form(generated_text)
        method = "first_line" if predicted is not None else "none"
        correct = short_form_match(predicted, gold)
        lines = [line for line in str(generated_text or "").splitlines() if line.strip()]
        compliant = len(lines) == 1 and predicted is not None and len(predicted.split()) <= 3
        lenient = correct or short_form_contains(generated_text, gold)
    return {
        "prediction": predicted,
        "correct": correct,
        "correct_lenient": lenient,
        "extraction_method": method,
        "format_compliant": compliant,
        "unextractable": predicted is None,
        "truncated": length_capped,
    }


def answer_cap(domain: str) -> int:
    if domain == "math":
        return MATH_ANSWER_MAX_NEW
    return SHORT_ANSWER_MAX_NEW


def diagnostic_math_prompts(examples: Sequence[PromptExample], count: int = DIAGNOSTIC_MATH_PROMPTS) -> list[PromptExample]:
    math_examples = [example for example in examples if example.domain == "math"]
    if len(math_examples) < count:
        raise RuntimeError(f"need {count} math prompts, found {len(math_examples)}")
    return math_examples[:count]


def split_prefill_max_abs(model, token_ids: Sequence[int], split: int) -> float:
    """Max absolute logit gap between a full forward and a two-piece prefill."""
    if split <= 0 or split >= len(token_ids):
        raise ValueError("split must lie strictly inside the token sequence")
    device = next(model.parameters()).device
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        full = model(input_ids=ids, use_cache=False).logits
        first = model(input_ids=ids[:, :split], use_cache=True)
        position = torch.arange(split, ids.shape[1], device=device)
        second = model(
            input_ids=ids[:, split:],
            past_key_values=first.past_key_values,
            use_cache=True,
            cache_position=position,
        )
        joined = torch.cat([first.logits, second.logits], dim=1)
    return float((full.float() - joined.float()).abs().max().cpu())


def _cache_layers(cache) -> list[tuple]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def cache_kv_max_abs(model, cache, token_ids: Sequence[int]) -> float:
    """Largest gap between a rolled-back/advanced cache and a fresh one-pass cache.

    Every layer's keys and values at every position are compared, so a wrong
    rotary position or a stale entry left by a crop shows up here.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        fresh = model(input_ids=ids, use_cache=True).past_key_values
    worst = 0.0
    mine = _cache_layers(cache)
    theirs = _cache_layers(fresh)
    if len(mine) != len(theirs):
        raise RuntimeError("cache layer counts differ")
    for (key, value), (ref_key, ref_value) in zip(mine, theirs):
        if key.shape != ref_key.shape or value.shape != ref_value.shape:
            raise RuntimeError(f"cache shape {tuple(key.shape)} differs from fresh {tuple(ref_key.shape)}")
        worst = max(worst, float((key.float() - ref_key.float()).abs().max()), float((value.float() - ref_value.float()).abs().max()))
    return worst


def cache_matches_full_forward(target_model, draft_model, token_ids: Sequence[int], eos_id: int | None, max_new_tokens: int = 32) -> dict:
    """Run speculative rounds (with rollbacks), then compare both caches with a one-pass forward."""
    decoder = SpeculativeDecoder(target_model, draft_model, np.random.default_rng(6304))
    with torch.inference_mode():
        decoder.prefill(token_ids)
        result = decoder.generate(gamma=4, max_new_tokens=max_new_tokens, eos_id=eos_id)
        expected = list(token_ids) + list(result.tokens)
        pending_ok = (
            decoder.committed == expected
            and decoder.target.token_ids == expected[: decoder.target.seq_len]
            and decoder.draft.token_ids == expected[: decoder.draft.seq_len]
            and decoder.target.seq_len < len(expected)
            and decoder.draft.seq_len < len(expected)
        )
        decoder.sync_caches()
    length_ok = (
        pending_ok
        and decoder.target.token_ids == expected
        and decoder.draft.token_ids == expected
        and cache_length(decoder.target.cache) == len(expected)
        and cache_length(decoder.draft.cache) == len(expected)
    )
    device = next(target_model.parameters()).device
    ids = torch.tensor([expected], dtype=torch.long, device=device)
    with torch.inference_mode():
        full = target_model(input_ids=ids, use_cache=False).logits[0, -1].float()
    cached = decoder.target.next_logits.float()
    error = float((full - cached).abs().max().cpu())
    rounds = result.rounds
    return {
        "length_ok": length_ok,
        "next_logit_max_abs": error,
        "target_kv_max_abs": cache_kv_max_abs(target_model, decoder.target.cache, expected),
        "draft_kv_max_abs": cache_kv_max_abs(draft_model, decoder.draft.cache, expected),
        "generated_tokens": len(result.tokens),
        "cache_length": len(expected),
        "rounds": len(rounds),
        "rounds_with_rejection": sum(1 for item in rounds if item.rejected),
        "rounds_with_bonus": sum(1 for item in rounds if item.bonus_token is not None),
    }


def isolated_forward_seconds(model, prompt_len: int, n_tokens: int, *, warmup: int = 2, repeats: int = 10) -> float:
    """Mean seconds for one cached forward of `n_tokens`, cropped back afterwards.

    The crop keeps the measured length fixed across `time_call` repetitions.
    """
    if prompt_len < 1 or n_tokens < 1:
        raise ValueError("prompt_len and n_tokens must be positive")
    device = next(model.parameters()).device
    vocab = int(getattr(model.config, "vocab_size", 1024))
    prefix = torch.randint(0, min(vocab, 1000), (1, prompt_len), device=device)
    follow = torch.randint(0, min(vocab, 1000), (1, n_tokens), device=device)
    with torch.inference_mode():
        past_box = [model(input_ids=prefix, use_cache=True).past_key_values]

        def step() -> None:
            output = model(input_ids=follow, past_key_values=past_box[0], use_cache=True)
            past_box[0] = output.past_key_values
            past_box[0].crop(prompt_len)

        samples = time_call(step, warmup=warmup, repeats=repeats)
    return float(np.mean(samples))


def summarize_speculative(result, elapsed_s: float, peak_bytes: int) -> dict:
    """Per-prompt speculative record. Every forward is counted where it happened.

    Under the pending-token round structure each round makes `draft_forwards`
    draft calls (one per proposal; the first also consumes the not-yet-cached
    emitted tokens) and exactly one target call of `n_proposed + 1` tokens.
    Rollback is a pair of cache crops and does no model work.
    """
    proposed = sum(item.n_proposed for item in result.rounds)
    examined = sum(item.n_examined for item in result.rounds)
    accepted = sum(item.n_accepted for item in result.rounds)
    emitted = len(result.tokens)
    emitted_per_round = [item.n_emitted for item in result.rounds]
    draft_s = sum(item.draft_proposal_s for item in result.timings)
    verify_s = sum(item.target_verification_s for item in result.timings)
    decide_s = sum(item.acceptance_residual_s for item in result.timings)
    rollback_s = sum(item.cache_rollback_s for item in result.timings)
    accounted = result.prefill_s + draft_s + verify_s + decide_s + rollback_s
    host_s = max(0.0, elapsed_s - accounted)
    rounds = len(result.rounds)
    target_passes = sum(1 for item in result.timings if item.target_input_tokens > 0)
    decode_s = max(0.0, elapsed_s - result.prefill_s)
    return {
        "output_tokens": emitted,
        "accepted": accepted,
        "examined": examined,
        "proposed": proposed,
        "accepted_over_examined": None if examined == 0 else accepted / examined,
        "accepted_over_proposed": None if proposed == 0 else accepted / proposed,
        "rounds": rounds,
        "mean_accepted_per_round": None if rounds == 0 else accepted / rounds,
        "mean_emitted_per_round": None if rounds == 0 else float(np.mean(emitted_per_round)),
        "rounds_with_rejection": sum(1 for item in result.rounds if item.rejected),
        "rounds_with_bonus": sum(1 for item in result.rounds if item.bonus_token is not None),
        "rounds_bonus_suppressed": sum(1 for item in result.rounds if item.suppressed_bonus),
        "target_passes": target_passes,
        "target_passes_per_output_token": None if emitted == 0 else target_passes / emitted,
        "target_verify_input_tokens": sum(item.target_input_tokens for item in result.timings),
        "draft_proposal_forwards": sum(item.draft_forwards for item in result.timings),
        "draft_input_tokens": sum(item.draft_input_tokens for item in result.timings),
        "draft_forwards_per_output_token": None if emitted == 0 else sum(item.draft_forwards for item in result.timings) / emitted,
        "tokens_per_second": None if elapsed_s <= 0 else emitted / elapsed_s,
        "latency_s": elapsed_s,
        "elapsed_s": elapsed_s,
        "prefill_s": result.prefill_s,
        "decode_s": decode_s,
        "steady_decode_tokens_per_second": None if decode_s <= 0 else emitted / decode_s,
        "draft_proposal_s": draft_s,
        "target_verification_s": verify_s,
        "acceptance_residual_s": decide_s,
        "cache_rollback_s": rollback_s,
        "host_overhead_s": host_s,
        "hit_eos": result.hit_eos,
        "length_capped": result.length_capped,
        "peak_memory_bytes": peak_bytes,
    }


def timed_target_only(model, prompt_ids: Sequence[int], *, max_new_tokens: int, eos_id: int | None, rng, ignore_eos: bool) -> dict:
    """Target-only sampling with the same cache convention as the speculative decoder.

    The prompt minus its last token is prefilled; every decode step feeds the
    last committed token and samples the next one, so N tokens cost N forwards.
    """
    stop_id = None if ignore_eos else eos_id
    state = KVCacheState(model)
    ids = [int(token) for token in prompt_ids]
    cuda_sync()
    start = time.perf_counter()
    with torch.inference_mode():
        if len(ids) > 1:
            state.forward_tokens(ids[:-1])
        cuda_sync()
        prefill_s = time.perf_counter() - start
        decode_start = time.perf_counter()
        tokens: list[int] = []
        hit_eos = False
        pending = ids[-1]
        for _ in range(max_new_tokens):
            state.forward_tokens([pending])
            token = sample_categorical(logits_to_probs(state.next_logits), rng)
            tokens.append(token)
            pending = token
            if stop_id is not None and token == stop_id:
                hit_eos = True
                break
        cuda_sync()
        decode_s = time.perf_counter() - decode_start
    elapsed = prefill_s + decode_s
    return {
        "tokens": tokens,
        "output_tokens": len(tokens),
        "hit_eos": hit_eos,
        "length_capped": len(tokens) >= max_new_tokens and not hit_eos,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "elapsed_s": elapsed,
        "latency_s": elapsed,
        "tokens_per_second": None if elapsed <= 0 else len(tokens) / elapsed,
        "steady_decode_tokens_per_second": None if decode_s <= 0 else len(tokens) / decode_s,
        "target_passes": len(tokens),
        "target_passes_per_output_token": 1.0 if tokens else None,
    }


def _memory_marks() -> dict:
    if not torch.cuda.is_available():
        return {"resident_memory_bytes": 0}
    resident = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    return {"resident_memory_bytes": resident}


def _peak_bytes() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def timed_speculative(
    decoder: SpeculativeDecoder,
    prompt_ids: Sequence[int],
    *,
    gamma: int,
    max_new_tokens: int,
    eos_id: int | None,
    seed: int,
    ignore_eos: bool = False,
) -> dict:
    """One synchronized end-to-end speculative generation (prefill included)."""
    marks = _memory_marks()
    decoder.rng = np.random.default_rng(seed)
    cuda_sync()
    start = time.perf_counter()
    with torch.inference_mode():
        decoder.prefill(prompt_ids)
        result = decoder.generate(gamma=gamma, max_new_tokens=max_new_tokens, eos_id=eos_id, ignore_eos=ignore_eos)
    cuda_sync()
    elapsed = time.perf_counter() - start
    summary = summarize_speculative(result, elapsed, _peak_bytes())
    summary.update(marks)
    summary["tokens"] = list(result.tokens)
    return summary


def benchmark_prompt(
    decoder: SpeculativeDecoder,
    target_model,
    prompt_ids: Sequence[int],
    *,
    gamma: int,
    max_new_tokens: int,
    eos_id: int | None,
    seed: int,
    repetition: int,
    ignore_eos: bool = False,
) -> dict:
    """One paired repetition. Order alternates so the target-only run stays adjacent.

    Peak memory is reset before each run; `resident_memory_bytes` (weights and
    anything else already allocated) is recorded so the per-run increment is
    peak minus resident.
    """
    rows = {}
    for kind in repetition_order(repetition):
        if kind == "target_only":
            marks = _memory_marks()
            rows[kind] = timed_target_only(
                target_model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_id=eos_id,
                rng=np.random.default_rng(seed + 3),
                ignore_eos=ignore_eos,
            )
            rows[kind].update(marks)
            rows[kind]["peak_memory_bytes"] = _peak_bytes()
        else:
            rows[kind] = timed_speculative(
                decoder, prompt_ids, gamma=gamma, max_new_tokens=max_new_tokens, eos_id=eos_id, seed=seed, ignore_eos=ignore_eos
            )
    rows["order"] = list(repetition_order(repetition))
    rows["gamma"] = gamma
    rows["seed"] = seed
    rows["repetition"] = repetition
    return rows


def warm_setup(decoder: SpeculativeDecoder, target_model, prompt_ids: Sequence[int], *, gamma: int, eos_id: int | None, max_new_tokens: int = 32) -> dict:
    """First and second end-to-end call of each path on one prompt, before any timed row.

    The first call carries one-time costs (kernel selection, allocator growth,
    lazy CUDA init); the second is a warm reference. Neither enters the paired rows.
    """
    out = {"prompt_tokens": len(prompt_ids), "max_new_tokens": max_new_tokens, "gamma": gamma}
    for label in ("first", "second"):
        spec = timed_speculative(decoder, prompt_ids, gamma=gamma, max_new_tokens=max_new_tokens, eos_id=eos_id, seed=99)
        target = timed_target_only(target_model, prompt_ids, max_new_tokens=max_new_tokens, eos_id=eos_id, rng=np.random.default_rng(99), ignore_eos=False)
        out[label] = {
            "speculative_elapsed_s": spec["elapsed_s"],
            "speculative_prefill_s": spec["prefill_s"],
            "target_only_elapsed_s": target["elapsed_s"],
            "target_only_prefill_s": target["prefill_s"],
        }
    return out


ISOLATED_PROMPT_LENGTHS = (32, 128, 512)


def estimate_speed_model(draft_model, target_model, gamma: int, prompt_lengths: Sequence[int] = ISOLATED_PROMPT_LENGTHS) -> dict:
    """Isolated draft step, target step and verification cost at several cache lengths.

    A verification forward carries gamma + 1 tokens (the pending emitted token
    plus gamma proposals), which is what one round of the decoder runs, so
    v_gamma uses that width. The gamma-token width is recorded alongside.
    """
    draft_steps = [isolated_forward_seconds(draft_model, length, 1) for length in prompt_lengths]
    target_steps = [isolated_forward_seconds(target_model, length, 1) for length in prompt_lengths]
    verifies = [isolated_forward_seconds(target_model, length, gamma + 1) for length in prompt_lengths]
    verifies_gamma = [isolated_forward_seconds(target_model, length, gamma) for length in prompt_lengths]
    ratios = cost_ratios(float(np.mean(draft_steps)), float(np.mean(target_steps)), float(np.mean(verifies)))
    per_length = {
        str(length): cost_ratios(draft, target, verify)
        for length, draft, target, verify in zip(prompt_lengths, draft_steps, target_steps, verifies)
    }
    return {
        "gamma": gamma,
        "prompt_lengths": list(prompt_lengths),
        "draft_step_s": draft_steps,
        "target_step_s": target_steps,
        "verify_s": verifies,
        "verify_tokens": gamma + 1,
        "verify_gamma_tokens_s": verifies_gamma,
        "v_gamma_tokens_only": float(np.mean(verifies_gamma)) / float(np.mean(target_steps)),
        "per_prompt_length": per_length,
        **ratios,
    }


def _prompt_seed(base_seed: int, prompt_index: int, repetition: int) -> int:
    return int(base_seed + 10007 * prompt_index + 17 * repetition)


def decode_text(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)


FP32_GATE = 1e-3


def _fp32_gate() -> float:
    """Hard FP32 tolerance. PA4_T3_FP32_GATE may loosen it; the value used is recorded."""
    import os

    return float(os.environ.get("PA4_T3_FP32_GATE", FP32_GATE))


def _smoke_checks(target, draft, tokenizer, token_ids: Sequence[int], probe: Sequence[int]) -> dict:
    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    return {
        "split_prefill_max_abs": split_prefill_max_abs(target, token_ids, split=17),
        "cache": cache_matches_full_forward(target, draft, probe, eos_id, max_new_tokens=32),
    }


def run_smoke() -> dict:
    """Token ids, split-prefill and cache-position consistency, in FP16 and FP32.

    FP16 numbers are reported only: SDPA picks different kernels for a split
    prefill, and FP16 logits of size ~20 round in steps of ~0.016, so gaps of a
    few tenths are expected. The hard gate is FP32 (TF32 off): split prefill vs
    one full forward, the cached next logits after speculative rounds with
    rollbacks, and every cached key/value against a one-pass cache must agree
    within PA4_T3_FP32_GATE (default 1e-3). Refuses without CUDA before any download.
    """
    import gc

    from task3.src.distill import _require_cuda, load_frozen_causal

    _require_cuda("smoke")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target, target_tokenizer = load_frozen_causal(TARGET_MODEL)
    draft, draft_tokenizer = load_frozen_causal(DRAFT_MODEL)
    assert_same_token_ids(target_tokenizer, draft_tokenizer)
    probe = render_chat(target_tokenizer, "What is 2 + 2? Reply with the integer.")
    long_probe = render_chat(target_tokenizer, "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
    token_ids = (long_probe * 2)[:64]
    fp16 = _smoke_checks(target, draft, target_tokenizer, token_ids, probe)
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()
    target, _ = load_frozen_causal(TARGET_MODEL, dtype=torch.float32)
    draft, _ = load_frozen_causal(DRAFT_MODEL, dtype=torch.float32)
    fp32 = _smoke_checks(target, draft, target_tokenizer, token_ids, probe)
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()
    gate = _fp32_gate()
    fp32_errors = {
        "split_prefill_max_abs": fp32["split_prefill_max_abs"],
        "next_logit_max_abs": fp32["cache"]["next_logit_max_abs"],
        "target_kv_max_abs": fp32["cache"]["target_kv_max_abs"],
        "draft_kv_max_abs": fp32["cache"]["draft_kv_max_abs"],
    }
    failures = [name for name, value in fp32_errors.items() if not value < gate]
    if not fp16["cache"]["length_ok"]:
        failures.append("fp16_cache_length")
    if not fp32["cache"]["length_ok"]:
        failures.append("fp32_cache_length")
    report = {
        "fp16_diagnostic": fp16,
        "fp32": fp32,
        "fp32_gate": gate,
        "fp32_gate_failures": failures,
        "passed": not failures,
        # Kept for older readers of smoke.json.
        "split_prefill_max_abs": fp16["split_prefill_max_abs"],
        "cache": fp16["cache"],
        "token_id_prompts": list(TOKEN_CHECK_PROMPTS),
        "target": TARGET_MODEL,
        "draft": DRAFT_MODEL,
        "revisions": {name: runtime.pinned_revision(name) for name in (TARGET_MODEL, DRAFT_MODEL)},
    }
    _dump(results_dir() / "smoke.json", report)
    if failures:
        raise RuntimeError(f"smoke check failed ({failures}): {report}")
    return report


def _dump(path: Path, payload: dict) -> None:
    """Atomic, numpy-safe JSON write; a smoke run adds a SMOKE_RUN marker."""
    runtime.dump_json(path, payload)


_json_default = runtime.json_default


def load_result(name: str) -> dict:
    path = results_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}. Run the GPU stage that writes it before analyze.")
    return json.loads(path.read_text(encoding="utf-8"))
