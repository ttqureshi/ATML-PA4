"""Latency and memory. Dense FP16 execution is timed separately from packed-byte estimates."""

from __future__ import annotations

import statistics

from shared.timing import time_call
from task1.src.constants import DECODE_PROMPT_TOKENS, DECODE_TOKENS, PREFILL_LENGTHS, TIMING_REPEATS, TIMING_WARMUP


def _peak_bytes(device_type: str):
    if device_type != "cuda":
        return None
    import torch

    return int(torch.cuda.max_memory_allocated())


def _reset_peak(device_type: str) -> None:
    if device_type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()


def measure_latencies(
    model,
    vocab_size: int,
    *,
    prefill_lengths: tuple[int, ...] = PREFILL_LENGTHS,
    decode_tokens: int = DECODE_TOKENS,
    prompt_tokens: int = DECODE_PROMPT_TOKENS,
    warmup: int = TIMING_WARMUP,
    repeats: int = TIMING_REPEATS,
) -> dict:
    """Time dense forwards. This does not execute a packed or sparse kernel.

    Prefill is one forward at the requested length. Decode is a short prompt
    prefill plus ``decode_tokens`` cached greedy steps, timed together and
    labeled with the prompt length so the prefill figures stay separate.
    """
    import torch

    device = next(model.parameters()).device
    prefill = {}
    for length in prefill_lengths:
        ids = torch.arange(length, device=device).remainder(vocab_size).view(1, length)

        def forward(ids=ids):
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)

        entry = _time_region(forward, device.type, warmup, repeats)
        entry["tokens"] = int(length)
        entry["execution"] = "dense_fp16"
        prefill[str(length)] = entry

    prompt = torch.arange(prompt_tokens, device=device).remainder(vocab_size).view(1, prompt_tokens)

    def decode():
        with torch.inference_mode():
            out = model(input_ids=prompt, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(decode_tokens):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:, :].argmax(dim=-1)

    decode_entry = _time_region(decode, device.type, warmup, repeats)
    decode_entry["new_tokens"] = int(decode_tokens)
    decode_entry["prompt_tokens_included"] = int(prompt_tokens)
    decode_entry["execution"] = "dense_fp16"
    return {
        "execution": "dense_fp16",
        "packed_or_sparse_kernel": False,
        "warmup": int(warmup),
        "repeats": int(repeats),
        "prefill": prefill,
        "decode_64": decode_entry,
    }


def _time_region(fn, device_type: str, warmup: int, repeats: int) -> dict:
    import torch

    _reset_peak(device_type)
    try:
        samples = time_call(fn, warmup=warmup, repeats=repeats)
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        if device_type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "out_of_memory",
            "seconds": [],
            "median_seconds": None,
            "peak_allocated_bytes": _peak_bytes(device_type),
        }
    return {
        "status": status,
        "seconds": [float(value) for value in samples],
        "median_seconds": float(statistics.median(samples)),
        "peak_allocated_bytes": _peak_bytes(device_type),
    }
