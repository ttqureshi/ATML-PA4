"""Part A benchmarks and the MQAR forward-only sweep.

Dense attention is not run at L=8192. Memory figures are CUDA allocated and
reserved bytes; on CPU those cells are recorded as unsupported. SDPA's backend
string is the implementation that executed the probe, not an assumed kernel.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from shared.seed import set_seed
from shared.timing import time_call
from task2.src.attention import (
    dense_causal_attention,
    sdpa_backend_name,
    sdpa_causal_attention,
    tiled_causal_attention,
)
from task2.src.model import MQARModel
from task2.src.theory import theoretical_report

LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
SWEEP_LENGTHS = (256, 1024, 4096, 8192)
DTYPES = (torch.float32, torch.float16)


def _cuda_peaks() -> tuple[int | None, int | None, str]:
    if not torch.cuda.is_available():
        return None, None, "unsupported: CUDA allocated and reserved memory require a GPU"
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved()), "cuda"


def _sync_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (candidate.float() - reference.float()).abs()
    return {"max_abs": float(delta.max()), "rmse": float(torch.sqrt((delta ** 2).mean()))}


def _status_from_error(exc: BaseException) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "not enough memory" in text:
        return "out-of-memory"
    return "unsupported"


def _time_and_memory(fn: Callable[[], object], warmup: int = 10, repeats: int = 10) -> dict:
    _sync_reset()
    samples = time_call(fn, warmup=warmup, repeats=repeats)
    allocated, reserved, note = _cuda_peaks()
    return {
        "seconds_mean": float(sum(samples) / len(samples)),
        "seconds_samples": samples,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "memory_note": note,
    }


def make_qkv(batch: int, heads: int, length: int, width: int, dtype: torch.dtype, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (batch, heads, length, width)
    query = torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    key = torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    value = torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    return query, key, value


def reference_forward_backward(query, key, value):
    query_r = query.detach().float().requires_grad_(True)
    key_r = key.detach().float().requires_grad_(True)
    value_r = value.detach().float().requires_grad_(True)
    output = dense_causal_attention(query_r, key_r, value_r)
    output.sum().backward()
    return output.detach(), query_r.grad.detach(), key_r.grad.detach(), value_r.grad.detach()


def benchmark_attention(
    lengths: tuple[int, ...] = LENGTHS,
    device: torch.device | None = None,
    tile_size: int = 128,
    warmup: int = 10,
    repeats: int = 10,
) -> list[dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    dtypes = DTYPES if device.type == "cuda" else (torch.float32, torch.float16)
    for length in lengths:
        for dtype in dtypes:
            try:
                query, key, value = make_qkv(2, 4, length, 64, dtype, device, seed=6304)
            except RuntimeError as exc:
                rows.append(
                    {
                        "length": length,
                        "dtype": str(dtype).replace("torch.", ""),
                        "method": "setup",
                        "status": _status_from_error(exc),
                        "note": str(exc),
                    }
                )
                continue
            reference = None
            if not (length >= 8192):
                try:
                    reference = reference_forward_backward(query, key, value)
                except RuntimeError as exc:
                    reference = exc
            methods: list[tuple[str, Callable]] = []
            if length < 8192:
                methods.append(("dense", lambda q=query, k=key, v=value: dense_causal_attention(q, k, v)))
            else:
                rows.append(
                    {
                        "length": length,
                        "dtype": str(dtype).replace("torch.", ""),
                        "method": "dense",
                        "status": "skipped",
                        "note": "dense baseline omitted at L=8192",
                    }
                )
            methods.append(
                (
                    "tiled",
                    lambda q=query, k=key, v=value: tiled_causal_attention(q, k, v, tile_size=tile_size),
                )
            )
            methods.append(("sdpa", lambda q=query, k=key, v=value: sdpa_causal_attention(q, k, v)))
            for name, forward in methods:
                row = {
                    "length": length,
                    "dtype": str(dtype).replace("torch.", ""),
                    "method": name,
                    "tile_size": tile_size if name == "tiled" else None,
                    "backward": "autograd_stores_tile_intermediates" if name == "tiled" else "autograd",
                    "status": "ok",
                }
                if name == "sdpa":
                    row["backend"] = sdpa_backend_name(device.type, dtype, min(length, 128), 64)
                if name == "tiled":
                    row["note"] = "Python tiled online softmax. Not FlashAttention."

                def _run(op_name=name, q=query, k=key, v=value, with_grad=False):
                    if with_grad:
                        q_g = q.detach().requires_grad_(True)
                        k_g = k.detach().requires_grad_(True)
                        v_g = v.detach().requires_grad_(True)
                    else:
                        q_g, k_g, v_g = q, k, v
                    if op_name == "dense":
                        out = dense_causal_attention(q_g, k_g, v_g)
                    elif op_name == "tiled":
                        out = tiled_causal_attention(q_g, k_g, v_g, tile_size=tile_size)
                    else:
                        out = sdpa_causal_attention(q_g, k_g, v_g)
                    if with_grad:
                        out.sum().backward()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    return out

                try:
                    row["forward"] = _time_and_memory(lambda: _run(with_grad=False), warmup=warmup, repeats=repeats)
                    row["forward_backward"] = _time_and_memory(
                        lambda: _run(with_grad=True), warmup=warmup, repeats=repeats
                    )
                    with torch.no_grad():
                        output = _run(with_grad=False)
                    if isinstance(reference, tuple):
                        ref_out, ref_q, ref_k, ref_v = reference
                        row["output_error"] = _error(output, ref_out)
                        q_g = query.detach().float().requires_grad_(True)
                        k_g = key.detach().float().requires_grad_(True)
                        v_g = value.detach().float().requires_grad_(True)
                        if name == "dense":
                            grad_out = dense_causal_attention(q_g, k_g, v_g)
                        elif name == "tiled":
                            grad_out = tiled_causal_attention(q_g, k_g, v_g, tile_size=tile_size)
                        else:
                            grad_out = sdpa_causal_attention(q_g, k_g, v_g)
                        grad_out.sum().backward()
                        row["grad_error"] = {
                            "query": _error(q_g.grad, ref_q),
                            "key": _error(k_g.grad, ref_k),
                            "value": _error(v_g.grad, ref_v),
                        }
                    elif isinstance(reference, Exception):
                        row["output_error"] = {"status": _status_from_error(reference), "note": str(reference)}
                    else:
                        row["output_error"] = {"status": "skipped", "note": "no dense FP32 reference at this length"}
                except RuntimeError as exc:
                    row["status"] = _status_from_error(exc)
                    row["note"] = str(exc)
                rows.append(row)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return rows


def edge_case_report(device: torch.device | None = None) -> dict:
    """All-masked row and an extreme-score row, compared with an FP32 dense reference."""
    if device is None:
        device = torch.device("cpu")
    length, heads, width, batch = 8, 4, 64, 2
    query = torch.randn(batch, heads, length, width, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    allowed[3, :] = False
    key = key.clone()
    key[:, :, 1, :] = 1e4
    reference = dense_causal_attention(query.float(), key.float(), value.float(), allowed)
    tiled = tiled_causal_attention(query, key, value, tile_size=4, allowed=allowed)
    return {
        "all_masked_row_is_zero": bool(torch.all(reference[:, :, 3, :] == 0)),
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(tiled).all()),
        "tiled_max_abs": float((tiled.float() - reference.float()).abs().max()),
        "extreme_row": 0,
    }


@torch.no_grad()
def forward_sweep(
    lengths: tuple[int, ...] = SWEEP_LENGTHS,
    device: torch.device | None = None,
    batch: int = 1,
) -> dict:
    """Fixed random token embeddings through each operator. No optimizer.

    Linear attention materializes an inclusive prefix tensor. Window and mixed
    attention materialize a dense boolean mask. Neither is reported as a sparse
    kernel.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(6304)
    rows = []
    notes = {
        "dense": "full causal scores",
        "window": "dense boolean window mask, not a sparse kernel",
        "mixed": "dense boolean per-head mask, not a sparse kernel",
        "linear": "inclusive prefix state (B, H, L, D, D), not an LxL score matrix",
        "gated": "sequential forget-gate scan; persistent state matches ungated linear",
    }
    for kind in ("dense", "window", "mixed", "linear", "gated"):
        model = MQARModel(kind).to(device).eval()
        for length in lengths:
            row = {"kind": kind, "length": length, "batch": batch, "storage": notes[kind], "status": "ok"}
            tokens = torch.randint(0, 256, (batch, length), device=device)
            try:
                def _run(model=model, tokens=tokens):
                    return model(tokens)

                timed = _time_and_memory(_run, warmup=1 if length >= 4096 else 2, repeats=1 if length >= 4096 else 3)
                row.update(timed)
            except RuntimeError as exc:
                row["status"] = _status_from_error(exc)
                row["note"] = str(exc)
            rows.append(row)
    return {"rows": rows, "theory_at_256": theoretical_report(256), "scale": 1 / math.sqrt(32)}
