"""Exact causal attention: dense scores, tiled online softmax, and PyTorch SDPA.

The tiled forward does not allocate an L×L score matrix. Its backward pass is
ordinary autograd through the tile intermediates, so those tiles are stored for
the backward. This is not a fused FlashAttention kernel.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

NEG_FILL = -1e30


def stable_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Softmax that returns zeros on a fully masked row instead of NaN.

    ``scores`` may contain ``-inf``. The reduction is computed in FP32 and
    cast back to the input dtype.
    """
    scores_f = scores.float()
    finite = torch.isfinite(scores_f)
    filled = torch.where(finite, scores_f, torch.full_like(scores_f, NEG_FILL))
    maximum = filled.amax(dim=-1, keepdim=True)
    shifted = torch.where(finite, scores_f - maximum, torch.full_like(scores_f, NEG_FILL))
    exp_scores = torch.exp(shifted)
    exp_scores = torch.where(finite, exp_scores, torch.zeros_like(exp_scores))
    denom = exp_scores.sum(dim=-1, keepdim=True)
    weights = torch.where(denom > 0, exp_scores / denom.clamp_min(1e-20), torch.zeros_like(exp_scores))
    return weights.to(dtype=scores.dtype)


def _as_allowed(allowed: Optional[torch.Tensor], length: int, device: torch.device) -> torch.Tensor:
    if allowed is None:
        positions = torch.arange(length, device=device)
        return positions.view(1, -1) <= positions.view(-1, 1)
    return allowed.bool()


def dense_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit ``Q K^T / sqrt(d)``, boolean keep-mask, stable softmax, times ``V``.

    ``query``, ``key``, and ``value`` are ``(B, H, L, D)``. ``allowed`` uses
    SDPA's boolean convention: ``True`` means the query may attend to that key.
    A missing mask is causal, including the diagonal.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("dense attention expects matching Q, K, V shapes")
    length = query.shape[-2]
    width = query.shape[-1]
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(width)
    keep = _as_allowed(allowed, length, query.device)
    scores = scores.masked_fill(~keep, float("-inf"))
    weights = stable_softmax(scores)
    output = torch.matmul(weights.float(), value.float())
    return output.to(dtype=query.dtype)


def _online_update(
    running_max: torch.Tensor,
    normalizer: torch.Tensor,
    accumulator: torch.Tensor,
    scores: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One online-softmax tile. ``running_max`` uses ``-inf`` before any finite score."""
    scores_f = scores.float()
    finite = torch.isfinite(scores_f)
    filled = torch.where(finite, scores_f, torch.full_like(scores_f, NEG_FILL))
    tile_max = filled.amax(dim=-1, keepdim=True)
    new_max = torch.maximum(running_max, tile_max)
    previous = torch.where(
        torch.isfinite(running_max),
        torch.exp(running_max - new_max),
        torch.zeros_like(running_max),
    )
    probabilities = torch.exp(torch.where(finite, scores_f - new_max, torch.full_like(scores_f, NEG_FILL)))
    probabilities = torch.where(finite, probabilities, torch.zeros_like(probabilities))
    new_normalizer = previous * normalizer + probabilities.sum(dim=-1, keepdim=True)
    new_accumulator = previous * accumulator + torch.matmul(probabilities, values.float())
    return new_max, new_normalizer, new_accumulator


def tiled_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tile_size: int,
    allowed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blockwise causal attention. Only a query-tile by key-tile score block is stored.

    Backward, if requested, is PyTorch autograd and keeps those tile intermediates.
    Do not describe this routine as FlashAttention.
    """
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    batch, heads, length, width = query.shape
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("tiled attention expects matching Q, K, V shapes")
    keep = _as_allowed(allowed, length, query.device)
    scale = 1.0 / math.sqrt(width)
    output = query.new_empty(query.shape)
    query_positions = torch.arange(length, device=query.device)
    for query_start in range(0, length, tile_size):
        query_end = min(query_start + tile_size, length)
        query_tile = query[:, :, query_start:query_end, :]
        query_count = query_end - query_start
        running_max = torch.full(
            (batch, heads, query_count, 1),
            float("-inf"),
            device=query.device,
            dtype=torch.float32,
        )
        normalizer = torch.zeros((batch, heads, query_count, 1), device=query.device, dtype=torch.float32)
        accumulator = torch.zeros(
            (batch, heads, query_count, width), device=query.device, dtype=torch.float32
        )
        # Keys after the last query in this tile are strictly future for every row.
        for key_start in range(0, query_end, tile_size):
            key_end = min(key_start + tile_size, query_end)
            key_tile = key[:, :, key_start:key_end, :]
            value_tile = value[:, :, key_start:key_end, :]
            scores = torch.matmul(query_tile.float(), key_tile.float().transpose(-2, -1)) * scale
            key_positions = query_positions[key_start:key_end]
            causal = key_positions.view(1, -1) <= query_positions[query_start:query_end].view(-1, 1)
            block_keep = keep[..., query_start:query_end, key_start:key_end] & causal
            scores = scores.masked_fill(~block_keep, float("-inf"))
            running_max, normalizer, accumulator = _online_update(
                running_max, normalizer, accumulator, scores, value_tile
            )
        tile_out = torch.where(
            normalizer > 0,
            accumulator / normalizer.clamp_min(1e-20),
            torch.zeros_like(accumulator),
        )
        output[:, :, query_start:query_end, :] = tile_out.to(dtype=query.dtype)
    return output


def sdpa_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``scaled_dot_product_attention`` on the same layout as the dense reference."""
    if allowed is None:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)
    keep = allowed.bool()
    if keep.shape[-2:] != (query.shape[-2], key.shape[-2]):
        raise ValueError("SDPA keep-mask must end in (query_len, key_len)")
    return F.scaled_dot_product_attention(query, key, value, attn_mask=keep, is_causal=False)


def sdpa_backend_name(device_type: str, dtype: torch.dtype, length: int, width: int) -> str:
    """Name the backend that can execute this shape, forcing it when several exist.

    CPU SDPA is the math implementation. On CUDA the probe tries the dispatch
    priority and returns the first backend that accepts the shape. The caller
    records that string as the backend that ran for the timed call.
    """
    if device_type != "cuda":
        return "math"
    from torch.nn.attention import SDPBackend, sdpa_kernel

    priority = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    probe_q = torch.randn(1, 1, length, width, device="cuda", dtype=dtype)
    probe_k = torch.randn_like(probe_q)
    probe_v = torch.randn_like(probe_q)
    for backend in priority:
        try:
            with sdpa_kernel(backend):
                sdpa_causal_attention(probe_q, probe_k, probe_v)
            return backend.name.lower()
        except Exception:
            continue
    return "unsupported"


def future_perturbation_changes_only_future(
    attention_fn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """Return the absolute difference of outputs before ``position`` after a future edit."""
    baseline = attention_fn(query, key, value)
    edited_key = key.clone()
    edited_value = value.clone()
    edited_key[:, :, position + 1 :, :] = edited_key[:, :, position + 1 :, :] + 1
    edited_value[:, :, position + 1 :, :] = edited_value[:, :, position + 1 :, :] + 1
    edited = attention_fn(query, edited_key, edited_value)
    return (baseline[:, :, : position + 1, :] - edited[:, :, : position + 1, :]).abs()
