"""Serialized KV byte budgets. Dimensions come from the model config.

``M_b`` is the uniform grouped cache: each group of 64 has an FP16 scale and
an FP16 offset (4 metadata bytes) plus a packed payload, and the trailing
``L mod 64`` elements of every channel stay FP16. ``M_16`` is the FP16 payload
of both K and V with no metadata.
"""

from __future__ import annotations

from typing import Any

GROUP = 64
SINK_TOKENS = 8
RECENT_TOKENS = 64
INDEX_BYTES = 4


def grouped_bytes_per_channel(length: int, bits: int) -> int:
    if length < 0:
        raise ValueError("length must be non-negative")
    if bits not in (2, 4, 16):
        raise ValueError("bits must be 2, 4, or 16")
    groups = length // GROUP
    remainder = length % GROUP
    return groups * ((GROUP * bits) // 8 + 4) + 2 * remainder


def memory_budget(length: int, n_layers: int, h_kv: int, d_h: int, bits: int = 4) -> int:
    """``M_b(L)`` for ``bits=4``. The same expression is used for INT2 groups."""
    per_channel = grouped_bytes_per_channel(length, bits)
    return 2 * n_layers * h_kv * d_h * per_channel


def memory_fp16(length: int, n_layers: int, h_kv: int, d_h: int) -> int:
    """``M_16(L) = 4 * n_layers * H_kv * d_h * L``."""
    return 4 * n_layers * h_kv * d_h * length


def kv_dims_from_config(config: Any) -> tuple[int, int, int]:
    """Read ``n_layers``, ``H_kv``, and ``d_h`` and check them against the config."""
    n_layers = int(config.num_hidden_layers)
    h_kv = int(config.num_key_value_heads)
    n_q = int(config.num_attention_heads)
    hidden = int(config.hidden_size)
    if n_q <= 0 or hidden % n_q != 0:
        raise AssertionError("hidden size must be divisible by the query-head count")
    derived = hidden // n_q
    configured = getattr(config, "head_dim", None)
    if configured is None:
        d_h = derived
    else:
        d_h = int(configured)
        if d_h != derived:
            raise AssertionError(f"config.head_dim {d_h} != hidden/heads {derived}")
    if n_layers <= 0 or h_kv <= 0 or d_h <= 0:
        raise AssertionError("KV dimensions must be positive")
    if n_q % h_kv != 0:
        raise AssertionError("query heads must be divisible by KV heads")
    return n_layers, h_kv, d_h


def eviction_bytes(n_positions: int, n_layers: int, h_kv: int, d_h: int, index_bytes: int = INDEX_BYTES) -> int:
    """FP16 K and V for the retained positions, plus one int32 index per position per layer."""
    payload = n_layers * 2 * h_kv * d_h * n_positions * 2
    indices = n_layers * n_positions * index_bytes
    return payload + indices


def largest_fp16_count(
    budget: int,
    length: int,
    n_layers: int,
    h_kv: int,
    d_h: int,
) -> int:
    """Largest position count whose eviction bytes fit in ``budget``."""
    if budget < 0 or length < 0:
        raise ValueError("budget and length must be non-negative")
    low, high = 0, length
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if eviction_bytes(mid, n_layers, h_kv, d_h) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def hybrid_bytes(
    fp16_tokens: int,
    quantized_tokens: int,
    n_layers: int,
    h_kv: int,
    d_h: int,
    bits: int = 2,
) -> int:
    """FP16 payload, INT2 grouped payload, and int32 indices for both sets.

    ``quantized_tokens`` must be a multiple of 64 so there is no partial group.
    Index bytes cover every stored position, once per layer.
    """
    if quantized_tokens % GROUP != 0:
        raise ValueError("INT2 positions must form complete groups of 64")
    fp16_payload = memory_fp16(fp16_tokens, n_layers, h_kv, d_h)
    quantized_payload = memory_budget(quantized_tokens, n_layers, h_kv, d_h, bits) if quantized_tokens else 0
    indices = n_layers * (fp16_tokens + quantized_tokens) * INDEX_BYTES
    return fp16_payload + quantized_payload + indices
