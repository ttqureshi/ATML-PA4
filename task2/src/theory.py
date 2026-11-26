"""Theoretical pair counts and recurrent-state bytes for the MQAR models.

Sequence length 256 is the training length. Far queries are outside a two-layer
window-64 stack because each layer extends the receptive field by ``W - 1``.
"""

from __future__ import annotations

from task2.src.masks import (
    DILATION_OFFSETS,
    WINDOW,
    dilation_allowed_count,
    window_allowed_count,
)

N_LAYERS = 2
N_HEADS = 4
HEAD_DIM = 32
LOCAL_HEADS = 2
DILATED_HEADS = 2
FP32_BYTES = 4

RECEPTIVE_FIELD_NOTE = (
    "In a 2-layer window-64 model the receptive field is 2*(64-1)=126, "
    "so far queries are unreachable by construction."
)


def receptive_field(layers: int = N_LAYERS, window: int = WINDOW) -> int:
    return layers * (window - 1)


def dense_pairs(length: int, heads: int = N_HEADS) -> int:
    return heads * length * (length + 1) // 2


def window_pairs(length: int, window: int = WINDOW, heads: int = N_HEADS) -> int:
    return heads * sum(window_allowed_count(i, window) for i in range(length))


def dilated_pairs(length: int, heads: int = DILATED_HEADS) -> int:
    return heads * sum(dilation_allowed_count(i) for i in range(length))


def mixed_pairs(length: int, window: int = WINDOW) -> int:
    local = LOCAL_HEADS * sum(window_allowed_count(i, window) for i in range(length))
    return local + dilated_pairs(length, DILATED_HEADS)


def dense_mask_bytes(length: int, heads: int, dtype_bytes: int = 1) -> int:
    """One dense boolean keep-mask. Torch stores bool as one byte."""
    return heads * length * length * dtype_bytes


def linear_state_bytes(
    layers: int = N_LAYERS,
    heads: int = N_HEADS,
    width: int = HEAD_DIM,
    dtype_bytes: int = FP32_BYTES,
) -> int:
    """Bytes of ``S`` (``D×D``) plus ``z`` (``D``) per sequence, all layers.

    The forget gate does not change this count: it is a scalar computed from
    the current query and is not appended to the recurrent state.
    """
    elements = layers * heads * (width * width + width)
    return elements * dtype_bytes


def kv_cache_bytes(
    length: int,
    layers: int = N_LAYERS,
    heads: int = N_HEADS,
    width: int = HEAD_DIM,
    dtype_bytes: int = FP32_BYTES,
) -> int:
    """Full key and value cache for one sequence, four stored heads (not GQA)."""
    return 2 * layers * heads * width * length * dtype_bytes


def linear_prefix_bytes(
    length: int,
    batch: int = 1,
    heads: int = N_HEADS,
    width: int = HEAD_DIM,
    dtype_bytes: int = FP32_BYTES,
) -> int:
    """Inclusive prefix tensor ``(B, H, L, D, D)`` used by ungated linear attention."""
    return batch * heads * length * width * width * dtype_bytes


def theoretical_report(length: int = 256) -> dict[str, int | str]:
    return {
        "length": length,
        "dense_pairs_per_layer": dense_pairs(length),
        "window_pairs_per_layer": window_pairs(length),
        "mixed_pairs_per_layer": mixed_pairs(length),
        "dilated_offsets": len(DILATION_OFFSETS),
        "window_mask_bytes": dense_mask_bytes(length, heads=1),
        "mixed_mask_bytes": dense_mask_bytes(length, heads=N_HEADS),
        "mask_storage": "dense_bool",
        "linear_state_bytes": linear_state_bytes(),
        "gated_state_bytes": linear_state_bytes(),
        "dense_kv_cache_fp32_bytes": kv_cache_bytes(length),
        "linear_prefix_bytes_batch1": linear_prefix_bytes(length),
        "receptive_field": receptive_field(),
        "receptive_field_note": RECEPTIVE_FIELD_NOTE,
    }
