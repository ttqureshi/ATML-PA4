"""Packed-byte accounts for dense FP16, unstructured sparsity, and quantization.

Tied parameters are counted once. The caller passes unique element counts.
Eligible weights are the seven projections; everything else stays FP16.

The unstructured estimate is one mask bit per eligible weight, 16 bits per
nonzero eligible weight, and enough padding to finish the last byte, plus the
FP16 bytes of the excluded parameters. It is not a CSR layout and it is not
the dense array with explicit zeros.
"""

from __future__ import annotations

import math

from task1.src.constants import FP16_BYTES, GROUP_SIZE
from task1.src.quantize import tensor_packed_bytes


def dense_fp16_bytes(n_unique_params: int) -> int:
    if n_unique_params < 0:
        raise ValueError("parameter count cannot be negative")
    return int(n_unique_params) * FP16_BYTES


def sparse_storage_bytes(n_eligible: int, n_nonzero: int, n_excluded: int) -> dict[str, int]:
    if n_eligible < 0 or n_nonzero < 0 or n_excluded < 0:
        raise ValueError("counts cannot be negative")
    if n_nonzero > n_eligible:
        raise ValueError("nonzero count cannot exceed the eligible count")
    mask_bits = int(n_eligible)
    value_bits = 16 * int(n_nonzero)
    raw_bits = mask_bits + value_bits
    payload = (raw_bits + 7) // 8
    excluded = dense_fp16_bytes(n_excluded)
    return {
        "mask_bits": mask_bits,
        "nonzero_value_bits": value_bits,
        "padding_bits": payload * 8 - raw_bits,
        "eligible_sparse_bytes": payload,
        "excluded_fp16_bytes": excluded,
        "unstructured_sparse_bytes": payload + excluded,
        "dense_fp16_bytes": dense_fp16_bytes(n_eligible + n_excluded),
    }


def quantized_model_bytes(
    projections: list[dict[str, int]],
    n_excluded: int,
    group_size: int = GROUP_SIZE,
) -> dict[str, int]:
    """Sum payload, group metadata, serialized scales, and excluded FP16.

    Each projection dict has ``rows``, ``cols``, ``bits``, and ``scale_bytes``.
    ``scale_bytes`` is 0 when the inverse scale was folded into a preceding
    parameter that is already inside ``n_excluded``.
    """
    payload = 0
    metadata = 0
    scales = 0
    eligible = 0
    for item in projections:
        rows = int(item["rows"])
        cols = int(item["cols"])
        bits = int(item["bits"])
        scale_bytes = int(item["scale_bytes"])
        packed = tensor_packed_bytes(rows, cols, bits, group_size)
        n_groups = rows * math.ceil(cols / group_size)
        group_meta = n_groups * 4
        payload += packed - group_meta
        metadata += group_meta
        scales += scale_bytes
        eligible += rows * cols
    excluded = dense_fp16_bytes(n_excluded)
    total = payload + metadata + scales + excluded
    return {
        "payload_bytes": payload,
        "metadata_bytes": metadata,
        "serialized_scale_bytes": scales,
        "excluded_fp16_bytes": excluded,
        "packed_bytes": total,
        "eligible_params": eligible,
    }
