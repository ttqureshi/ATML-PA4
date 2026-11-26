"""Numpy compression of one eligible matrix. Torch IO lives in model_ops."""

from __future__ import annotations

import math

import numpy as np

from task1.src.constants import FP16_BYTES, GROUP_SIZE
from task1.src.pruning import (
    keep_mask,
    magnitude_scores,
    score_summary,
    two_four_keep_mask,
    uniform_keep_mask,
    wanda_scores,
    zeros_per_row_from_total,
    zeros_per_row_uniform,
)
from task1.src.quantize import awq_reconstruct, quantize_weight, tensor_packed_bytes


def fp16_round(values: np.ndarray) -> np.ndarray:
    """Values as they will sit in the FP16 module used for numerical evaluation."""
    return np.asarray(values, dtype=np.float16).astype(np.float64)


def prune_matrix(weight: np.ndarray, column_norm: np.ndarray | None, method: str, sparsity: float) -> dict:
    matrix = np.asarray(weight, dtype=np.float64)
    if method == "magnitude":
        scores = magnitude_scores(matrix)
    elif method in {"wanda", "wanda_2to4"}:
        if column_norm is None:
            raise ValueError("Wanda needs per-channel input norms")
        scores = wanda_scores(matrix, column_norm)
    else:
        raise ValueError(f"unknown prune method {method}")
    if method == "wanda_2to4":
        mask = two_four_keep_mask(scores)
        zeros = int((~mask).sum())
    else:
        per_row = zeros_per_row_uniform(matrix.shape[1], sparsity)
        mask = uniform_keep_mask(scores, sparsity)
        zeros = int(matrix.shape[0] * per_row)
    pruned = fp16_round(matrix * mask)
    return {
        "values": pruned,
        "scores": scores,
        "mask": mask,
        "zeros": zeros,
        "nonzero": int(mask.sum()),
        "score_summary": score_summary(scores),
    }


def p3_matrix(weight: np.ndarray, column_norm: np.ndarray, n_zeros: int) -> dict:
    matrix = np.asarray(weight, dtype=np.float64)
    scores = wanda_scores(matrix, column_norm)
    counts = zeros_per_row_from_total(matrix.shape[0], int(n_zeros), matrix.shape[1])
    mask = keep_mask(scores, counts)
    if int((~mask).sum()) != int(n_zeros):
        raise RuntimeError("per-row Wanda mask did not hit the repaired zero count")
    return {
        "values": fp16_round(matrix * mask),
        "scores": scores,
        "mask": mask,
        "zeros": int(n_zeros),
        "nonzero": int(mask.sum()),
        "score_summary": score_summary(scores),
        "zeros_per_row_min": int(counts.min()),
        "zeros_per_row_max": int(counts.max()),
    }


def quantize_uniform(weight: np.ndarray, bits: int, group_size: int = GROUP_SIZE) -> dict:
    quantized = quantize_weight(weight, bits, group_size)
    values = fp16_round(quantized.values)
    rows, cols = values.shape
    return {
        "values_original_space": values,
        "values_scaled_space": values,
        "scales": None,
        "bits": int(bits),
        "pad_count": int(quantized.pad_counts.sum()),
        "n_groups": int(quantized.n_groups),
        "packed_tensor_bytes": tensor_packed_bytes(rows, cols, bits, group_size),
        "scale_bytes": 0,
        "nonzero": int(np.count_nonzero(values)),
    }


def quantize_awq(weight: np.ndarray, mean_abs: np.ndarray, alpha: float, bits: int, group_size: int = GROUP_SIZE) -> dict:
    scales, original, scaled = awq_reconstruct(weight, mean_abs, alpha, bits, group_size)
    rows, cols = weight.shape
    return {
        "scales": scales,
        "values_original_space": fp16_round(original),
        "values_scaled_space": fp16_round(scaled),
        "bits": int(bits),
        "alpha": float(alpha),
        "packed_tensor_bytes": tensor_packed_bytes(rows, cols, bits, group_size),
        "nonzero": int(np.count_nonzero(fp16_round(original))),
    }


def projection_scale_bytes(n_in: int, mode: str) -> int:
    if mode == "fold":
        return 0
    if mode == "store":
        return int(n_in) * FP16_BYTES
    raise ValueError(mode)


def average_bits(records: list[dict]) -> float:
    numerator = sum(int(row["bits"]) * int(row["numel"]) for row in records)
    denominator = sum(int(row["numel"]) for row in records)
    if denominator == 0:
        raise ValueError("average bits needs eligible weights")
    return numerator / denominator


def group_count(rows: int, cols: int, group_size: int = GROUP_SIZE) -> int:
    return rows * math.ceil(cols / group_size)
