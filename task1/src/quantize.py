"""Group-wise asymmetric weight quantization and the 3-point AWQ scale.

Rounding is ``numpy.rint`` (half to even), which is the ``round`` in the brief
for values that are not exactly halfway; halfway cases follow half-to-even.

A trailing partial group repeats its last real weight only while the min/max
are computed. Those repeated values are not quantized and are counted in
``pad_counts``. A degenerate group (``w_max == w_min``) uses scale 1 so the
division in the formula is defined. An all-zero group then reconstructs as 0.

Channel scales follow

    m_j = mean_t |X_{t,j}|
    c_j(alpha) = m_j^alpha / mean_j(m_j^alpha)

with ``m_j^0 = 1``, including where ``m_j`` is 0, so alpha 0 is the all-ones
scale and reproduces uniform quantization. Non-positive scales are replaced
by 1 before they multiply a weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from task1.src.constants import AWQ_ALPHAS, GROUP_METADATA_BYTES, GROUP_SIZE


@dataclass(frozen=True)
class QuantizedTensor:
    values: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    pad_counts: np.ndarray
    codes: np.ndarray
    bits: int
    group_size: int

    @property
    def n_groups(self) -> int:
        return int(self.scales.shape[0])


def tensor_packed_bytes(n_rows: int, n_cols: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    """Payload plus per-group metadata for one eligible matrix.

    The payload is little-endian bit packing of the real weights only, padded
    to a byte at the end of the tensor. Each group contributes 4 metadata
    bytes: an FP16 scale, a uint8 zero-point, and one alignment byte.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("tensor shape must be positive")
    if bits < 1 or bits > 8:
        raise ValueError("bits must lie in 1..8 so the zero-point fits in one byte")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    n_weights = n_rows * n_cols
    n_groups = n_rows * math.ceil(n_cols / group_size)
    payload = (n_weights * bits + 7) // 8
    return payload + n_groups * GROUP_METADATA_BYTES


def pack_codes_little_endian(codes: np.ndarray, bits: int) -> bytes:
    """Pack integer codes into little-endian bits, first code in the low bits."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must lie in 1..8")
    flat = np.asarray(codes, dtype=np.int64).reshape(-1)
    mask = (1 << bits) - 1
    if flat.size and (int(flat.min()) < 0 or int(flat.max()) > mask):
        raise ValueError("a code does not fit in the bit width")
    total_bits = int(flat.size) * bits
    out = bytearray((total_bits + 7) // 8)
    position = 0
    for code in flat.tolist():
        remaining = bits
        value = int(code)
        while remaining:
            byte_index = position // 8
            bit_in_byte = position % 8
            take = min(8 - bit_in_byte, remaining)
            chunk = value & ((1 << take) - 1)
            out[byte_index] |= chunk << bit_in_byte
            value >>= take
            remaining -= take
            position += take
    return bytes(out)


def quantize_weight(weight: np.ndarray, bits: int, group_size: int = GROUP_SIZE) -> QuantizedTensor:
    """Asymmetric group quantizer along input columns."""
    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("weight must have shape [out, in]")
    if bits < 1 or bits > 8:
        raise ValueError("bits must lie in 1..8")
    rows, cols = matrix.shape
    qmax = (1 << bits) - 1
    n_groups_per_row = math.ceil(cols / group_size)
    n_groups = rows * n_groups_per_row
    # Vectorized over all groups. Every element goes through the same float64
    # operations as a per-group loop, so the output is bit-identical.
    padded_cols = n_groups_per_row * group_size
    pad = padded_cols - cols
    if pad:
        # The trailing partial group repeats its last real weight for min/max only.
        ranged = np.concatenate([matrix, np.repeat(matrix[:, -1:], pad, axis=1)], axis=1)
    else:
        ranged = matrix
    groups = ranged.reshape(rows, n_groups_per_row, group_size)
    w_min = groups.min(axis=-1)
    w_max = groups.max(axis=-1)
    scale = (w_max - w_min) / qmax
    scale = np.where((scale == 0.0) | ~np.isfinite(scale), 1.0, scale)
    zero = np.clip(np.rint(-w_min / scale), 0, qmax).astype(np.int64)
    quantized = np.clip(np.rint(groups / scale[..., None]) + zero[..., None], 0, qmax).astype(np.int64)
    values = (scale[..., None] * (quantized - zero[..., None])).reshape(rows, padded_cols)[:, :cols]
    values = np.ascontiguousarray(values)
    codes = np.ascontiguousarray(quantized.reshape(rows, padded_cols)[:, :cols]).reshape(-1)
    scales = scale.reshape(-1)
    zero_points = zero.reshape(-1)
    pad_counts = np.zeros((rows, n_groups_per_row), dtype=np.int64)
    pad_counts[:, -1] = pad
    pad_counts = pad_counts.reshape(-1)
    assert scales.shape[0] == n_groups
    return QuantizedTensor(
        values=values,
        scales=scales,
        zero_points=zero_points,
        pad_counts=pad_counts,
        codes=codes,
        bits=int(bits),
        group_size=int(group_size),
    )


def channel_scales(mean_abs: np.ndarray, alpha: float) -> np.ndarray:
    """AWQ-style per-input-channel scales for one alpha."""
    magnitude = np.asarray(mean_abs, dtype=np.float64).reshape(-1)
    if alpha == 0.0:
        powered = np.ones_like(magnitude)
    else:
        powered = np.power(np.maximum(magnitude, 0.0), float(alpha))
    denominator = float(powered.mean()) if powered.size else 0.0
    if denominator == 0.0 or not math.isfinite(denominator):
        scales = np.ones_like(magnitude)
    else:
        scales = powered / denominator
    scales = np.where(scales > 0.0, scales, 1.0)
    return scales.astype(np.float64)


def scale_weight_columns(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight, dtype=np.float64)
    factor = np.asarray(scales, dtype=np.float64).reshape(1, -1)
    if factor.shape[1] != matrix.shape[1]:
        raise ValueError("scale length must match the input dimension")
    return matrix * factor


def scale_activation_columns(inputs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    matrix = np.asarray(inputs, dtype=np.float64)
    factor = np.asarray(scales, dtype=np.float64).reshape(1, -1)
    if factor.shape[1] != matrix.shape[1]:
        raise ValueError("scale length must match the input dimension")
    return matrix / factor


def rescaled_outputs(weight: np.ndarray, inputs: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Original output and the output after the FP16-free column rescaling.

    Before rounding, ``(X / c) (W diag(c))^T`` equals ``X W^T``.
    """
    original = np.asarray(inputs, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
    scaled_weight = scale_weight_columns(weight, scales)
    scaled_input = scale_activation_columns(inputs, scales)
    rescaled = scaled_input @ scaled_weight.T
    return original, rescaled


def relative_rescale_error(weight: np.ndarray, inputs: np.ndarray, scales: np.ndarray) -> float:
    original, rescaled = rescaled_outputs(weight, inputs, scales)
    denominator = float(np.linalg.norm(original))
    if denominator == 0.0:
        return float(np.linalg.norm(rescaled))
    return float(np.linalg.norm(original - rescaled) / denominator)


def mean_output_mse(weight: np.ndarray, reconstructed: np.ndarray, inputs: np.ndarray) -> float:
    """Mean squared error of one projection, averaged over outputs and tokens."""
    original = np.asarray(inputs, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
    approx = np.asarray(inputs, dtype=np.float64) @ np.asarray(reconstructed, dtype=np.float64).T
    error = original - approx
    return float(np.mean(error * error))


@dataclass(frozen=True)
class AWQChoice:
    alpha: float
    scales: np.ndarray
    values_original_space: np.ndarray
    values_scaled_space: np.ndarray
    quantized_scaled: QuantizedTensor
    mse_by_alpha: dict[float, float]


def search_awq(
    weight: np.ndarray,
    inputs: np.ndarray,
    bits: int,
    group_size: int = GROUP_SIZE,
    alphas: tuple[float, ...] = AWQ_ALPHAS,
) -> AWQChoice:
    """Pick alpha by calibration output MSE. Equal MSE keeps the smaller alpha."""
    matrix = np.asarray(weight, dtype=np.float64)
    acts = np.asarray(inputs, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[1] != matrix.shape[1]:
        raise ValueError("inputs must be [tokens, in_features]")
    mean_abs = np.mean(np.abs(acts), axis=0)
    records: dict[float, float] = {}
    best: tuple[float, float, AWQChoice] | None = None
    for alpha in alphas:
        scales = channel_scales(mean_abs, alpha)
        scaled = scale_weight_columns(matrix, scales)
        quantized = quantize_weight(scaled, bits, group_size)
        original_space = quantized.values / scales.reshape(1, -1)
        mse = mean_output_mse(matrix, original_space, acts)
        records[float(alpha)] = mse
        choice = AWQChoice(
            alpha=float(alpha),
            scales=scales,
            values_original_space=original_space,
            values_scaled_space=quantized.values,
            quantized_scaled=quantized,
            mse_by_alpha=records,
        )
        if best is None or mse < best[0] - 1e-18 or (abs(mse - best[0]) <= 1e-18 and alpha < best[1]):
            best = (mse, float(alpha), choice)
    assert best is not None
    # The dataclass stores the dict that is still being filled; rebuild so the
    # returned object sees every alpha, not only those tried so far.
    chosen = best[2]
    return AWQChoice(
        alpha=chosen.alpha,
        scales=chosen.scales,
        values_original_space=chosen.values_original_space,
        values_scaled_space=chosen.values_scaled_space,
        quantized_scaled=chosen.quantized_scaled,
        mse_by_alpha=dict(records),
    )


def awq_reconstruct(
    weight: np.ndarray,
    mean_abs: np.ndarray,
    alpha: float,
    bits: int,
    group_size: int = GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(scales, values in original space, values in scaled space)``.

    Original space has already divided columns by ``c``, which is the stored-scale
    reconstruction. Scaled space keeps ``c`` inside the weight so a preceding
    norm or linear can carry the inverse scale.
    """
    scales = channel_scales(mean_abs, alpha)
    quantized = quantize_weight(scale_weight_columns(weight, scales), bits, group_size)
    original = quantized.values / scales.reshape(1, -1)
    return scales, original, quantized.values


def compensation_plan(scales_by_projection: dict[str, np.ndarray]) -> dict[str, str]:
    """Decide fold versus store for one block.

    ``down_proj`` and ``o_proj`` always store FP16 channel scales. Projections
    that share an RMSNorm fold into that norm only when every consumer has the
    same scale vector; otherwise each of them stores its own scales. A stored
    scale is later baked into the reconstructed weight for dense FP16 evaluation
    and still counted in the packed bytes.
    """
    from task1.src.constants import SHARED_RMSNORM_GROUPS, STORE_SCALE_PROJECTIONS

    plan = {name: "store" for name in scales_by_projection}
    for name in STORE_SCALE_PROJECTIONS:
        if name in plan:
            plan[name] = "store"
    for members in SHARED_RMSNORM_GROUPS.values():
        present = [name for name in members if name in scales_by_projection]
        if not present:
            continue
        matched = fold_scales_match([scales_by_projection[name] for name in present])
        mode = "fold" if matched else "store"
        for name in present:
            plan[name] = mode
    return plan


def fold_scales_match(scales: list[np.ndarray], rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """True when every consumer of a shared norm has the same channel scale."""
    if len(scales) <= 1:
        return True
    head = scales[0]
    return all(np.allclose(head, other, rtol=rtol, atol=atol) for other in scales[1:])
