"""Causal linear attention with ``phi(x) = ELU(x) + 1``, plus the forget gate.

Ungated training uses an inclusive prefix of outer products. That tensor is a
linear prefix state, not an ``L×L`` score matrix, and it matches the recurrence

    S_t = S_{t-1} + phi(k_t) v_t^T
    z_t = z_{t-1} + phi(k_t)
    output = phi(q)^T S / (phi(q)^T z + eps)

The gated update is a sequential scan. The multiplier is applied to the
previous state before the current token is written, so the persistent ``(S, z)``
shape is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LINEAR_EPS = 1e-6
GATE_BIAS_INIT = 2.0


def feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1


def linear_attention_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float = LINEAR_EPS,
) -> torch.Tensor:
    """Explicit token loop of the ungated recurrence. Includes the current token."""
    phi_q = feature_map(query)
    phi_k = feature_map(key)
    batch, heads, length, width = query.shape
    state = query.new_zeros(batch, heads, width, width)
    normalizer = query.new_zeros(batch, heads, width)
    outputs = []
    for step in range(length):
        state = state + phi_k[:, :, step, :, None] * value[:, :, step, None, :]
        normalizer = normalizer + phi_k[:, :, step, :]
        numerator = torch.einsum("bhd,bhde->bhe", phi_q[:, :, step, :], state)
        denominator = (phi_q[:, :, step, :] * normalizer).sum(dim=-1, keepdim=True) + eps
        outputs.append(numerator / denominator)
    return torch.stack(outputs, dim=2)


def linear_attention_prefix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float = LINEAR_EPS,
) -> torch.Tensor:
    """Same recurrence via an inclusive cumsum of ``phi(k) v^T``.

    The prefix tensor has shape ``(B, H, L, D, D)``. Report that storage when
    this path is timed. It is the training path because a Python scan launches
    one kernel per token.
    """
    phi_q = feature_map(query)
    phi_k = feature_map(key)
    outer = phi_k.unsqueeze(-1) * value.unsqueeze(-2)
    state = torch.cumsum(outer, dim=2)
    normalizer = torch.cumsum(phi_k, dim=2)
    numerator = torch.einsum("bhld,bhlde->bhle", phi_q, state)
    denominator = (phi_q * normalizer).sum(dim=-1, keepdim=True) + eps
    return numerator / denominator


def explicit_feature_sum(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eps: float = LINEAR_EPS,
) -> torch.Tensor:
    """``sum_{j<=t} (phi(q_t)·phi(k_j)) v_j`` over the causal prefix, per query."""
    phi_q = feature_map(query)
    phi_k = feature_map(key)
    length = query.shape[-2]
    outputs = []
    for step in range(length):
        dots = torch.einsum("bhd,bhjd->bhj", phi_q[:, :, step, :], phi_k[:, :, : step + 1, :])
        numerator = torch.einsum("bhj,bhjd->bhd", dots, value[:, :, : step + 1, :])
        denominator = dots.sum(dim=-1, keepdim=True) + eps
        outputs.append(numerator / denominator)
    return torch.stack(outputs, dim=2)


def forget_gate(
    phi_query_step: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """``sigmoid(w_h · phi(q_t) + b_h)`` with ``weight`` shaped ``(H, D)``."""
    return torch.sigmoid((phi_query_step * weight).sum(dim=-1) + bias)


def gated_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = LINEAR_EPS,
) -> torch.Tensor:
    """Per-head forget gate. ``bias`` is initialized to 2 so the gate starts near 0.88.

    ``S`` stays ``(B, H, D, D)`` and ``z`` stays ``(B, H, D)``. The gate is a
    scalar per head and is not stored as extra recurrent state.
    """
    phi_q = feature_map(query)
    phi_k = feature_map(key)
    batch, heads, length, width = query.shape
    state = query.new_zeros(batch, heads, width, width)
    normalizer = query.new_zeros(batch, heads, width)
    outputs = []
    for step in range(length):
        gate = forget_gate(phi_q[:, :, step, :], weight, bias)
        state = gate[:, :, None, None] * state + phi_k[:, :, step, :, None] * value[:, :, step, None, :]
        normalizer = gate[:, :, None] * normalizer + phi_k[:, :, step, :]
        numerator = torch.einsum("bhd,bhde->bhe", phi_q[:, :, step, :], state)
        denominator = (phi_q[:, :, step, :] * normalizer).sum(dim=-1, keepdim=True) + eps
        outputs.append(numerator / denominator)
    return torch.stack(outputs, dim=2)
