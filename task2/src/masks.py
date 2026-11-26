"""Causal, sliding-window, and mixed local/dilated keep-masks.

Boolean ``True`` means the query may attend to that key, matching SDPA.
A dense boolean tensor is a correctness mask, not a sparse kernel.
"""

from __future__ import annotations

import torch

WINDOW = 64
DILATION_OFFSETS = tuple(range(0, 253, 4))  # 0, 4, ..., 252: 64 causal offsets


def causal_keep(length: int, device: torch.device | None = None) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    return positions.view(1, -1) <= positions.view(-1, 1)


def window_keep(length: int, window: int = WINDOW, device: torch.device | None = None) -> torch.Tensor:
    """Query ``i`` may use keys in ``[i - window + 1, i]`` intersected with ``[0, i]``."""
    if window < 1:
        raise ValueError("window must be positive")
    positions = torch.arange(length, device=device)
    queries = positions.view(-1, 1)
    keys = positions.view(1, -1)
    return (keys <= queries) & (keys >= queries - (window - 1))


def dilation_keep(
    length: int,
    offsets: tuple[int, ...] = DILATION_OFFSETS,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Query ``i`` may use ``i - offset`` for each offset that lands at a non-negative key."""
    keep = torch.zeros(length, length, dtype=torch.bool, device=device)
    positions = torch.arange(length, device=device)
    for offset in offsets:
        if offset < 0:
            raise ValueError("dilation offsets are causal and non-negative")
        if offset >= length:
            continue
        rows = positions[offset:]
        keep[rows, rows - offset] = True
    return keep


def mixed_local_dilated_keep(
    length: int,
    window: int = WINDOW,
    offsets: tuple[int, ...] = DILATION_OFFSETS,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Heads 0 and 1 are window-64. Heads 2 and 3 are the dilated offsets.

    Shape is ``(1, 4, L, L)`` so it broadcasts over the batch. Storage is a
    dense boolean mask.
    """
    local = window_keep(length, window, device)
    dilated = dilation_keep(length, offsets, device)
    mask = torch.stack((local, local, dilated, dilated), dim=0).unsqueeze(0)
    return mask


def allowed_key_counts(keep: torch.Tensor) -> torch.Tensor:
    """Number of allowed keys per query. ``keep`` is ``(L, L)`` or ``(1, H, L, L)``."""
    return keep.sum(dim=-1)


def window_allowed_count(position: int, window: int = WINDOW) -> int:
    return min(position + 1, window)


def dilation_allowed_count(position: int, offsets: tuple[int, ...] = DILATION_OFFSETS) -> int:
    return sum(1 for offset in offsets if 0 <= offset <= position)
