"""FP16 eviction and hybrid index selection.

Both eviction rules reserve 8 sink tokens and 64 most-recent context tokens.
Sink-plus-recent fills the rest with the next most recent positions.
Question-aware fills the rest with the highest FP32 attention scores.
Each layer keeps the same count, so the physical cache length is one number,
while the index set itself may differ by layer. Bytes include the int32 indices.
"""

from __future__ import annotations

import math

import torch

from task2.src.kv_budget import (
    GROUP,
    RECENT_TOKENS,
    SINK_TOKENS,
    eviction_bytes,
    hybrid_bytes,
    largest_fp16_count,
    memory_budget,
)


def protected_positions(length: int, sink: int = SINK_TOKENS, recent: int = RECENT_TOKENS) -> list[int]:
    chosen = set(range(min(sink, length)))
    if length > 0 and recent > 0:
        chosen.update(range(max(length - recent, 0), length))
    return sorted(chosen)


def select_sink_plus_recent(
    length: int,
    n_keep: int,
    sink: int = SINK_TOKENS,
    recent: int = RECENT_TOKENS,
) -> list[int]:
    protected = protected_positions(length, sink, recent)
    if n_keep < len(protected):
        raise ValueError("budget cannot hold the 8 sinks and 64 recent tokens")
    n_keep = min(n_keep, length)
    extra = n_keep - len(protected)
    protected_set = set(protected)
    recent_start = max(length - recent, 0)
    older = [index for index in range(recent_start - 1, -1, -1) if index not in protected_set]
    return sorted(protected + older[:extra])


def select_question_aware(
    scores: torch.Tensor,
    n_keep: int,
    sink: int = SINK_TOKENS,
    recent: int = RECENT_TOKENS,
) -> list[int]:
    """Rank positions that are not sinks or recent tokens. Ties prefer the later index."""
    length = int(scores.numel())
    protected = protected_positions(length, sink, recent)
    if n_keep < len(protected):
        raise ValueError("budget cannot hold the 8 sinks and 64 recent tokens")
    n_keep = min(n_keep, length)
    extra = n_keep - len(protected)
    protected_set = set(protected)
    candidates = [index for index in range(length) if index not in protected_set]
    candidates.sort(key=lambda index: (float(scores[index]), index), reverse=True)
    return sorted(protected + candidates[:extra])


def counts_for_budget(length: int, n_layers: int, h_kv: int, d_h: int, bits: int = 4) -> int:
    budget = memory_budget(length, n_layers, h_kv, d_h, bits)
    n_keep = largest_fp16_count(budget, length, n_layers, h_kv, d_h)
    protected = len(protected_positions(length))
    if n_keep < protected:
        raise ValueError(
            f"INT4 budget holds {n_keep} FP16 positions but the safeguards need {protected}"
        )
    if eviction_bytes(n_keep, n_layers, h_kv, d_h) > budget:
        raise AssertionError("selected eviction count exceeds the byte budget")
    return n_keep


def select_eviction(
    length: int,
    n_layers: int,
    h_kv: int,
    d_h: int,
    mode: str,
    scores_per_layer: torch.Tensor | None = None,
) -> tuple[list[list[int]], int, int]:
    """Return per-layer index sets, the shared count, and the serialized byte size."""
    n_keep = counts_for_budget(length, n_layers, h_kv, d_h, bits=4)
    budget = memory_budget(length, n_layers, h_kv, d_h, bits=4)
    chosen: list[list[int]] = []
    for layer in range(n_layers):
        if mode == "sink_recent":
            chosen.append(select_sink_plus_recent(length, n_keep))
        elif mode == "question":
            if scores_per_layer is None:
                raise ValueError("question-aware eviction needs one score vector per layer")
            chosen.append(select_question_aware(scores_per_layer[layer], n_keep))
        else:
            raise ValueError(f"unknown eviction mode {mode}")
    total = eviction_bytes(n_keep, n_layers, h_kv, d_h)
    if total > budget:
        raise AssertionError("eviction bytes exceed the case budget")
    return chosen, n_keep, total


def hybrid_keep_count(length: int, n_layers: int, h_kv: int, d_h: int) -> tuple[int, int, int]:
    """Largest FP16 count with the complement stored as complete INT2 groups.

    Returns ``(fp16_count, int2_count, total_bytes)``. The total includes
    metadata, int32 indices, and both payloads, and it does not exceed ``M_b``.
    """
    budget = memory_budget(length, n_layers, h_kv, d_h, bits=4)
    protected = len(protected_positions(length))
    best: tuple[int, int, int] | None = None
    for groups in range(0, length // GROUP + 1):
        quantized = groups * GROUP
        fp16_count = length - quantized
        if fp16_count < protected:
            continue
        total = hybrid_bytes(fp16_count, quantized, n_layers, h_kv, d_h, bits=2)
        if total <= budget and (best is None or fp16_count > best[0]):
            best = (fp16_count, quantized, total)
    if best is None:
        raise ValueError("budget cannot hold sinks, recent tokens, and the INT2 remainder")
    return best


def select_hybrid(
    scores_per_layer: torch.Tensor,
    n_layers: int,
    h_kv: int,
    d_h: int,
) -> dict:
    length = int(scores_per_layer.shape[-1])
    fp16_count, quantized, total = hybrid_keep_count(length, n_layers, h_kv, d_h)
    fp16_sets = [select_question_aware(scores_per_layer[layer], fp16_count) for layer in range(n_layers)]
    int2_sets = []
    all_positions = set(range(length))
    for kept in fp16_sets:
        rest = sorted(all_positions.difference(kept))
        if len(rest) % GROUP != 0:
            raise AssertionError("INT2 positions must form complete groups of 64")
        int2_sets.append(rest)
    return {
        "fp16_indices": fp16_sets,
        "int2_indices": int2_sets,
        "fp16_count": fp16_count,
        "int2_count": quantized,
        "bytes": total,
        "budget": memory_budget(length, n_layers, h_kv, d_h, bits=4),
        "padding_bytes": 0,
    }


def suffix_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    question_start: int,
) -> torch.Tensor:
    """Mean FP32 attention from question-suffix queries onto cached keys.

    ``query`` is ``(n_q, Q, D)`` and ``key`` is ``(n_kv, K, D)``, both already
    rotated. Scores are averaged across the query heads that share a KV head
    and then across KV heads. Positions are absolute rotary ids, so a kept
    subset is not renumbered before this call.
    """
    n_q, _, width = query.shape
    n_kv = key.shape[0]
    if n_q % n_kv != 0:
        raise ValueError("query heads must be divisible by KV heads")
    question = query_positions >= question_start
    if int(question.sum()) == 0:
        raise ValueError("question suffix is empty")
    # Score only the suffix. A full query-key matrix at 4,096 tokens does not fit on a T4.
    query = query[:, question, :]
    query_positions = query_positions[question]
    groups = n_q // n_kv
    query_f = query.float().reshape(n_kv, groups, query.shape[1], width)
    key_f = key.float()
    scores = torch.matmul(query_f, key_f.unsqueeze(1).transpose(-1, -2)) / math.sqrt(width)
    query_ids = query_positions.view(1, 1, -1, 1)
    key_ids = key_positions.view(1, 1, 1, -1)
    scores = scores.masked_fill(key_ids > query_ids, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights.mean(dim=(1, 2)).mean(dim=0)
