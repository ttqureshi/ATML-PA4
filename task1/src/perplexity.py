"""Token NLL and perplexity.

Perplexity is ``exp(sum of token NLL / N)``. The first token of a window is
never a scored target: a caller that passes a start offset of 0 is moved to 1.
``score_target_offsets`` from the locked windows are half-open ``[start, end)``
token positions. NLL at position ``t`` is ``-log p(x_t | x_<t)`` and is read
from the logits at position ``t - 1``.
"""

from __future__ import annotations

import math

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    peak = np.max(scores, axis=-1, keepdims=True)
    shifted = scores - peak
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def nll_from_logits(
    logits: np.ndarray,
    token_ids: np.ndarray,
    score_start: int,
    score_end: int,
) -> tuple[float, int]:
    """Sum of token NLLs and the scored-token count.

    ``logits[t]`` predicts ``token_ids[t + 1]``. Positions below 1 are dropped.
    """
    scores = np.asarray(logits, dtype=np.float64)
    tokens = np.asarray(token_ids)
    if scores.ndim != 2:
        raise ValueError("logits must have shape [positions, vocab]")
    if tokens.ndim != 1:
        raise ValueError("token_ids must be 1-D")
    if scores.shape[0] < tokens.shape[0]:
        raise ValueError("logits must cover every token position")
    start = max(int(score_start), 1)
    end = int(score_end)
    if end < start:
        raise ValueError("empty or reversed score range")
    if end > tokens.shape[0]:
        raise ValueError("score range extends past the tokens")
    log_probs = log_softmax(scores)
    total = 0.0
    for position in range(start, end):
        total += -float(log_probs[position - 1, int(tokens[position])])
    return total, end - start


def nll_from_token_logprobs(
    token_logprobs: np.ndarray,
    score_start: int = 1,
    score_end: int | None = None,
) -> tuple[float, int]:
    """Sum ``-log p`` over scored positions. Index 0 is not a target.

    ``token_logprobs[t]`` is already ``log p(x_t | x_<t)``. The value at index
    0 is ignored even when ``score_start`` is 0.
    """
    log_probs = np.asarray(token_logprobs, dtype=np.float64)
    if log_probs.ndim != 1:
        raise ValueError("token log-probabilities must be 1-D")
    end = int(log_probs.shape[0] if score_end is None else score_end)
    start = max(int(score_start), 1)
    if end < start or end > log_probs.shape[0]:
        raise ValueError("score range is outside the log-probability vector")
    chosen = log_probs[start:end]
    return float(-np.sum(chosen)), int(chosen.shape[0])


def aggregate_perplexity(window_nll_sums: np.ndarray, window_counts: np.ndarray) -> dict[str, float | int]:
    """Pool whole windows, then exponentiate the mean token NLL."""
    sums = np.asarray(window_nll_sums, dtype=np.float64)
    counts = np.asarray(window_counts, dtype=np.float64)
    if sums.shape != counts.shape or sums.ndim != 1:
        raise ValueError("window NLL sums and counts must be equal-length 1-D arrays")
    n_tokens = float(counts.sum())
    if n_tokens <= 0:
        raise ValueError("perplexity needs at least one scored token")
    total_nll = float(sums.sum())
    mean_nll = total_nll / n_tokens
    return {
        "nll_sum": total_nll,
        "mean_nll": mean_nll,
        "perplexity": float(math.exp(mean_nll)),
        "n_tokens": int(n_tokens),
        "n_windows": int(sums.shape[0]),
    }
