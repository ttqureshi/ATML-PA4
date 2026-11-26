"""Perplexity windows and HellaSwag conditional log-likelihoods."""

from __future__ import annotations

import numpy as np

from task1.src.hellaswag import aggregate_accuracy, hellaswag_prompt, score_endings
from task1.src.perplexity import aggregate_perplexity


def _loglikelihood_span(model, token_ids: list[int], start: int, end: int) -> float:
    """Sum of log p(token_t | tokens[:t]) for ``t`` in ``[start, end)``."""
    import torch
    import torch.nn.functional as F

    if end <= start:
        return float("-inf")
    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0]
    targets = ids[0, start:end]
    predicting = logits[start - 1 : end - 1, :].float()
    log_probs = F.log_softmax(predicting, dim=-1)
    chosen = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(chosen.sum().item())


def window_nll(model, token_ids: list[int], score_start: int, score_end: int) -> tuple[float, int]:
    """NLL sum and count. The first token of the window is never scored."""
    start = max(int(score_start), 1)
    end = int(score_end)
    if end > len(token_ids) or start > end:
        raise ValueError("score range does not fit the window")
    if end == start:
        return 0.0, 0
    total_ll = _loglikelihood_span(model, token_ids, start, end)
    return -total_ll, end - start


def evaluate_windows(model, records: list[dict]) -> dict:
    sums = []
    counts = []
    for record in records:
        start, end = record["score_target_offsets"]
        total, count = window_nll(model, [int(token) for token in record["token_ids"]], int(start), int(end))
        expected = int(record["scored_tokens"])
        if count != expected:
            raise ValueError(
                f"window scored {count} tokens but the lock says {expected}. "
                "The first token is excluded and offsets are half-open."
            )
        sums.append(total)
        counts.append(count)
    summary = aggregate_perplexity(np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64))
    summary["nll_sums"] = np.asarray(sums, dtype=np.float64)
    summary["counts"] = np.asarray(counts, dtype=np.float64)
    return summary


def development_nll(model, sequences: list[list[int]]) -> dict:
    """Mean token NLL on development sequences, dropping each first token."""
    sums = []
    counts = []
    for token_ids in sequences:
        total, count = window_nll(model, token_ids, 1, len(token_ids))
        sums.append(total)
        counts.append(count)
    return aggregate_perplexity(np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64))


def encode_context_continuation(tokenizer, context: str, continuation: str) -> tuple[list[int], int]:
    """Tokenize the pair the way lm-eval scores a continuation.

    Trailing spaces on the context move onto the continuation so the boundary
    does not depend on a trailing space being its own token. The continuation
    is the suffix of the joint encoding after the context encoding.
    """
    spaces = len(context) - len(context.rstrip(" "))
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    return whole, len(context_ids)


def continuation_loglikelihood(model, tokenizer, context: str, continuation: str) -> float:
    whole, start = encode_context_continuation(tokenizer, context, continuation)
    if start < 1 or start >= len(whole):
        return float("-inf")
    return _loglikelihood_span(model, whole, start, len(whole))


def evaluate_hellaswag(model, tokenizer, records: list[dict]) -> tuple[dict, list[dict]]:
    scored_rows = []
    for record in records:
        query, choices, gold = hellaswag_prompt(record["example"])
        loglikelihoods = [
            continuation_loglikelihood(model, tokenizer, query, choice) for choice in choices
        ]
        row = score_endings(np.asarray(loglikelihoods, dtype=np.float64), choices, gold)
        row["order"] = int(record["order"])
        row["source_row_index"] = int(record["source_row_index"])
        scored_rows.append(row)
    return aggregate_accuracy(scored_rows), scored_rows
