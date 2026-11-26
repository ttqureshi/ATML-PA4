"""Prefix alignment, analytical speedup, and paired bootstrap summaries.

Forward KL is KL(teacher || draft). Reverse KL is KL(draft || teacher).
Expected one-token acceptance at a fixed prefix is 1 - TV(teacher, draft),
which equals the sum of the pointwise minima. Token-weighted averages divide
by the number of scored prefix positions, not by the number of prompts.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from shared.bootstrap import bootstrap_paired_mean_diff

POSITION_BINS: tuple[tuple[str, int, int], ...] = (
    ("1-16", 1, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-192", 65, 192),
)
DOMAINS: tuple[str, ...] = ("math", "familiar", "shifted")


def _vector(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("distributions must be one-dimensional")
    return array


def forward_kl(teacher, draft) -> float:
    """KL(teacher || draft), in nats. Mass where the teacher is zero is skipped."""
    teacher_probs = _vector(teacher)
    draft_probs = _vector(draft)
    if teacher_probs.shape != draft_probs.shape:
        raise ValueError("teacher and draft distributions differ in length")
    total = 0.0
    for teacher_prob, draft_prob in zip(teacher_probs, draft_probs):
        if teacher_prob == 0.0:
            continue
        if teacher_prob < 0.0 or draft_prob < 0.0:
            raise ValueError("probabilities must be non-negative")
        if draft_prob == 0.0:
            return math.inf
        total += float(teacher_prob) * math.log(float(teacher_prob) / float(draft_prob))
    return total


def reverse_kl(teacher, draft) -> float:
    """KL(draft || teacher)."""
    return forward_kl(draft, teacher)


def total_variation(teacher, draft) -> float:
    teacher_probs = _vector(teacher)
    draft_probs = _vector(draft)
    return 0.5 * float(np.abs(teacher_probs - draft_probs).sum())


def expected_acceptance(teacher, draft) -> float:
    """Sum of pointwise minima, equal to 1 - TV for distributions that sum to one."""
    return float(np.minimum(_vector(teacher), _vector(draft)).sum())


def greedy_agreement(teacher, draft) -> float:
    """1 when the first maximizing token agrees, else 0. Ties take the smallest index."""
    teacher_probs = _vector(teacher)
    draft_probs = _vector(draft)
    return float(int(np.argmax(teacher_probs)) == int(np.argmax(draft_probs)))


def prefix_alignment(teacher, draft) -> dict[str, float]:
    tv = total_variation(teacher, draft)
    return {
        "forward_kl": forward_kl(teacher, draft),
        "reverse_kl": reverse_kl(teacher, draft),
        "tv": tv,
        "greedy_agreement": greedy_agreement(teacher, draft),
        "expected_acceptance": 1.0 - tv,
    }


def position_bin(position: int) -> str:
    for name, low, high in POSITION_BINS:
        if low <= position <= high:
            return name
    return "other"


def aggregate_token_metrics(rows: list[dict]) -> dict[str, float]:
    """Mean of per-prefix metrics. Each row is one scored token position."""
    if not rows:
        return {
            "n_tokens": 0,
            "forward_kl": math.nan,
            "reverse_kl": math.nan,
            "tv": math.nan,
            "greedy_agreement": math.nan,
            "expected_acceptance": math.nan,
        }
    n = len(rows)
    totals = {key: 0.0 for key in ("forward_kl", "reverse_kl", "tv", "greedy_agreement", "expected_acceptance")}
    for row in rows:
        for key in totals:
            totals[key] += float(row[key])
    summary = {key: value / n for key, value in totals.items()}
    summary["n_tokens"] = n
    return summary


def breakdown_by_domain_and_bin(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["bin"])].append(row)
    report: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        for name, _low, _high in POSITION_BINS:
            report[f"{domain}|{name}"] = aggregate_token_metrics(grouped.get((domain, name), []))
    return report


def seed_dispersion(values_by_seed: dict[int, float]) -> dict[str, float | dict[int, float]]:
    if not values_by_seed:
        raise ValueError("seed dispersion needs at least one seed")
    values = [float(value) for value in values_by_seed.values()]
    return {
        "per_seed": {int(seed): float(value) for seed, value in values_by_seed.items()},
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def analytical_speedup(alpha: float, gamma: int, draft_over_target: float, verify_over_target: float) -> float:
    """S_hat(gamma) = (1 + sum_{j=1}^{gamma} alpha^j) / (gamma * c + v_gamma)."""
    if gamma < 1:
        raise ValueError("gamma must be at least 1")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    numerator = 1.0 + sum(alpha**j for j in range(1, gamma + 1))
    denominator = gamma * draft_over_target + verify_over_target
    if denominator <= 0.0:
        raise ValueError("predicted cost must be positive")
    return numerator / denominator


def _speedup_numerator(alpha: float, gamma: int) -> float:
    return 1.0 + sum(alpha**j for j in range(1, gamma + 1))


def break_even_acceptance(gamma: int, draft_over_target: float, verify_over_target: float) -> float | None:
    """Smallest alpha in [0, 1] with predicted speedup strictly above 1.

    Returns None when even alpha = 1 does not clear the cost. The value is the
    root of the equality when the crossing lies inside the interval; every
    larger acceptance is predicted to be faster than target-only decoding.
    """
    if gamma < 1:
        raise ValueError("gamma must be at least 1")
    rhs = gamma * draft_over_target + verify_over_target
    if _speedup_numerator(1.0, gamma) <= rhs:
        return None
    if _speedup_numerator(0.0, gamma) > rhs:
        return 0.0
    low = 0.0
    high = 1.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if _speedup_numerator(mid, gamma) > rhs:
            high = mid
        else:
            low = mid
    return high


def requests_to_amortize(upfront_seconds: float, seconds_saved_per_request: float) -> float | None:
    """How many requests pay back distillation plus merge. None if there is no saving."""
    if upfront_seconds < 0.0:
        raise ValueError("upfront cost must be non-negative")
    if seconds_saved_per_request <= 0.0:
        return None
    return upfront_seconds / seconds_saved_per_request


def bootstrap_throughput(speculative_tps, target_tps, *, seed: int = 6304) -> dict[str, float]:
    """Paired bootstrap of speculative throughput minus the adjacent target-only run."""
    return bootstrap_paired_mean_diff(
        np.asarray(speculative_tps, dtype=np.float64),
        np.asarray(target_tps, dtype=np.float64),
        seed=seed,
    )


def bootstrap_mean(values, *, seed: int = 6304) -> dict[str, float]:
    """Bootstrap CI for a mean, via the paired helper against a zero baseline."""
    samples = np.asarray(values, dtype=np.float64)
    return bootstrap_paired_mean_diff(samples, np.zeros(samples.shape, dtype=np.float64), seed=seed)


def cost_ratios(draft_step_s: float, target_step_s: float, verify_s: float) -> dict[str, float]:
    """c = draft step / target step, v_gamma = verification / one target step."""
    if target_step_s <= 0.0:
        raise ValueError("target step cost must be positive")
    return {
        "c": draft_step_s / target_step_s,
        "v_gamma": verify_s / target_step_s,
    }


def aligned_disagreements(rows: list[dict], limit: int = 8) -> list[dict]:
    """A few matched prefixes where the greedy tokens differ."""
    disagreements = [row for row in rows if float(row.get("greedy_agreement", 1.0)) == 0.0]
    disagreements.sort(key=lambda row: (-float(row.get("tv", 0.0)), str(row.get("prompt_id")), int(row.get("position", 0))))
    picked = []
    for row in disagreements[:limit]:
        picked.append(
            {
                "prompt_id": row.get("prompt_id"),
                "position": row.get("position"),
                "domain": row.get("domain"),
                "bin": row.get("bin"),
                "bank": row.get("bank"),
                "draft": row.get("draft"),
                "teacher_token": row.get("teacher_token"),
                "draft_token": row.get("draft_token"),
                "forward_kl": row.get("forward_kl"),
                "tv": row.get("tv"),
            }
        )
    return picked
