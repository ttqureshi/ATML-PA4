"""AdamW learning-rate schedule for the MQAR runs.

200 updates of linear warmup to the peak, then linear decay to 15% of the peak
at the final update. ``update_index`` is zero-based: index 199 is the 200th
update and sits at the peak; index 4499 is the last update.
"""

from __future__ import annotations

PEAK_LR = 8e-4
WEIGHT_DECAY = 0.01
WARMUP_UPDATES = 200
TOTAL_UPDATES = 4500
FINAL_LR_FRACTION = 0.15
GRAD_CLIP = 1.0
BATCH_SIZE = 32
EVAL_UPDATES = (500, 1000, 1500, 2000, 3000, 4500)


def learning_rate(
    update_index: int,
    total: int = TOTAL_UPDATES,
    warmup: int = WARMUP_UPDATES,
    peak: float = PEAK_LR,
    final_fraction: float = FINAL_LR_FRACTION,
) -> float:
    if update_index < 0 or update_index >= total:
        raise ValueError("update_index must lie in [0, total)")
    if warmup < 1 or warmup > total:
        raise ValueError("warmup must be in [1, total]")
    if update_index < warmup:
        return peak * float(update_index + 1) / float(warmup)
    span = total - warmup
    progress = (update_index - warmup + 1) / span
    floor = peak * final_fraction
    return peak + (floor - peak) * progress
