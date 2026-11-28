"""Full-run sizes versus the GPU pipeline smoke. Smoke shrinks sizes only, never logic.

``PA4_T1_SMOKE=1`` (or ``run.py <stage> --smoke``) selects the smoke profile:
few calibration/development sequences, shorter sequences, two evaluation
windows per suite, 16 HellaSwag items, the last decoder layer's seven
projections only, and one timed repetition. Every stage, method family,
allocation rule and the test-file gate still run. A smoke output directory
carries ``SMOKE_MARKER`` and every condition summary carries ``"smoke": true``,
so a smoke number can never be mixed into, or mistaken for, a real result.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from task1.src.constants import (
    CALIBRATION_SEQUENCES,
    DECODE_TOKENS,
    DEVELOPMENT_SEQUENCES,
    HELLASWAG_DIAGNOSTIC,
    TIMING_REPEATS,
    TIMING_WARMUP,
)

SMOKE_ENV = "PA4_T1_SMOKE"

# Smoke resources rebuilds one condition per rebuild branch (baseline, magnitude, Wanda without a
# sensitivity, 2:4, P3, Q1, Q2 alphas, Q3 bit map, Q3m) rather than all twenty.
SMOKE_RESOURCE_CONDITIONS = ("fp16", "p1_s50", "p2_s25", "p2_2to4", "p3_s75", "q1_b4", "q2_b3", "q3_b4", "q3m_b3")
SMOKE_MARKER = "SMOKE_RUN_NOT_RESULTS.txt"


@dataclass(frozen=True)
class RunProfile:
    smoke: bool
    calibration_sequences: int | None  # None keeps every locked row
    development_sequences: int | None
    sequence_length: int | None  # None keeps the locked length
    eval_windows: int | None  # per perplexity suite
    hellaswag_full: int | None  # None keeps all 2,000 rows
    hellaswag_diagnostic: int
    last_layer_only: bool  # compress the seven projections of the last block only
    timing_warmup: int
    timing_repeats: int
    decode_tokens: int

    def as_dict(self) -> dict:
        return asdict(self)


FULL = RunProfile(
    smoke=False,
    calibration_sequences=None,
    development_sequences=None,
    sequence_length=None,
    eval_windows=None,
    hellaswag_full=None,
    hellaswag_diagnostic=HELLASWAG_DIAGNOSTIC,
    last_layer_only=False,
    timing_warmup=TIMING_WARMUP,
    timing_repeats=TIMING_REPEATS,
    decode_tokens=DECODE_TOKENS,
)

SMOKE = RunProfile(
    smoke=True,
    calibration_sequences=8,
    development_sequences=4,
    sequence_length=128,
    eval_windows=2,
    hellaswag_full=16,
    hellaswag_diagnostic=8,
    last_layer_only=True,
    timing_warmup=1,
    timing_repeats=1,
    decode_tokens=8,
)


def current() -> RunProfile:
    return SMOKE if os.environ.get(SMOKE_ENV) == "1" else FULL


def shrink_rows(rows: list[list[int]], count: int | None, length: int | None) -> list[list[int]]:
    picked = rows if count is None else rows[:count]
    if length is not None:
        picked = [row[:length] for row in picked]
    return picked


def check_split_sizes(calibration: list[list[int]], development: list[list[int]], profile: RunProfile) -> None:
    """Full runs must use the locked 128 calibration and 32 development sequences."""
    if profile.smoke:
        return
    if len(calibration) != CALIBRATION_SEQUENCES:
        raise SystemExit(f"calibration lock has {len(calibration)} sequences, expected {CALIBRATION_SEQUENCES}")
    if len(development) != DEVELOPMENT_SEQUENCES:
        raise SystemExit(f"development lock has {len(development)} sequences, expected {DEVELOPMENT_SEQUENCES}")


def select_modules(modules: list[tuple[str, object]], num_layers: int, profile: RunProfile):
    if not profile.last_layer_only:
        return modules
    from task1.src.model_ops import layer_index

    return [(name, module) for name, module in modules if layer_index(name) == int(num_layers) - 1]


def summary_is_smoke(path: Path) -> bool:
    return bool(json.loads(path.read_text(encoding="utf-8")).get("smoke", False))


def check_output_dir(output: Path, profile: RunProfile) -> None:
    """Refuse to mix smoke and real conditions in one directory. Writes the smoke marker."""
    marker = output / SMOKE_MARKER
    summaries = sorted(output.glob("conditions/*/summary.json")) if output.is_dir() else []
    if profile.smoke:
        real = [path.parent.name for path in summaries if not summary_is_smoke(path)]
        if real:
            raise SystemExit(
                f"smoke refuses {output}: it already holds real conditions ({', '.join(real[:5])}). "
                "Give the smoke its own directory."
            )
        output.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "This directory is a Task 1 pipeline SMOKE run (tiny sizes, last layer only). "
            "Nothing here is a reportable result.\n",
            encoding="utf-8",
        )
        return
    if marker.exists():
        raise SystemExit(f"{output} is a smoke directory ({SMOKE_MARKER}). Write real results elsewhere.")
    smoke = [path.parent.name for path in summaries if summary_is_smoke(path)]
    if smoke:
        raise SystemExit(f"{output} holds smoke conditions ({', '.join(smoke[:5])}). Write real results elsewhere.")
