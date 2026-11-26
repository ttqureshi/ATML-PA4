"""Task 1 stages: unit, smoke, baseline, prune, quantize, resources, analyze.

Run from the repository root:

    python task1/run.py unit
    python task1/run.py smoke --output task1/runs/smoke
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task1.src.gpu_smoke import run_gpu_smoke
from task1.src.pipeline import run_analyze, run_baseline, run_prune, run_quantize, run_resources
from task1.src.smoke import run_smoke


def _unit() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "task1" / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Task 1 pruning and quantization")
    parser.add_argument(
        "stage",
        choices=("unit", "smoke", "gpu-smoke", "baseline", "prune", "quantize", "resources", "analyze"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--remeasure", action="store_true", help="resources stage times every condition again")
    args = parser.parse_args(argv)
    if args.stage == "unit":
        _unit()
        return
    if args.stage == "smoke":
        output = args.output or (ROOT / "task1" / "runs" / "smoke")
        run_smoke(output, ROOT)
        return
    if args.stage == "gpu-smoke":
        output = args.output or (ROOT / "task1" / "runs" / "gpu-smoke")
        run_gpu_smoke(output, ROOT)
        return
    output = args.output or (ROOT / "task1" / "results")
    if args.stage == "baseline":
        run_baseline(output, args.microbatch, args.skip_latency)
    elif args.stage == "prune":
        run_prune(output, args.microbatch, args.skip_latency)
    elif args.stage == "quantize":
        run_quantize(output, args.microbatch, args.skip_latency)
    elif args.stage == "resources":
        run_resources(output, args.microbatch, args.remeasure)
    elif args.stage == "analyze":
        run_analyze(output)
    else:
        raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
