"""Task 1 stages: unit, smoke, baseline, prune, quantize, quantize_q3m, resources, analyze.

Run from the repository root:

    python task1/run.py unit
    python task1/run.py smoke --output task1/runs/smoke            # CPU, tiny hand-built block
    python task1/run.py pipeline-smoke --output /kaggle/working/task1_smoke   # GPU, real Qwen, every stage

``--smoke`` (or ``PA4_T1_SMOKE=1``) runs any GPU stage with the smoke profile in
``task1/src/run_profile.py``. ``pipeline-smoke`` runs all six stages that way, each
in its own process exactly like the notebook, into a directory that is marked
as a smoke run and refused by the real stages.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task1.src.gpu_smoke import run_gpu_smoke
from task1.src.pipeline import (
    run_analyze,
    run_baseline,
    run_prune,
    run_quantize,
    run_quantize_q3m,
    run_resources,
)
from task1.src.run_profile import SMOKE_ENV, SMOKE_RESOURCE_CONDITIONS
from task1.src.smoke import assert_smoke_output, run_smoke

PIPELINE_STAGES = ("baseline", "prune", "quantize", "quantize_q3m", "resources", "analyze")


def _unit() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "task1" / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


def _pipeline_smoke(output: Path, microbatch: int) -> None:
    """Every GPU stage in order with the smoke profile, one subprocess per stage like the notebook.

    resources runs with --remeasure on one condition per rebuild branch, so the recipe rebuild
    path (normally only used after a skipped or failed latency) is exercised too.
    """
    assert_smoke_output(output, ROOT)
    env = dict(os.environ)
    env[SMOKE_ENV] = "1"
    started = time.time()
    for stage in PIPELINE_STAGES:
        cmd = [sys.executable, str(ROOT / "task1" / "run.py"), stage, "--output", str(output), "--smoke",
               "--microbatch", str(microbatch)]
        if stage == "resources":
            cmd += ["--remeasure", "--only", ",".join(SMOKE_RESOURCE_CONDITIONS)]
        stage_start = time.time()
        print(f"==== pipeline-smoke: {stage} ====", flush=True)
        code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        print(f"pipeline-smoke: {stage} exit {code} in {time.time() - stage_start:.0f}s "
              f"(total {time.time() - started:.0f}s)", flush=True)
        if code != 0:
            raise SystemExit(f"pipeline-smoke FAILED at {stage} (exit {code})")
    print(f"PIPELINE SMOKE OK in {time.time() - started:.0f}s; outputs in {output}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Task 1 pruning and quantization")
    parser.add_argument(
        "stage",
        choices=(
            "unit",
            "smoke",
            "gpu-smoke",
            "pipeline-smoke",
            "baseline",
            "prune",
            "quantize",
            "quantize_q3m",
            "resources",
            "analyze",
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--remeasure", action="store_true", help="resources stage times every condition again")
    parser.add_argument("--only", default="", help="resources: comma-separated conditions to (re)measure")
    parser.add_argument("--smoke", action="store_true", help=f"tiny-size profile (same as {SMOKE_ENV}=1); never a result")
    args = parser.parse_args(argv)
    if args.smoke:
        os.environ[SMOKE_ENV] = "1"
    if args.stage == "pipeline-smoke":
        _pipeline_smoke(args.output or (ROOT / "task1" / "runs" / "pipeline-smoke"), args.microbatch)
        return
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
    smoke = os.environ.get(SMOKE_ENV) == "1"
    output = args.output or (ROOT / "task1" / ("runs/pipeline-smoke" if smoke else "results"))
    if smoke:
        assert_smoke_output(output, ROOT)
    if args.stage == "baseline":
        run_baseline(output, args.microbatch, args.skip_latency)
    elif args.stage == "prune":
        run_prune(output, args.microbatch, args.skip_latency)
    elif args.stage == "quantize":
        run_quantize(output, args.microbatch, args.skip_latency)
    elif args.stage == "quantize_q3m":
        run_quantize_q3m(output, args.microbatch, args.skip_latency)
    elif args.stage == "resources":
        only = [name for name in args.only.split(",") if name] or None
        run_resources(output, args.microbatch, args.remeasure, only)
    elif args.stage == "analyze":
        run_analyze(output)
    else:
        raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
