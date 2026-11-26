"""Task 2 entry point. Run from the worktree root: ``python task2/run.py <stage>``."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task2.src.analyze import analyze, missing_gpu_work  # noqa: E402
from task2.src.attention import dense_causal_attention, future_perturbation_changes_only_future  # noqa: E402
from task2.src.benchmark import benchmark_attention, forward_sweep  # noqa: E402
from task2.src.kv_budget import memory_budget  # noqa: E402
from task2.src.model import MQARModel  # noqa: E402
from task2.src.theory import linear_state_bytes  # noqa: E402
from task2.src.train import JOBS, SEEDS, results_dir, smoke_optimizer_step, train_job  # noqa: E402


def _results_files() -> set[Path]:
    root = results_dir()
    if not root.exists():
        return set()
    return {path for path in root.rglob("*") if path.is_file()}


def stage_unit() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "task2" / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def stage_smoke() -> None:
    before = _results_files()
    loss = smoke_optimizer_step()
    if loss != loss:
        raise SystemExit("smoke optimizer step was not finite")
    import torch

    query = torch.randn(1, 2, 16, 32)
    key = torch.randn(1, 2, 16, 32)
    value = torch.randn(1, 2, 16, 32)
    delta = future_perturbation_changes_only_future(dense_causal_attention, query, key, value, position=4)
    if float(delta.max()) != 0.0:
        raise SystemExit("smoke causality check failed")
    baseline = sum(parameter.numel() for parameter in MQARModel("linear").parameters())
    gated = sum(parameter.numel() for parameter in MQARModel("gated").parameters())
    if (gated - baseline) / baseline >= 0.05:
        raise SystemExit("forget gate exceeds the 5% parameter budget")
    if linear_state_bytes() <= 0:
        raise SystemExit("linear state bytes were not computed")
    hand = 2 * 2 * 3 * 5 * (1 * (32 + 4) + 2 * 16)
    if memory_budget(80, 2, 3, 5, 4) != hand:
        raise SystemExit("smoke budget check failed")
    created = _results_files() - before
    if created:
        raise SystemExit(f"smoke wrote into the final results directory: {sorted(created)}")
    print(f"smoke ok loss={loss:.4f}")


def stage_attention(max_length: int | None) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lengths = (256, 512, 1024, 2048, 4096, 8192)
    if device.type != "cuda":
        cap = 128 if max_length is None else max_length
        lengths = tuple(length for length in lengths if length <= cap)
        print("CPU attention run is a short check. The official grid is the GPU stage.")
    elif max_length is not None:
        lengths = tuple(length for length in lengths if length <= max_length)
    rows = benchmark_attention(lengths=lengths, device=device)
    sweep_lengths = (256, 1024, 4096, 8192)
    if device.type != "cuda":
        sweep_lengths = tuple(length for length in sweep_lengths if length <= (max_length or 128))
    sweep = forward_sweep(lengths=sweep_lengths, device=device)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "attention.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "forward_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print(f"wrote {out / 'attention.json'}")


def stage_train(evaluate_test: bool, updates: int, only: str | None) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for kind, k, regime in JOBS:
        name = f"{kind}_K{k}_{regime}"
        if only is not None and only not in name:
            continue
        for seed in SEEDS:
            print(f"training {name} seed {seed}")
            train_job(
                kind,
                k,
                regime,
                seed,
                updates=updates,
                device=device,
                evaluate_test=evaluate_test,
            )


def stage_kv(limit: int | None, pack_benchmark: bool) -> None:
    import torch

    from task2.src.kv_eval import benchmark_pack_unpack, load_qwen, run_kv_suite, write_records

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        records = run_kv_suite(device=device, limit=limit)
    except Exception as exc:
        if device.type != "cuda":
            print("Qwen weights are not available locally and will not be downloaded on CPU.")
            print(f"{type(exc).__name__}: {exc}")
            print("GPU stage still required: kv on Qwen/Qwen2.5-1.5B-Instruct in FP16.")
            raise SystemExit(2) from exc
        raise
    out = results_dir()
    write_records(records, out / "kv_records.json")
    if pack_benchmark:
        _model, _tokenizer, dims = load_qwen(device, local_files_only=True)
        n_layers, h_kv, d_h = dims
        n_q = int(_model.config.num_attention_heads)
        bench = benchmark_pack_unpack(n_layers, h_kv, d_h, n_q, device)
        (out / "kv_pack_benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    print(f"wrote {out / 'kv_records.json'}")


def stage_analyze() -> None:
    report = analyze()
    print(report["receptive_field_note"])
    missing = report["missing_gpu_stages"]
    if missing:
        print("GPU stages still required:")
        for item in missing:
            print(f"- {item}")
    else:
        print("saved measurements are present")
    print("missing check:", missing_gpu_work())


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 stages")
    parser.add_argument("stage", choices=("unit", "smoke", "attention", "train", "kv", "analyze"))
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--updates", type=int, default=4500)
    parser.add_argument("--evaluate-test", action="store_true", default=True)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--only", default=None, help="substring filter for a training job name")
    parser.add_argument("--limit", type=int, default=None, help="optional prompt cap for a KV debug run")
    parser.add_argument("--skip-pack-benchmark", action="store_true")
    args = parser.parse_args()
    if args.stage == "unit":
        stage_unit()
    elif args.stage == "smoke":
        stage_smoke()
    elif args.stage == "attention":
        stage_attention(args.max_length)
    elif args.stage == "train":
        stage_train(evaluate_test=not args.skip_test, updates=args.updates, only=args.only)
    elif args.stage == "kv":
        stage_kv(limit=args.limit, pack_benchmark=not args.skip_pack_benchmark)
    else:
        stage_analyze()


if __name__ == "__main__":
    main()
