"""One CUDA forward and one locked WikiText window. Not a reported result."""

from __future__ import annotations

from pathlib import Path

import torch

from task1.src.data import TestFileGate, load_locked, write_json
from task1.src.evaluate import evaluate_windows
from task1.src.model_ops import list_eligible, load_qwen, parameter_census
from task1.src.smoke import assert_smoke_output


def run_gpu_smoke(output: Path, root: Path) -> None:
    assert_smoke_output(output, root)
    if not torch.cuda.is_available():
        raise SystemExit("gpu-smoke requires CUDA and will not download Qwen on CPU.")
    model, _tokenizer = load_qwen()
    name = getattr(model.config, "name_or_path", "")
    if "Qwen2.5-1.5B" not in name or name.endswith("Instruct"):
        raise RuntimeError(f"gpu-smoke loaded {name}, expected the base Qwen2.5-1.5B checkpoint")
    layers = int(model.config.num_hidden_layers)
    eligible = list_eligible(model)
    if len(eligible) != layers * 7:
        raise RuntimeError(f"found {len(eligible)} eligible projections, expected {layers * 7}")
    census = parameter_census(model)
    device = next(model.parameters()).device
    probe = torch.randint(0, 128, (1, 32), device=device)
    with torch.inference_mode():
        logits = model(input_ids=probe, use_cache=False).logits
    if logits.shape[1] != 32 or not torch.isfinite(logits).all():
        raise RuntimeError("gpu-smoke forward was the wrong shape or non-finite")
    gate = TestFileGate()
    gate.freeze("gpu smoke reads one locked window and does not choose an allocation")
    window = load_locked("wikitext", gate)[:1]
    scored = evaluate_windows(model, window)
    summary = {
        "model": name,
        "eligible_projections": len(eligible),
        "tied_names": [row["name"] for row in census["tied"]],
        "parameters_counted_once": census["n_unique"],
        "forward_tokens": 32,
        "logits_finite": True,
        "wikitext_windows": 1,
        "wikitext_scored_tokens": int(scored["n_tokens"]),
        "wikitext_mean_nll": float(scored["mean_nll"]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "note": "One window only. Do not cite this number as the Task 1 baseline.",
    }
    write_json(output / "gpu_smoke.json", summary)
    print(
        f"gpu-smoke ok projections={len(eligible)} "
        f"scored_tokens={summary['wikitext_scored_tokens']} "
        f"mean_nll={summary['wikitext_mean_nll']:.4f}",
        flush=True,
    )
