"""MQAR training and evaluation.

Loss is cross-entropy on the value id at each query position. The training
stream is the course generator's ``train`` split. Development and test records
are the frozen files. Test accuracy is computed only by ``evaluate_split`` when
the caller asks for the test split; the pre-registered prediction is already in
``task2/PREDICTION.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW

from shared.paths import REPO_ROOT
from shared.seed import set_seed
from task2.src.model import MQARModel, query_cross_entropy, query_predictions
from task2.src.mqar_data import (
    bin_name,
    check_generator_invariants,
    distance_from_source,
    load_frozen,
    load_generator,
)
from task2.src.schedule import (
    BATCH_SIZE,
    EVAL_UPDATES,
    GRAD_CLIP,
    PEAK_LR,
    TOTAL_UPDATES,
    WEIGHT_DECAY,
    learning_rate,
)
from task2.src.theory import RECEPTIVE_FIELD_NOTE

SEEDS = (6304, 6305, 6306)

# Fourteen architectures. Three seeds make the 42 required runs.
JOBS: tuple[tuple[str, int, str], ...] = (
    ("dense", 4, "mixed"),
    ("dense", 8, "mixed"),
    ("dense", 16, "mixed"),
    ("dense", 16, "near"),
    ("window", 4, "near"),
    ("window", 8, "near"),
    ("window", 16, "near"),
    ("window", 16, "mixed"),
    ("linear", 4, "mixed"),
    ("linear", 8, "mixed"),
    ("linear", 16, "mixed"),
    ("mixed", 4, "mixed"),
    ("mixed", 16, "mixed"),
    ("gated", 16, "mixed"),
)

# Test-time window on a dense checkpoint versus a window model trained under the mask.
MASK_CONTROLS: tuple[tuple[tuple[str, int, str], tuple[str, int, str]], ...] = (
    (("dense", 16, "mixed"), ("window", 16, "mixed")),
    (("dense", 16, "near"), ("window", 16, "near")),
    (("dense", 4, "mixed"), ("window", 4, "near")),
    (("dense", 8, "mixed"), ("window", 8, "near")),
)


def results_dir() -> Path:
    return REPO_ROOT / "results" / "task2"


def checkpoints_dir() -> Path:
    return REPO_ROOT / "checkpoints" / "task2"


def job_name(kind: str, k: int, regime: str, seed: int) -> str:
    return f"{kind}_K{k}_{regime}_seed{seed}"


def build_optimizer(model: torch.nn.Module) -> AdamW:
    return AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)


def examples_to_batch(examples: list[dict], device: torch.device) -> dict:
    tokens = torch.tensor([example["tokens"] for example in examples], dtype=torch.long, device=device)
    query_positions = torch.tensor(
        [example["query_positions"] for example in examples], dtype=torch.long, device=device
    )
    answer_ids = torch.tensor([example["answer_ids"] for example in examples], dtype=torch.long, device=device)
    distances = torch.tensor([distance_from_source(example) for example in examples], dtype=torch.long, device=device)
    regimes = [example["regime"] for example in examples]
    return {
        "tokens": tokens,
        "query_positions": query_positions,
        "answer_ids": answer_ids,
        "distances": distances,
        "regimes": regimes,
    }


def summarize_hits(hits: list[dict], k: int) -> dict:
    def _pack(rows: list[dict]) -> dict:
        if not rows:
            return {"accuracy": None, "n": 0}
        correct = sum(row["hit"] for row in rows)
        return {"accuracy": correct / len(rows), "n": len(rows)}

    by_regime: dict[str, dict] = {}
    by_bin: dict[str, dict] = {}
    for regime in ("near", "far", "mixed"):
        by_regime[regime] = _pack([row for row in hits if row["regime"] == regime])
    for low, high in ((0, 32), (32, 64), (64, 128), (128, 256)):
        name = f"[{low},{high})"
        by_bin[name] = _pack([row for row in hits if row["bin"] == name])
    return {
        "accuracy": _pack(hits)["accuracy"],
        "n_queries": len(hits),
        "by_regime": by_regime,
        "by_bin": by_bin,
        "chance_uniform_vocab": 1 / 64,
        "chance_within_example": 1 / k,
        "receptive_field_note": RECEPTIVE_FIELD_NOTE,
    }


@torch.no_grad()
def evaluate_examples(model: MQARModel, examples: list[dict], device: torch.device, batch_size: int = BATCH_SIZE) -> dict:
    model.eval()
    hits: list[dict] = []
    k = int(examples[0]["K"]) if examples else 0
    for start in range(0, len(examples), batch_size):
        batch = examples_to_batch(examples[start : start + batch_size], device)
        logits = model(batch["tokens"])
        predictions = query_predictions(logits, batch["query_positions"]).cpu()
        answers = batch["answer_ids"].cpu()
        distances = batch["distances"].cpu()
        for row in range(predictions.shape[0]):
            regime = batch["regimes"][row]
            for query_index in range(predictions.shape[1]):
                distance = int(distances[row, query_index])
                hits.append(
                    {
                        "hit": int(predictions[row, query_index].item() == answers[row, query_index].item()),
                        "distance": distance,
                        "bin": bin_name(distance),
                        "regime": regime,
                    }
                )
    return summarize_hits(hits, k)


def evaluate_split(
    model: MQARModel,
    split: str,
    k: int,
    device: torch.device,
    regime: str | None = None,
) -> dict:
    """Evaluate a frozen split. ``split='test'`` is the held-out file, read only."""
    examples = load_frozen(split, k=k, regime=regime)
    metrics = evaluate_examples(model, examples, device)
    metrics["split"] = split
    metrics["K"] = k
    metrics["regime_filter"] = regime
    return metrics


def _train_batch(generator, k: int, regime: str, start: int, count: int) -> list[dict]:
    return [generator.make_example("train", k, regime, index) for index in range(start, start + count)]


def train_job(
    kind: str,
    k: int,
    regime: str,
    seed: int,
    updates: int = TOTAL_UPDATES,
    device: torch.device | None = None,
    evaluate_test: bool = False,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train one architecture. Checkpoints and curves go under results/ and checkpoints/."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_generator_invariants()
    generator = load_generator()
    development = load_frozen("development", k=k)
    set_seed(seed)
    model = MQARModel(kind).to(device)
    optimizer = build_optimizer(model)
    curve: list[dict] = []
    running_loss = 0.0
    running_steps = 0
    model.train()
    for update_index in range(updates):
        lr = learning_rate(update_index, total=updates)
        for group in optimizer.param_groups:
            group["lr"] = lr
        examples = _train_batch(generator, k, regime, update_index * batch_size, batch_size)
        batch = examples_to_batch(examples, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["tokens"])
        loss = query_cross_entropy(logits, batch["query_positions"], batch["answer_ids"])
        if not torch.isfinite(loss):
            curve.append({"update": update_index + 1, "train_loss": None, "failed": True})
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        running_loss += float(loss.item())
        running_steps += 1
        step = update_index + 1
        if step in EVAL_UPDATES or step == updates:
            dev_metrics = evaluate_examples(model, development, device, batch_size=batch_size)
            curve.append(
                {
                    "update": step,
                    "lr": lr,
                    "train_loss": running_loss / max(running_steps, 1),
                    "development": dev_metrics,
                }
            )
            running_loss = 0.0
            running_steps = 0
            model.train()
    name = job_name(kind, k, regime, seed)
    checkpoint_path = checkpoints_dir() / f"{name}.pt"
    curve_path = results_dir() / "learning_curves" / f"{name}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "kind": kind,
            "K": k,
            "regime": regime,
            "seed": seed,
            "updates": updates,
        },
        checkpoint_path,
    )
    payload = {
        "job": name,
        "kind": kind,
        "K": k,
        "regime": regime,
        "seed": seed,
        "updates": updates,
        "checkpoint": str(checkpoint_path),
        "learning_curve": curve,
        "receptive_field_note": RECEPTIVE_FIELD_NOTE,
        "development_note": (
            "Development accuracy describes this fixed run. "
            "It is not used to change the pre-registered forget gate."
        ),
    }
    if evaluate_test:
        payload["test"] = evaluate_split(model, "test", k, device)
    curve_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["curve_path"] = str(curve_path)
    return payload


def load_checkpoint(path: Path, device: torch.device) -> MQARModel:
    saved = torch.load(path, map_location=device, weights_only=False)
    model = MQARModel(saved["kind"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


@torch.no_grad()
def test_time_window_control(dense_path: Path, window_path: Path, k: int, device: torch.device) -> dict:
    """Put a W=64 mask on a trained dense model and compare it with a window-trained model."""
    dense = load_checkpoint(dense_path, device)
    window = load_checkpoint(window_path, device)
    test = load_frozen("test", k=k)
    dense.set_kind("window")
    masked = evaluate_examples(dense, test, device)
    trained = evaluate_examples(window, test, device)
    return {
        "K": k,
        "dense_with_test_time_window": masked,
        "window_trained_under_mask": trained,
        "receptive_field_note": RECEPTIVE_FIELD_NOTE,
    }


def smoke_optimizer_step(device: torch.device | None = None) -> float:
    """One AdamW step on random tokens. Does not read the test set or write results."""
    if device is None:
        device = torch.device("cpu")
    set_seed(6304)
    model = MQARModel("dense").to(device)
    optimizer = build_optimizer(model)
    tokens = torch.randint(0, 256, (2, 32), device=device)
    query_positions = torch.tensor([[1, 4], [2, 8]], device=device)
    answer_ids = torch.tensor([[130, 140], [150, 160]], device=device)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(0, total=TOTAL_UPDATES)
    optimizer.zero_grad(set_to_none=True)
    loss = query_cross_entropy(model(tokens), query_positions, answer_ids)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("smoke step produced a non-finite loss")
    return float(loss.item())
