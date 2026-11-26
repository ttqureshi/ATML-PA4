"""Qwen weight IO, activation statistics, and dense FP16 evaluation forwards.

Loading the checkpoint is refused when ``PA4_FORBID_DOWNLOAD=1``. Smoke sets
that variable and never reaches Hugging Face.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from task1.src.constants import ELIGIBLE_PROJECTIONS, MODEL_ID, SHARED_RMSNORM_GROUPS


def load_qwen():
    """Load a fresh FP16 ``Qwen/Qwen2.5-1.5B`` and its tokenizer."""
    if os.environ.get("PA4_FORBID_DOWNLOAD") == "1":
        raise RuntimeError(
            "Refusing to load Qwen weights because PA4_FORBID_DOWNLOAD=1. Smoke must not download weights."
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
    except (TypeError, ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def list_eligible(model) -> list[tuple[str, object]]:
    import torch

    found = []
    for name, module in model.named_modules():
        kind = name.rsplit(".", 1)[-1]
        if kind in ELIGIBLE_PROJECTIONS and isinstance(module, torch.nn.Linear):
            found.append((name, module))
    return found


def parameter_census(model) -> dict:
    eligible = list_eligible(model)
    eligible_weights = {f"{name}.weight" for name, _module in eligible}
    seen: dict[int, str] = {}
    tied = []
    n_eligible = 0
    n_excluded = 0
    # remove_duplicate would hide lm_head when it shares storage with the embedding.
    for name, param in model.named_parameters(remove_duplicate=False):
        pointer = param.data_ptr()
        if pointer in seen:
            tied.append({"name": name, "shares_with": seen[pointer], "numel": int(param.numel())})
            continue
        seen[pointer] = name
        if name in eligible_weights:
            n_eligible += int(param.numel())
        else:
            n_excluded += int(param.numel())
    return {
        "n_eligible": n_eligible,
        "n_excluded": n_excluded,
        "n_unique": n_eligible + n_excluded,
        "n_eligible_matrices": len(eligible),
        "tied": tied,
        "eligible_names": [name for name, _module in eligible],
    }


def validate_layout(model, census: dict) -> None:
    config = model.config
    expected = int(config.num_hidden_layers) * len(ELIGIBLE_PROJECTIONS)
    if census["n_eligible_matrices"] != expected:
        raise RuntimeError(
            f"found {census['n_eligible_matrices']} eligible matrices, expected {expected}"
        )
    if getattr(config, "tie_word_embeddings", False):
        linked = {
            frozenset((row["name"], row["shares_with"]))
            for row in census["tied"]
        }
        wanted = frozenset(("lm_head.weight", "model.embed_tokens.weight"))
        if wanted not in linked:
            raise RuntimeError(
                "tie_word_embeddings is set, but lm_head.weight and model.embed_tokens.weight do not share storage"
            )


def snapshot_state(model) -> dict:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters(remove_duplicate=False)
    }


def restore_state(model, snapshot: dict) -> None:
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters(remove_duplicate=False):
            param.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))


def read_weight(module) -> np.ndarray:
    return module.weight.detach().float().cpu().numpy().astype(np.float64, copy=False)


def write_weight(module, values: np.ndarray) -> None:
    import torch

    array = np.ascontiguousarray(values, dtype=np.float32)
    tensor = torch.from_numpy(array)
    with torch.no_grad():
        module.weight.copy_(tensor.to(device=module.weight.device, dtype=module.weight.dtype))


def block_prefix(module_name: str) -> str:
    parts = module_name.split(".")
    if "layers" not in parts:
        return module_name.rsplit(".", 1)[0]
    index = parts.index("layers")
    return ".".join(parts[: index + 2])


def projection_kind(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    if "layers" not in parts:
        return -1
    return int(parts[parts.index("layers") + 1])


def norm_name_for(module_name: str) -> str | None:
    kind = projection_kind(module_name)
    prefix = block_prefix(module_name)
    for norm, members in SHARED_RMSNORM_GROUPS.items():
        if kind in members:
            return f"{prefix}.{norm}"
    return None


def _batched_ids(sequences: list[list[int]], microbatch: int):
    import torch

    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    for start in range(0, len(sequences), microbatch):
        chunk = sequences[start : start + microbatch]
        yield torch.tensor(chunk, dtype=torch.long)


def run_forwards(model, sequences: list[list[int]], microbatch: int) -> None:
    import torch

    device = next(model.parameters()).device
    for batch in _batched_ids(sequences, microbatch):
        with torch.inference_mode():
            model(input_ids=batch.to(device), use_cache=False)


def collect_activation_stats(model, modules, sequences: list[list[int]], microbatch: int) -> dict[str, dict]:
    stats = {name: {"sum_abs": None, "sum_sq": None, "n": 0} for name, _module in modules}
    hooks = []
    for name, module in modules:
        def pre_hook(mod, args, name=name):
            hidden = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            sum_abs = hidden.abs().sum(dim=0).cpu().numpy()
            sum_sq = hidden.square().sum(dim=0).cpu().numpy()
            slot = stats[name]
            if slot["sum_abs"] is None:
                slot["sum_abs"] = np.array(sum_abs, dtype=np.float64)
                slot["sum_sq"] = np.array(sum_sq, dtype=np.float64)
            else:
                slot["sum_abs"] += sum_abs
                slot["sum_sq"] += sum_sq
            slot["n"] += int(hidden.shape[0])

        hooks.append(module.register_forward_pre_hook(pre_hook))
    try:
        run_forwards(model, sequences, microbatch)
    finally:
        for hook in hooks:
            hook.remove()
    packed = {}
    for name, slot in stats.items():
        if slot["n"] <= 0 or slot["sum_abs"] is None:
            raise RuntimeError(f"no calibration activations reached {name}")
        packed[name] = {
            "mean_abs": slot["sum_abs"] / slot["n"],
            "column_l2": np.sqrt(slot["sum_sq"]),
            "n_tokens": int(slot["n"]),
        }
    return packed


def measure_reconstruction(model, modules, sequences, microbatch, reconstructed: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Streaming output SSE of ``X @ W_hat.T`` against the live (original) weight."""
    import torch

    totals = {name: {"sse": 0.0, "base_sse": 0.0, "n_elements": 0} for name in reconstructed}
    hooks = []
    for name, module in modules:
        if name not in reconstructed:
            continue

        def pre_hook(mod, args, name=name):
            hidden = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            weight = mod.weight.detach().float()
            candidate = torch.from_numpy(np.ascontiguousarray(reconstructed[name], dtype=np.float32))
            candidate = candidate.to(device=weight.device)
            original = torch.nn.functional.linear(hidden, weight)
            approx = torch.nn.functional.linear(hidden, candidate)
            diff = original - approx
            slot = totals[name]
            slot["sse"] += float(diff.square().sum().item())
            slot["base_sse"] += float(original.square().sum().item())
            slot["n_elements"] += int(diff.numel())

        hooks.append(module.register_forward_pre_hook(pre_hook))
    try:
        run_forwards(model, sequences, microbatch)
    finally:
        for hook in hooks:
            hook.remove()
    report = {}
    for name, slot in totals.items():
        if slot["n_elements"] <= 0:
            raise RuntimeError(f"reconstruction check saw no tokens at {name}")
        base = slot["base_sse"]
        report[name] = {
            "sse": slot["sse"],
            "base_sse": base,
            "n_elements": slot["n_elements"],
            "mse": slot["sse"] / slot["n_elements"],
            "relative_mse": 0.0 if base == 0.0 else slot["sse"] / base,
        }
    return report


def grouped_modules(modules: list[tuple[str, object]]) -> dict[str, list[tuple[str, object]]]:
    groups: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for name, module in modules:
        groups[block_prefix(name)].append((name, module))
    return dict(groups)


def scale_norm_weight(model, norm_module_name: str, scales: np.ndarray) -> None:
    import torch

    norm = model.get_submodule(norm_module_name)
    factor = torch.as_tensor(np.ascontiguousarray(scales, dtype=np.float32), device=norm.weight.device)
    if factor.numel() != norm.weight.numel():
        raise RuntimeError(
            f"cannot fold scales of length {factor.numel()} into {norm_module_name} "
            f"of length {norm.weight.numel()}"
        )
    with torch.no_grad():
        updated = norm.weight.detach().float() / factor
        norm.weight.copy_(updated.to(dtype=norm.weight.dtype))
