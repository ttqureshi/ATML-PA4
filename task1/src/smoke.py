"""Tiny CPU smoke test. It never loads Qwen and it never writes into results/."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from task1.src.data import write_json
from task1.src.model_ops import collect_activation_stats, list_eligible, load_qwen, parameter_census, read_weight, write_weight
from task1.src.pruning import magnitude_scores, uniform_keep_mask
from task1.src.quantize import quantize_weight, tensor_packed_bytes


def final_results_directories(root: Path) -> list[Path]:
    return [(root / "results").resolve(), (root / "task1" / "results").resolve()]


def assert_smoke_output(path: Path, root: Path) -> None:
    resolved = path.resolve()
    for forbidden in final_results_directories(root):
        if resolved == forbidden or forbidden in resolved.parents:
            raise SystemExit(
                f"smoke refuses to write into the final results directory ({resolved}). "
                "Point smoke at a scratch directory such as task1/runs/smoke."
            )


class _Attn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _MLP(nn.Module):
    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)


class _Block(nn.Module):
    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden)
        self.self_attn = _Attn(hidden)
        self.post_attention_layernorm = nn.RMSNorm(hidden)
        self.mlp = _MLP(hidden, inner)


class TinyLM(nn.Module):
    """One block with the Qwen projection names. Embeddings are tied to the head."""

    def __init__(self, vocab: int = 32, hidden: int = 16, inner: int = 32) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Block(hidden, inner)])
        self.norm = nn.RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, use_cache=False):
        del use_cache
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            residual = hidden
            normed = layer.input_layernorm(hidden)
            query = layer.self_attn.q_proj(normed)
            key = layer.self_attn.k_proj(normed)
            value = layer.self_attn.v_proj(normed)
            mixed = query + 0.0 * key + 0.0 * value
            hidden = residual + layer.self_attn.o_proj(mixed)
            residual = hidden
            normed = layer.post_attention_layernorm(hidden)
            gate = torch.nn.functional.silu(layer.mlp.gate_proj(normed))
            up = layer.mlp.up_proj(normed)
            hidden = residual + layer.mlp.down_proj(gate * up)
        logits = self.lm_head(self.norm(hidden))
        return type("TinyOut", (), {"logits": logits})()


def run_smoke(output: Path, root: Path) -> None:
    assert_smoke_output(output, root)
    previous = os.environ.get("PA4_FORBID_DOWNLOAD")
    os.environ["PA4_FORBID_DOWNLOAD"] = "1"
    try:
        try:
            load_qwen()
        except RuntimeError as exc:
            if "download" not in str(exc).lower():
                raise
        else:
            raise RuntimeError("smoke reached Qwen weight loading")
        torch.manual_seed(6304)
        model = TinyLM()
        modules = list_eligible(model)
        if len(modules) != 7:
            raise RuntimeError(f"smoke model exposed {len(modules)} eligible projections")
        census = parameter_census(model)
        tied = {row["name"] for row in census["tied"]}
        if "lm_head.weight" not in tied and "embed_tokens.weight" not in tied:
            raise RuntimeError("smoke model did not count the tied embedding once")
        sequences = torch.randint(0, 32, (2, 8)).tolist()
        stats = collect_activation_stats(model, modules, sequences, microbatch=2)
        if any(slot["n_tokens"] != 16 for slot in stats.values()):
            raise RuntimeError("activation hooks did not see both smoke sequences")
        weight = read_weight(modules[0][1])
        mask = uniform_keep_mask(magnitude_scores(weight), 0.5)
        write_weight(modules[0][1], weight * mask)
        quantized = quantize_weight(read_weight(modules[-1][1]), bits=3, group_size=8)
        expected = tensor_packed_bytes(*quantized.values.shape, 3, 8)
        if quantized.n_groups * 4 + (quantized.codes.size * 3 + 7) // 8 != expected:
            raise RuntimeError("smoke packed-byte count disagrees with the quantizer")
        with torch.inference_mode():
            logits = model(input_ids=torch.randint(0, 32, (1, 8))).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("smoke forward produced non-finite logits")
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "summary.json",
            {
                "stage": "smoke",
                "downloaded_weights": False,
                "eligible_matrices": len(modules),
                "tied": census["tied"],
                "unique_parameters": census["n_unique"],
                "logits_finite": True,
                "limits": {"sequences": 2, "length": 8, "hidden": 16},
            },
        )
        print(f"smoke: wrote {output / 'summary.json'}", flush=True)
    finally:
        if previous is None:
            os.environ.pop("PA4_FORBID_DOWNLOAD", None)
        else:
            os.environ["PA4_FORBID_DOWNLOAD"] = previous
