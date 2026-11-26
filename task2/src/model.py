"""Two-block pre-norm decoder used for every MQAR comparison.

Only the attention operator changes. Width 128, four heads of dimension 32,
MLP width 512, learned token embeddings, fixed sinusoidal positions, vocabulary
256. Projections are bias-free. The forget gate adds a per-head vector and a
per-head bias inside the attention module and nothing else.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from task2.src.attention import dense_causal_attention
from task2.src.linear import (
    GATE_BIAS_INIT,
    gated_linear_attention,
    linear_attention_prefix,
)
from task2.src.masks import mixed_local_dilated_keep, window_keep

VOCAB = 256
WIDTH = 128
HEADS = 4
HEAD_DIM = 32
LAYERS = 2
MLP_WIDTH = 512
ATTENTION_KINDS = ("dense", "window", "linear", "gated", "mixed")


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    angles = position * frequencies
    encoded = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoded[:, 0::2] = torch.sin(angles)
    encoded[:, 1::2] = torch.cos(angles)
    return encoded


class Attention(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in ATTENTION_KINDS:
            raise ValueError(f"unknown attention kind {kind}")
        self.kind = kind
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.proj = nn.Linear(WIDTH, WIDTH, bias=False)
        if kind == "gated":
            self.gate_weight = nn.Parameter(torch.zeros(HEADS, HEAD_DIM))
            self.gate_bias = nn.Parameter(torch.full((HEADS,), GATE_BIAS_INIT))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        projected = self.qkv(hidden).view(batch, length, 3, HEADS, HEAD_DIM)
        projected = projected.permute(2, 0, 3, 1, 4)
        query, key, value = projected[0], projected[1], projected[2]
        if self.kind == "linear":
            mixed = linear_attention_prefix(query, key, value)
        elif self.kind == "gated":
            mixed = gated_linear_attention(query, key, value, self.gate_weight, self.gate_bias)
        elif self.kind == "window":
            mixed = dense_causal_attention(query, key, value, window_keep(length, device=hidden.device))
        elif self.kind == "mixed":
            mixed = dense_causal_attention(
                query, key, value, mixed_local_dilated_keep(length, device=hidden.device)
            )
        else:
            mixed = dense_causal_attention(query, key, value)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, WIDTH)
        return self.proj(mixed)


class Block(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(WIDTH)
        self.attn = Attention(kind)
        self.norm_2 = nn.LayerNorm(WIDTH)
        self.mlp_up = nn.Linear(WIDTH, MLP_WIDTH, bias=False)
        self.mlp_down = nn.Linear(MLP_WIDTH, WIDTH, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm_1(hidden))
        hidden = hidden + self.mlp_down(F.gelu(self.mlp_up(self.norm_2(hidden))))
        return hidden


class MQARModel(nn.Module):
    def __init__(self, kind: str = "dense") -> None:
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.blocks = nn.ModuleList(Block(kind) for _ in range(LAYERS))
        self.final_norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB, bias=False)

    def set_kind(self, kind: str) -> None:
        """Swap the operator without touching projections. Gated weights are not created here."""
        if kind == "gated" and self.kind != "gated":
            raise ValueError("a gated model has extra parameters; construct it directly")
        if self.kind == "gated" and kind != "gated":
            raise ValueError("dropping the forget gate is a different model")
        self.kind = kind
        for block in self.blocks:
            block.attn.kind = kind

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = sinusoidal_positions(length, WIDTH, tokens.device).to(dtype=self.embed.weight.dtype)
        hidden = self.embed(tokens) + positions.unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.final_norm(hidden))


def query_cross_entropy(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy at query positions only. Other positions do not enter the loss.

    ``query_positions`` and ``answer_ids`` have shape ``(B, K)``. The label is
    the value id, not the key token that sits at the query position.
    """
    batch, _, vocab = logits.shape
    if query_positions.shape != answer_ids.shape:
        raise ValueError("query positions and answer ids must share a shape")
    batch_index = torch.arange(batch, device=logits.device).unsqueeze(1)
    chosen = logits[batch_index, query_positions]
    return F.cross_entropy(chosen.reshape(-1, vocab), answer_ids.reshape(-1))


def query_predictions(logits: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    batch = logits.shape[0]
    batch_index = torch.arange(batch, device=logits.device).unsqueeze(1)
    return logits[batch_index, query_positions].argmax(dim=-1)
