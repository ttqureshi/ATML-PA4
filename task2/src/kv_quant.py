"""INT4 and INT2 group quantization for the KV cache.

Affine reconstruction is ``scale * code + offset``. ``offset`` is the group
minimum and ``scale`` is ``(max - min) / (2^b - 1)``. A constant group has
scale 0 and code 0, so the zero-point is not a second stored integer: code 0
maps to ``offset``. Rounding is PyTorch round-half-to-even. Codes are packed
little-nibble first (the earliest element sits in the low bits).

Channel-wise groups run along the token axis, one feature channel at a time.
Token-wise groups run along the feature axis, one token at a time. In both
layouts the trailing ``L mod 64`` tokens stay FP16 for K and for V.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from task2.src.kv_budget import GROUP, memory_budget


def quantize_groups(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the last dimension. ``values`` is ``(..., group_size)``."""
    if bits not in (2, 4):
        raise ValueError("bits must be 2 or 4")
    qmax = (1 << bits) - 1
    low = values.amin(dim=-1, keepdim=True)
    high = values.amax(dim=-1, keepdim=True)
    span = high - low
    scale = span / qmax
    safe_scale = torch.where(span > 0, scale, torch.ones_like(scale))
    codes = torch.round((values - low) / safe_scale).clamp(0, qmax)
    codes = torch.where(span > 0, codes, torch.zeros_like(codes))
    stored_scale = torch.where(span > 0, scale, torch.zeros_like(scale)).squeeze(-1)
    return codes.to(torch.int64), stored_scale, low.squeeze(-1)


def dequantize_groups(codes: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return codes.to(dtype=scale.dtype) * scale.unsqueeze(-1) + offset.unsqueeze(-1)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    per_byte = 8 // bits
    flat = codes.to(torch.int64).reshape(-1)
    if flat.numel() % per_byte != 0:
        raise ValueError("code count must fill complete bytes")
    grouped = flat.reshape(-1, per_byte)
    packed = torch.zeros(grouped.shape[0], dtype=torch.int64, device=codes.device)
    mask = (1 << bits) - 1
    for slot in range(per_byte):
        packed = packed | ((grouped[:, slot] & mask) << (slot * bits))
    return packed.to(torch.uint8)


def unpack_codes(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    values = packed.to(torch.int64)
    slots = [(values >> (slot * bits)) & mask for slot in range(per_byte)]
    return torch.stack(slots, dim=-1).reshape(-1)[:count]


@dataclass
class PackedTensor:
    """One quantized tensor plus its FP16 tail. ``lead`` does not include L or D."""

    bits: int
    layout: str
    length: int
    channels: int
    complete: int
    lead_shape: tuple[int, ...]
    codes_shape: tuple[int, ...]
    scale: torch.Tensor
    offset: torch.Tensor
    payload: torch.Tensor
    tail: torch.Tensor

    def serialized_bytes(self) -> int:
        return int(self.scale.numel() * 2 + self.offset.numel() * 2 + self.payload.numel() + self.tail.numel() * 2)

    def padding_bytes(self) -> int:
        return 0


def _pack_grouped(grouped: torch.Tensor, bits: int, layout: str, length: int, channels: int, complete: int, lead: tuple[int, ...], tail: torch.Tensor) -> PackedTensor:
    codes, scale, offset = quantize_groups(grouped.float(), bits)
    scale_h = scale.detach().to(torch.float16).cpu()
    offset_h = offset.detach().to(torch.float16).cpu()
    payload = pack_codes(codes.cpu(), bits)
    return PackedTensor(
        bits=bits,
        layout=layout,
        length=length,
        channels=channels,
        complete=complete,
        lead_shape=lead,
        codes_shape=tuple(codes.shape),
        scale=scale_h,
        offset=offset_h,
        payload=payload.cpu(),
        tail=tail.detach().to(torch.float16).cpu(),
    )


def pack_channelwise(tensor: torch.Tensor, bits: int) -> PackedTensor:
    """Group along tokens for each channel. ``tensor`` is ``(..., L, D)``."""
    *lead, length, channels = tensor.shape
    complete = (length // GROUP) * GROUP
    tail = tensor[..., complete:, :].contiguous()
    body = tensor[..., :complete, :].transpose(-1, -2).contiguous()
    groups = complete // GROUP
    grouped = body.reshape(*lead, channels, groups, GROUP) if groups else body.reshape(*lead, channels, 0, GROUP)
    return _pack_grouped(grouped, bits, "channel", length, channels, complete, tuple(lead), tail)


def pack_tokenwise(tensor: torch.Tensor, bits: int) -> PackedTensor:
    """Group along channels for each token. Requires ``D`` divisible by 64."""
    *lead, length, channels = tensor.shape
    if channels % GROUP != 0:
        raise ValueError("token-wise groups need a head dimension divisible by 64")
    complete = (length // GROUP) * GROUP
    tail = tensor[..., complete:, :].contiguous()
    body = tensor[..., :complete, :]
    per_token = channels // GROUP
    grouped = body.reshape(*lead, complete, per_token, GROUP)
    return _pack_grouped(grouped, bits, "token", length, channels, complete, tuple(lead), tail)


def unpack_tensor(packed: PackedTensor) -> torch.Tensor:
    """Dequantize to FP32 with the stored FP16 scale and offset, then append the tail."""
    count = int(torch.tensor(packed.codes_shape).prod().item()) if packed.codes_shape else 0
    if count == 0:
        body = torch.zeros(*packed.lead_shape, 0, packed.channels, dtype=torch.float32)
    else:
        codes = unpack_codes(packed.payload, count, packed.bits).reshape(packed.codes_shape)
        values = dequantize_groups(codes.float(), packed.scale.float(), packed.offset.float())
        if packed.layout == "channel":
            *lead, channels, groups, group = values.shape
            body = values.reshape(*lead, channels, groups * group).transpose(-1, -2).contiguous()
        elif packed.layout == "token":
            *lead, complete, per_token, group = values.shape
            body = values.reshape(*lead, complete, per_token * group)
        else:
            raise ValueError(f"unknown layout {packed.layout}")
    tail = packed.tail.float()
    if tail.numel() == 0:
        return body
    return torch.cat((body, tail), dim=-2)


def pack_kv(keys: torch.Tensor, values: torch.Tensor, bits: int, key_layout: str, value_layout: str) -> tuple[PackedTensor, PackedTensor]:
    packers = {"channel": pack_channelwise, "token": pack_tokenwise}
    return packers[key_layout](keys, bits), packers[value_layout](values, bits)


def serialized_pair_bytes(keys: PackedTensor, values: PackedTensor) -> int:
    return keys.serialized_bytes() + values.serialized_bytes()


def assert_int4_channel_token_matches_budget(length: int, n_layers: int, h_kv: int, d_h: int) -> int:
    """Actual packed bytes of channel-wise K and token-wise V at INT4 equal ``M_b``."""
    if d_h % GROUP != 0:
        raise ValueError("the closed form matches token-wise V when d_h is a multiple of 64")
    keys = torch.randn(n_layers, h_kv, length, d_h)
    values = torch.randn(n_layers, h_kv, length, d_h)
    packed_k, packed_v = pack_kv(keys, values, bits=4, key_layout="channel", value_layout="token")
    total = serialized_pair_bytes(packed_k, packed_v)
    expected = memory_budget(length, n_layers, h_kv, d_h, bits=4)
    if total != expected:
        raise AssertionError(f"packed bytes {total} != M_b {expected}")
    return total
