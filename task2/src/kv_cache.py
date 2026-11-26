"""KV-cache editor that keeps each retained token's original rotary position.

Hugging Face builds the causal mask from ``cache_position`` and the physical
cache length, and it builds RoPE from ``position_ids``. After eviction those
are different integers. The editor stores the absolute ids of whatever is
physically present and feeds the next generated token its original absolute
id (the uncompacted prompt length plus the number of generated tokens), while
``cache_position`` stays the physical append index. Keys already in the cache
keep the RoPE that was applied at those original ids; they are not rotated again
and they are not renumbered to ``0 .. P-1``.
"""

from __future__ import annotations

import torch


class RotaryCacheEditor:
    def __init__(self, cache, prompt_length: int) -> None:
        self.cache = cache
        length = int(cache.get_seq_length())
        if length != prompt_length:
            raise ValueError("prompt_length must match the cache length before editing")
        n_layers = len(cache.layers)
        self.position_ids = [torch.arange(length, dtype=torch.long) for _ in range(n_layers)]
        self.next_absolute = prompt_length
        self.prompt_length = prompt_length

    def physical_length(self) -> int:
        return int(self.cache.get_seq_length())

    def retain(self, indices_per_layer: list[list[int]]) -> None:
        """Keep ``indices`` into the current cache order. Ids are the positions stored now.

        The cache must still be in original prompt order, so an index is the
        absolute rotary id. Indices are sorted so the physical order stays causal.
        Every layer keeps the same count.
        """
        if len(indices_per_layer) != len(self.position_ids):
            raise ValueError("need one index set per layer")
        for layer_idx, indices in enumerate(indices_per_layer):
            current_ids = self.position_ids[layer_idx]
            ordered = sorted(set(int(i) for i in indices))
            if any(i < 0 or i >= int(current_ids.numel()) for i in ordered):
                raise IndexError("retention index is outside the current cache")
            # Map from the caller's index (into the original order at construction)
            # only when the cache has not yet been compacted. Callers pass original ids.
            id_to_physical = {int(absolute): physical for physical, absolute in enumerate(current_ids.tolist())}
            physical = [id_to_physical[absolute] for absolute in ordered]
            index = torch.tensor(physical, dtype=torch.long, device=self.cache.layers[layer_idx].keys.device)
            layer = self.cache.layers[layer_idx]
            layer.keys = layer.keys.index_select(-2, index)
            layer.values = layer.values.index_select(-2, index)
            self.position_ids[layer_idx] = torch.tensor(ordered, dtype=torch.long)
        lengths = {int(row.numel()) for row in self.position_ids}
        if len(lengths) != 1:
            raise ValueError("layers must retain the same count so one physical cache length is well defined")

    def decode_inputs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Absolute rotary id for the next token, and the physical append index."""
        position_ids = torch.tensor([[self.next_absolute]], dtype=torch.long, device=device)
        cache_position = torch.tensor([self.physical_length()], dtype=torch.long, device=device)
        return position_ids, cache_position

    def note_appended(self) -> int:
        """Record the absolute id of the token the model just wrote at the physical tail."""
        absolute = self.next_absolute
        extra = torch.tensor([absolute], dtype=torch.long)
        for layer_idx in range(len(self.position_ids)):
            self.position_ids[layer_idx] = torch.cat((self.position_ids[layer_idx], extra))
        self.next_absolute = absolute + 1
        return absolute

    def clone(self) -> "RotaryCacheEditor":
        """Copy tensors and absolute ids. The clone does not renumber positions."""
        from transformers.cache_utils import DynamicCache

        copied = DynamicCache()
        for layer_idx, layer in enumerate(self.cache.layers):
            copied.update(layer.keys.clone(), layer.values.clone(), layer_idx)
        twin = RotaryCacheEditor.__new__(RotaryCacheEditor)
        twin.cache = copied
        twin.position_ids = [row.clone() for row in self.position_ids]
        twin.next_absolute = self.next_absolute
        twin.prompt_length = self.prompt_length
        return twin

    def drop_last(self) -> int:
        """Remove the physical tail token and return the absolute id it carried."""
        absolute = int(self.position_ids[0][-1].item())
        for layer_idx, layer in enumerate(self.cache.layers):
            layer.keys = layer.keys[..., :-1, :]
            layer.values = layer.values[..., :-1, :]
            self.position_ids[layer_idx] = self.position_ids[layer_idx][:-1]
        self.next_absolute = absolute
        return absolute
