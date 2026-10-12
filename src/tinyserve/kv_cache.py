from dataclasses import dataclass

import torch
from torch import Tensor

from tinyserve.model.config import LlamaConfig


class KVCache:
    """Preallocated keys and values for every layer, addressed by flat slot index.

    Slots come from BlockManager: slot = block_id * block_size + offset.
    """

    def __init__(
        self,
        config: LlamaConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (config.num_layers, num_blocks * block_size, config.num_kv_heads, config.head_dim)
        # Zeros rather than empty: padded reads hit unused slots, and NaN garbage there would
        # survive a zero attention weight (0 * NaN = NaN).
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def bytes_per_block(config: LlamaConfig, block_size: int, dtype: torch.dtype) -> int:
        per_token = 2 * config.num_layers * config.num_kv_heads * config.head_dim * dtype.itemsize
        return per_token * block_size

    @classmethod
    def blocks_that_fit(
        cls, budget_bytes: int, config: LlamaConfig, block_size: int, dtype: torch.dtype
    ) -> int:
        return max(0, budget_bytes // cls.bytes_per_block(config, block_size, dtype))

    def write(self, layer: int, slots: Tensor, keys: Tensor, values: Tensor) -> None:
        """Store [num_tokens, kv_heads, head_dim] keys and values at the given slots."""
        self.keys[layer].index_copy_(0, slots, keys)
        self.values[layer].index_copy_(0, slots, values)

    def read(self, layer: int, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Gather keys and values for a [batch, context_len] tensor of slots."""
        return self.keys[layer][slots], self.values[layer][slots]

    def context_slots(self, block_tables: Tensor, context_len: int) -> Tensor:
        """Turn [batch, max_blocks] block tables into [batch, context_len] slot indices.

        Tables shorter than max_blocks are padded with block 0. The slots that padding produces
        point at real (stale) data, so callers must mask positions beyond each sequence's length.
        """
        positions = torch.arange(context_len, device=block_tables.device)
        blocks = block_tables[:, positions // self.block_size]
        return blocks * self.block_size + positions % self.block_size

    def batch_view(self, positions: Tensor, valid: Tensor, block_tables: Tensor) -> "PagedBatch":
        """Describe one forward pass over new tokens of several sequences.

        positions: [batch, seq_len] absolute position of each new token (0 on padding).
        valid: [batch, seq_len] False on padding.
        block_tables: [batch, max_blocks] padded with block 0.
        """
        context_len = block_tables.shape[1] * self.block_size
        read_slots = self.context_slots(block_tables, context_len)
        write_slots = read_slots.gather(1, positions)[valid]
        # Causal mask over cached positions. It also hides padded table entries, which always
        # lie beyond the query's own position. Padded queries see key 0 so no row is all -inf.
        key_positions = torch.arange(context_len, device=positions.device)
        mask = (key_positions <= positions.unsqueeze(-1)).unsqueeze(1)
        return PagedBatch(self, valid, write_slots, read_slots, mask)


@dataclass
class PagedBatch:
    """Where one forward pass writes its new keys and values, and what each query attends to."""

    cache: KVCache
    valid: Tensor  # [batch, seq_len] False on padding
    write_slots: Tensor  # [num_valid] slot of each valid new token, row-major order
    read_slots: Tensor  # [batch, context_len] slots each sequence reads
    mask: Tensor  # [batch, 1, seq_len, context_len] True where attention is allowed

    def update(self, layer: int, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """Cache this pass's [batch, kv_heads, seq_len, dim] keys and values, then return
        the full [batch, kv_heads, context_len, dim] context including them."""
        self.cache.write(
            layer,
            self.write_slots,
            keys.transpose(1, 2)[self.valid],
            values.transpose(1, 2)[self.valid],
        )
        context_keys, context_values = self.cache.read(layer, self.read_slots)
        return context_keys.transpose(1, 2), context_values.transpose(1, 2)
