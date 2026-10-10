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
