class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """Hands out ids of fixed-size KV cache blocks. Pure bookkeeping, no GPU memory is touched."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        # Used as a stack, so a just-freed block is handed out next while it is still in cache.
        self._free = list(range(num_blocks - 1, -1, -1))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError("no free KV cache blocks")
        block_id = self._free.pop()
        self._allocated.add(block_id)
        return block_id

    def free(self, block_id: int) -> None:
        if block_id not in self._allocated:
            raise ValueError(f"block {block_id} is not allocated")
        self._allocated.remove(block_id)
        self._free.append(block_id)


class BlockManager:
    """Maps each sequence to the cache blocks holding its tokens, in token order.

    Token `t` of a sequence lives in block `table[t // block_size]` at offset `t % block_size`.
    A slot is the flat index `block * block_size + offset` into the cache tensors.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._allocator = BlockAllocator(num_blocks)
        self._tables: dict[int, list[int]] = {}

    @property
    def num_blocks(self) -> int:
        return self._allocator.num_blocks

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def blocks_needed(self, seq_id: int, num_tokens: int) -> int:
        """Extra blocks the sequence needs to hold `num_tokens` tokens in total."""
        held = len(self._tables.get(seq_id, ()))
        return max(0, self.blocks_for(num_tokens) - held)

    def can_reserve(self, seq_id: int, num_tokens: int) -> bool:
        return self.blocks_needed(seq_id, num_tokens) <= self.num_free_blocks

    def reserve(self, seq_id: int, num_tokens: int) -> None:
        """Grow the sequence's table to hold `num_tokens` tokens. Allocates all or nothing."""
        needed = self.blocks_needed(seq_id, num_tokens)
        if needed > self.num_free_blocks:
            raise OutOfBlocksError(
                f"sequence {seq_id} needs {needed} blocks, {self.num_free_blocks} free"
            )
        table = self._tables.setdefault(seq_id, [])
        table.extend(self._allocator.allocate() for _ in range(needed))

    def free(self, seq_id: int) -> None:
        if seq_id not in self._tables:
            raise ValueError(f"sequence {seq_id} holds no blocks")
        for block_id in self._tables.pop(seq_id):
            self._allocator.free(block_id)

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._tables[seq_id])

    def slots(self, seq_id: int, start: int, end: int) -> list[int]:
        """Cache slots for token positions [start, end) of a sequence."""
        table = self._tables[seq_id]
        if end > len(table) * self.block_size:
            raise ValueError(f"position {end - 1} of sequence {seq_id} has no reserved block")
        size = self.block_size
        return [table[pos // size] * size + pos % size for pos in range(start, end)]
