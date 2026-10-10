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
