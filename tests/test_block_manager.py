import pytest

from tinyserve.block_manager import BlockAllocator, OutOfBlocksError


def test_allocates_every_block_once() -> None:
    allocator = BlockAllocator(4)

    blocks = [allocator.allocate() for _ in range(4)]

    assert sorted(blocks) == [0, 1, 2, 3]
    assert allocator.num_free == 0


def test_raises_when_exhausted() -> None:
    allocator = BlockAllocator(2)
    allocator.allocate()
    allocator.allocate()

    with pytest.raises(OutOfBlocksError):
        allocator.allocate()


def test_freed_block_is_reused_first() -> None:
    allocator = BlockAllocator(4)
    first = allocator.allocate()
    allocator.allocate()

    allocator.free(first)

    assert allocator.allocate() == first


@pytest.mark.parametrize("block_id", [0, 3, 99])
def test_rejects_freeing_unallocated_block(block_id: int) -> None:
    allocator = BlockAllocator(4)

    with pytest.raises(ValueError):
        allocator.free(block_id)


def test_rejects_double_free() -> None:
    allocator = BlockAllocator(4)
    block = allocator.allocate()
    allocator.free(block)

    with pytest.raises(ValueError):
        allocator.free(block)
    assert allocator.num_free == 4
