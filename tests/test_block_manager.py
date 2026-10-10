import random

import pytest

from tinyserve.block_manager import BlockAllocator, BlockManager, OutOfBlocksError


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


def test_blocks_are_reserved_only_when_crossing_a_boundary() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)

    manager.reserve(seq_id=0, num_tokens=4)
    assert len(manager.block_table(0)) == 1

    manager.reserve(seq_id=0, num_tokens=5)
    assert len(manager.block_table(0)) == 2

    manager.reserve(seq_id=0, num_tokens=8)
    assert len(manager.block_table(0)) == 2
    assert manager.num_free_blocks == 6


def test_failed_reserve_allocates_nothing() -> None:
    manager = BlockManager(num_blocks=3, block_size=4)
    manager.reserve(seq_id=0, num_tokens=4)

    with pytest.raises(OutOfBlocksError):
        manager.reserve(seq_id=1, num_tokens=12)

    assert manager.num_free_blocks == 2
    assert manager.blocks_needed(1, 12) == 3
    assert not manager.can_reserve(1, 12)


def test_slots_follow_non_contiguous_blocks() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)
    # Interleaving two sequences gives each of them non-adjacent blocks.
    manager.reserve(seq_id=0, num_tokens=4)
    manager.reserve(seq_id=1, num_tokens=4)
    manager.reserve(seq_id=0, num_tokens=6)

    first, second = manager.block_table(0)
    assert second != first + 1
    assert manager.slots(0, 0, 6) == [
        first * 4,
        first * 4 + 1,
        first * 4 + 2,
        first * 4 + 3,
        second * 4,
        second * 4 + 1,
    ]


def test_slots_beyond_reservation_are_rejected() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)
    manager.reserve(seq_id=0, num_tokens=3)

    assert len(manager.slots(0, 0, 4)) == 4  # the rest of the reserved block is usable
    with pytest.raises(ValueError):
        manager.slots(0, 0, 5)


def test_free_returns_all_blocks() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)
    manager.reserve(seq_id=0, num_tokens=10)
    manager.reserve(seq_id=1, num_tokens=3)

    manager.free(0)

    assert manager.num_free_blocks == 7
    with pytest.raises(ValueError):
        manager.free(0)


def test_random_operations_keep_blocks_consistent() -> None:
    rng = random.Random(0)
    manager = BlockManager(num_blocks=32, block_size=4)
    lengths: dict[int, int] = {}
    exhausted = 0

    for step in range(2000):
        if lengths and rng.random() < 0.2:
            seq_id = rng.choice(list(lengths))
            manager.free(seq_id)
            del lengths[seq_id]
        else:
            seq_id = rng.choice([*lengths, step])
            target = lengths.get(seq_id, 0) + rng.randint(1, 6)
            if manager.can_reserve(seq_id, target):
                manager.reserve(seq_id, target)
                lengths[seq_id] = target
            else:
                with pytest.raises(OutOfBlocksError):
                    manager.reserve(seq_id, target)
                exhausted += 1

        owned = [block for sid in lengths for block in manager.block_table(sid)]
        assert len(owned) == len(set(owned)), "a block is owned by two sequences"
        assert len(owned) + manager.num_free_blocks == manager.num_blocks
        for sid, length in lengths.items():
            assert len(manager.block_table(sid)) == manager.blocks_for(length)

    assert exhausted > 0, "the cache never filled up, so the failure path went untested"
