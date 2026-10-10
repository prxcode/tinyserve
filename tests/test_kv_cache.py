import pytest
import torch

from tinyserve.block_manager import BlockManager
from tinyserve.kv_cache import KVCache
from tinyserve.model.config import LlamaConfig

BLOCK_SIZE = 4


def padded_tables(manager: BlockManager, seq_ids: list[int]) -> torch.Tensor:
    tables = [manager.block_table(seq_id) for seq_id in seq_ids]
    width = max(len(table) for table in tables)
    return torch.tensor([table + [0] * (width - len(table)) for table in tables])


@pytest.fixture
def cache(tiny_config: LlamaConfig) -> KVCache:
    return KVCache(
        tiny_config, num_blocks=16, block_size=BLOCK_SIZE, dtype=torch.float32, device="cpu"
    )


def test_context_slots_match_block_manager(cache: KVCache) -> None:
    manager = BlockManager(num_blocks=16, block_size=BLOCK_SIZE)
    lengths = {0: 9, 1: 3, 2: 6}
    # Grow sequences in turns so their blocks interleave.
    for target in range(1, 10):
        for seq_id, length in lengths.items():
            manager.reserve(seq_id, min(target, length))

    slots = cache.context_slots(padded_tables(manager, [0, 1, 2]), context_len=9)

    for row, (seq_id, length) in enumerate(lengths.items()):
        assert slots[row, :length].tolist() == manager.slots(seq_id, 0, length)


def test_written_tokens_read_back_per_sequence(tiny_config: LlamaConfig, cache: KVCache) -> None:
    manager = BlockManager(num_blocks=16, block_size=BLOCK_SIZE)
    lengths = {0: 7, 1: 5}
    heads, dim = tiny_config.num_kv_heads, tiny_config.head_dim
    written = {}
    for seq_id, length in lengths.items():
        manager.reserve(seq_id, length)
        keys, values = torch.randn(length, heads, dim), torch.randn(length, heads, dim)
        slots = torch.tensor(manager.slots(seq_id, 0, length))
        cache.write(layer=1, slots=slots, keys=keys, values=values)
        written[seq_id] = (keys, values)

    slots = cache.context_slots(padded_tables(manager, [0, 1]), context_len=7)
    keys, values = cache.read(layer=1, slots=slots)

    for row, (seq_id, length) in enumerate(lengths.items()):
        torch.testing.assert_close(keys[row, :length], written[seq_id][0])
        torch.testing.assert_close(values[row, :length], written[seq_id][1])
    assert keys.shape == (2, 7, heads, dim)


def test_layers_do_not_share_storage(tiny_config: LlamaConfig, cache: KVCache) -> None:
    slots = torch.tensor([0, 1])
    data = torch.ones(2, tiny_config.num_kv_heads, tiny_config.head_dim)

    cache.write(layer=0, slots=slots, keys=data, values=data)

    assert cache.keys[1].abs().sum() == 0


def test_unused_slots_are_finite(cache: KVCache) -> None:
    assert torch.isfinite(cache.keys).all()
    assert torch.isfinite(cache.values).all()


def test_bytes_per_block_for_tinyllama() -> None:
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=22,
        num_heads=32,
        num_kv_heads=4,
        head_dim=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )

    # keys + values, 22 layers, 4 kv heads of 64 dims, 2 bytes each, 16 tokens per block
    assert KVCache.bytes_per_block(config, 16, torch.float16) == 2 * 22 * 4 * 64 * 2 * 16


def test_allocated_memory_matches_estimate(tiny_config: LlamaConfig, cache: KVCache) -> None:
    allocated = cache.keys.nbytes + cache.values.nbytes

    assert allocated == 16 * KVCache.bytes_per_block(tiny_config, BLOCK_SIZE, torch.float32)


def test_blocks_that_fit_rounds_down(tiny_config: LlamaConfig) -> None:
    per_block = KVCache.bytes_per_block(tiny_config, BLOCK_SIZE, torch.float16)

    assert KVCache.blocks_that_fit(10 * per_block - 1, tiny_config, BLOCK_SIZE, torch.float16) == 9
    assert KVCache.blocks_that_fit(-5, tiny_config, BLOCK_SIZE, torch.float16) == 0
