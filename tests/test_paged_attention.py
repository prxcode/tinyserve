import torch

from tinyserve.block_manager import BlockManager
from tinyserve.kv_cache import KVCache
from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM

BLOCK_SIZE = 4


def paged_step(
    model: LlamaForCausalLM,
    cache: KVCache,
    blocks: BlockManager,
    tokens: list[list[int]],
    num_cached: list[int],
) -> torch.Tensor:
    """Run every sequence's uncached tokens in one padded batch; return last-token logits."""
    chunks = [seq[cached:] for seq, cached in zip(tokens, num_cached, strict=True)]
    width = max(len(chunk) for chunk in chunks)
    input_ids = torch.zeros(len(tokens), width, dtype=torch.long)
    positions = torch.zeros(len(tokens), width, dtype=torch.long)
    valid = torch.zeros(len(tokens), width, dtype=torch.bool)
    for row, (chunk, cached) in enumerate(zip(chunks, num_cached, strict=True)):
        input_ids[row, : len(chunk)] = torch.tensor(chunk)
        positions[row, : len(chunk)] = torch.arange(cached, cached + len(chunk))
        valid[row, : len(chunk)] = True

    tables = [blocks.block_table(seq_id) for seq_id in range(len(tokens))]
    max_blocks = max(len(table) for table in tables)
    block_tables = torch.tensor([table + [0] * (max_blocks - len(table)) for table in tables])

    hidden = model.model(input_ids, positions, cache.batch_view(positions, valid, block_tables))
    last = torch.tensor([len(chunk) - 1 for chunk in chunks])
    return model.lm_head(hidden[torch.arange(len(tokens)), last])


def test_paged_prefill_and_decode_match_full_recompute(tiny_config: LlamaConfig) -> None:
    torch.manual_seed(0)
    # float64 so any mismatch is a real bug rather than rounding from a different op order.
    model = LlamaForCausalLM(tiny_config).double().eval()
    cache = KVCache(
        tiny_config, num_blocks=32, block_size=BLOCK_SIZE, dtype=torch.float64, device="cpu"
    )
    blocks = BlockManager(num_blocks=32, block_size=BLOCK_SIZE)
    tokens = [torch.randint(0, tiny_config.vocab_size, (n,)).tolist() for n in (7, 3, 10)]
    num_cached = [0, 0, 0]

    with torch.no_grad():
        for _ in range(9):
            for seq_id, seq in enumerate(tokens):
                blocks.reserve(seq_id, len(seq))
            logits = paged_step(model, cache, blocks, tokens, num_cached)

            for seq_id, seq in enumerate(tokens):
                expected = model(torch.tensor([seq]))[0, -1]
                torch.testing.assert_close(logits[seq_id], expected)

            num_cached = [len(seq) for seq in tokens]
            for seq, next_id in zip(tokens, logits.argmax(dim=-1).tolist(), strict=True):
                seq.append(next_id)

    # Growing three sequences in turn must have interleaved their blocks.
    tables = [blocks.block_table(seq_id) for seq_id in range(3)]
    assert any(table != list(range(table[0], table[0] + len(table))) for table in tables)


def test_padding_tokens_do_not_touch_the_cache(tiny_config: LlamaConfig) -> None:
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config).double().eval()
    cache = KVCache(
        tiny_config, num_blocks=8, block_size=BLOCK_SIZE, dtype=torch.float64, device="cpu"
    )
    blocks = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    tokens = [[5] * 6, [9]]
    for seq_id, seq in enumerate(tokens):
        blocks.reserve(seq_id, len(seq))

    with torch.no_grad():
        paged_step(model, cache, blocks, tokens, [0, 0])

    short_block = blocks.block_table(1)[0]
    stored = cache.keys[:, short_block * BLOCK_SIZE : (short_block + 1) * BLOCK_SIZE]
    # Only the first slot of the short sequence's block was written; padding left the rest zero.
    assert stored[:, 0].abs().sum() > 0
    assert stored[:, 1:].abs().sum() == 0
