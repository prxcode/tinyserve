import itertools
import random

import pytest

from tinyserve.block_manager import BlockManager
from tinyserve.scheduler import ScheduledBatch, Scheduler
from tinyserve.sequence import FinishReason, SamplingParams, Sequence, Status

_ids = itertools.count()


def make_seq(prompt_len: int, max_tokens: int = 4) -> Sequence:
    return Sequence(next(_ids), [7] * prompt_len, SamplingParams(max_tokens=max_tokens))


def make_scheduler(
    num_blocks: int = 64,
    block_size: int = 4,
    max_num_seqs: int = 8,
    max_prefill_tokens: int = 64,
    max_model_len: int = 64,
) -> Scheduler:
    blocks = BlockManager(num_blocks, block_size)
    return Scheduler(blocks, max_num_seqs, max_prefill_tokens, max_model_len)


def run_step(scheduler: Scheduler) -> ScheduledBatch | None:
    """Stand-in for the engine: every scheduled sequence produces one token."""
    batch = scheduler.schedule()
    if batch is None:
        return None
    for seq in batch.sequences:
        assert len(scheduler.blocks.block_table(seq.seq_id)) >= scheduler.blocks.blocks_for(
            seq.length
        ), "scheduled without cache space for its tokens"
        seq.num_cached = seq.length
        seq.output_ids.append(0)
        if len(seq.output_ids) == seq.params.max_tokens:
            scheduler.finish(seq, FinishReason.LENGTH)
    return batch


def test_prefills_new_requests_before_decoding() -> None:
    scheduler = make_scheduler()
    first, second = make_seq(5), make_seq(3)
    scheduler.add(first)
    scheduler.add(second)

    batch = run_step(scheduler)
    assert batch is not None and batch.is_prefill
    assert batch.sequences == [first, second]

    third = make_seq(2)
    scheduler.add(third)
    batch = run_step(scheduler)
    assert batch is not None and batch.is_prefill and batch.sequences == [third]

    batch = run_step(scheduler)
    assert batch is not None and not batch.is_prefill
    assert batch.sequences == [first, second, third]


def test_respects_max_num_seqs() -> None:
    scheduler = make_scheduler(max_num_seqs=2)
    seqs = [make_seq(2) for _ in range(3)]
    for seq in seqs:
        scheduler.add(seq)

    batch = run_step(scheduler)

    assert batch is not None and batch.sequences == seqs[:2]
    assert list(scheduler.waiting) == seqs[2:]


def test_prefill_budget_counts_padding() -> None:
    scheduler = make_scheduler(max_prefill_tokens=64, max_model_len=64)
    long, short_a, short_b = make_seq(40), make_seq(10), make_seq(10)
    for seq in (long, short_a, short_b):
        scheduler.add(seq)

    # Padding short_a to 40 tokens would make 80 padded tokens, over the budget of 64.
    batch = run_step(scheduler)
    assert batch is not None and batch.sequences == [long]

    batch = run_step(scheduler)
    assert batch is not None and batch.sequences == [short_a, short_b]


def test_waiting_queue_is_strictly_first_come_first_served() -> None:
    scheduler = make_scheduler(num_blocks=6, block_size=4)
    running = make_seq(8)
    scheduler.add(running)
    run_step(scheduler)

    big, small = make_seq(16), make_seq(2)
    scheduler.add(big)
    scheduler.add(small)
    batch = run_step(scheduler)

    # `small` would fit, but admitting it ahead of `big` could starve long prompts.
    assert batch is not None and not batch.is_prefill
    assert list(scheduler.waiting) == [big, small]


def test_admission_leaves_a_decode_block_per_running_sequence() -> None:
    scheduler = make_scheduler(num_blocks=4, block_size=4)
    first = make_seq(4)
    scheduler.add(first)
    run_step(scheduler)

    # 3 blocks are free. `second` needs 3, but `first` needs one for its next token.
    second = make_seq(12)
    scheduler.add(second)
    batch = run_step(scheduler)

    assert batch is not None and not batch.is_prefill
    assert second.status is Status.WAITING


def test_out_of_blocks_preempts_newest_and_recomputes_it_later() -> None:
    scheduler = make_scheduler(num_blocks=4, block_size=4)
    older, newer = make_seq(4, max_tokens=8), make_seq(4, max_tokens=8)
    scheduler.add(older)
    scheduler.add(newer)

    batches = []
    while scheduler.has_unfinished():
        batch = run_step(scheduler)
        assert batch is not None
        batches.append(batch)

    assert scheduler.num_preemptions >= 1
    recompute = next(b for b in batches[1:] if b.is_prefill)
    assert recompute.sequences == [newer]
    assert len(older.output_ids) == len(newer.output_ids) == 8
    assert scheduler.blocks.num_free_blocks == scheduler.blocks.num_blocks


def test_preempted_sequence_resets_its_cache_state() -> None:
    scheduler = make_scheduler(num_blocks=4, block_size=4)
    older, newer = make_seq(4, max_tokens=8), make_seq(4, max_tokens=8)
    scheduler.add(older)
    scheduler.add(newer)

    while scheduler.num_preemptions == 0:
        run_step(scheduler)

    assert newer.status is Status.WAITING and newer.num_cached == 0
    assert scheduler.waiting[0] is newer
    assert newer.uncached_token_ids() == newer.prompt_ids + newer.output_ids


def test_abort_frees_blocks_of_running_and_waiting_sequences() -> None:
    scheduler = make_scheduler(max_num_seqs=1)
    running, waiting = make_seq(6), make_seq(6)
    scheduler.add(running)
    scheduler.add(waiting)
    run_step(scheduler)

    scheduler.abort(running)
    scheduler.abort(waiting)

    assert running.finish_reason == waiting.finish_reason == FinishReason.ABORT
    assert not scheduler.has_unfinished()
    assert scheduler.blocks.num_free_blocks == scheduler.blocks.num_blocks


def test_abort_after_finish_is_a_no_op() -> None:
    scheduler = make_scheduler()
    seq = make_seq(2, max_tokens=1)
    scheduler.add(seq)
    run_step(scheduler)

    scheduler.abort(seq)

    assert seq.finish_reason == FinishReason.LENGTH


@pytest.mark.parametrize(
    ("prompt_len", "max_tokens", "num_blocks"),
    [(60, 10, 64), (30, 10, 4)],
    ids=["longer-than-model", "larger-than-cache"],
)
def test_rejects_requests_that_can_never_run(
    prompt_len: int, max_tokens: int, num_blocks: int
) -> None:
    scheduler = make_scheduler(num_blocks=num_blocks)

    with pytest.raises(ValueError):
        scheduler.add(make_seq(prompt_len, max_tokens))


def test_rejects_prefill_budget_smaller_than_model_length() -> None:
    with pytest.raises(ValueError):
        make_scheduler(max_prefill_tokens=32, max_model_len=64)


def test_random_workload_completes_without_leaks() -> None:
    rng = random.Random(0)
    scheduler = make_scheduler(num_blocks=24, block_size=4, max_num_seqs=6)
    seqs = []

    for _ in range(10_000):
        if len(seqs) < 60 and rng.random() < 0.3:
            seq = make_seq(rng.randint(1, 30), max_tokens=rng.randint(1, 30))
            scheduler.add(seq)
            seqs.append(seq)
        run_step(scheduler)
        if len(seqs) == 60 and not scheduler.has_unfinished():
            break
    else:
        pytest.fail("workload did not finish")

    assert scheduler.num_preemptions > 0, "cache never ran out, so preemption went untested"
    assert all(len(seq.output_ids) == seq.params.max_tokens for seq in seqs)
    assert scheduler.blocks.num_free_blocks == scheduler.blocks.num_blocks
