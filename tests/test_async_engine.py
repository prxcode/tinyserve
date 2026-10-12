import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import aclosing

import pytest
import torch

from tinyserve.async_engine import AsyncEngine, EngineStoppedError, OverloadedError, TokenEvent
from tinyserve.engine import Engine, EngineConfig
from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.sequence import FinishReason, SamplingParams

CONFIG = EngineConfig(
    num_blocks=64, block_size=4, max_num_seqs=8, max_prefill_tokens=64, max_model_len=64
)


@pytest.fixture
def model(tiny_config: LlamaConfig) -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_config).double().eval()


@pytest.fixture
def async_engine(model: LlamaForCausalLM) -> Iterator[AsyncEngine]:
    engine = AsyncEngine(Engine(model, CONFIG), max_pending=16)
    engine.start()
    yield engine
    engine.stop()


def prompt(i: int) -> list[int]:
    return [3 + i, 10 + 2 * i, 40, 7 + i][: 1 + i % 4]


async def collect(stream: AsyncIterator[TokenEvent]) -> tuple[list[int], FinishReason | None]:
    tokens: list[int] = []
    reason = None
    async for event in stream:
        if event.token_id is not None:
            tokens.append(event.token_id)
        reason = event.finish_reason
    return tokens, reason


def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.01)


def test_concurrent_requests_match_the_synchronous_engine(
    model: LlamaForCausalLM, async_engine: AsyncEngine
) -> None:
    params = SamplingParams(max_tokens=12)
    expected = [
        s.output_ids for s in Engine(model, CONFIG).generate([prompt(i) for i in range(10)], params)
    ]

    async def main() -> list[tuple[list[int], FinishReason | None]]:
        streams = [async_engine.submit(prompt(i), params) for i in range(10)]
        return await asyncio.gather(*(collect(stream) for stream in streams))

    results = asyncio.run(main())

    assert [tokens for tokens, _ in results] == expected
    assert async_engine.num_pending == 0


def test_closing_a_stream_aborts_the_request(async_engine: AsyncEngine) -> None:
    blocks = async_engine.engine.scheduler.blocks

    async def main() -> None:
        stream = async_engine.submit(prompt(1), SamplingParams(max_tokens=50))
        async with aclosing(stream):
            async for _ in stream:
                break  # the client goes away after the first token

    asyncio.run(main())

    wait_until(lambda: async_engine.num_pending == 0)
    wait_until(lambda: blocks.num_free_blocks == blocks.num_blocks)
    assert not async_engine.engine.has_unfinished()


def test_rejects_requests_beyond_max_pending(model: LlamaForCausalLM) -> None:
    async_engine = AsyncEngine(Engine(model, CONFIG), max_pending=2)
    async_engine.start()

    async def main() -> None:
        first = async_engine.submit(prompt(0), SamplingParams(max_tokens=20))
        second = async_engine.submit(prompt(1), SamplingParams(max_tokens=20))
        with pytest.raises(OverloadedError):
            async_engine.submit(prompt(2), SamplingParams(max_tokens=20))
        await collect(first)
        await collect(second)
        # Capacity is back once earlier requests finish.
        await collect(async_engine.submit(prompt(2), SamplingParams(max_tokens=2)))

    try:
        asyncio.run(main())
    finally:
        async_engine.stop()


def test_invalid_request_fails_before_queueing(async_engine: AsyncEngine) -> None:
    async def main() -> None:
        with pytest.raises(ValueError):
            async_engine.submit([], SamplingParams())
        with pytest.raises(ValueError):
            async_engine.submit(prompt(0), SamplingParams(max_tokens=1000))

    asyncio.run(main())

    assert async_engine.num_pending == 0


def test_failed_batch_reports_error_and_engine_recovers(
    async_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = async_engine.engine.model.model
    original = inner.forward

    def fail_once(*args: object, **kwargs: object) -> torch.Tensor:
        monkeypatch.setattr(inner, "forward", original)
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(inner, "forward", fail_once)

    async def main() -> None:
        tokens, reason = await collect(async_engine.submit(prompt(0), SamplingParams(max_tokens=8)))
        assert (tokens, reason) == ([], FinishReason.ERROR)
        assert async_engine.consecutive_failures == 1

        tokens, reason = await collect(async_engine.submit(prompt(0), SamplingParams(max_tokens=3)))
        assert len(tokens) == 3 and reason == FinishReason.LENGTH
        assert async_engine.consecutive_failures == 0

    asyncio.run(main())


def test_engine_thread_crash_fails_requests_instead_of_hanging(
    async_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_schedule() -> None:
        raise AssertionError("scheduler bug")

    monkeypatch.setattr(async_engine.engine.scheduler, "schedule", broken_schedule)

    async def main() -> None:
        tokens, reason = await asyncio.wait_for(
            collect(async_engine.submit(prompt(0), SamplingParams(max_tokens=8))), timeout=10
        )
        assert (tokens, reason) == ([], FinishReason.ERROR)

    asyncio.run(main())

    wait_until(lambda: not async_engine.is_alive)
    assert async_engine.num_pending == 0

    async def submit_after_crash() -> None:
        with pytest.raises(EngineStoppedError):
            async_engine.submit(prompt(0), SamplingParams(max_tokens=8))

    asyncio.run(submit_after_crash())


def test_stop_ends_in_flight_streams(model: LlamaForCausalLM) -> None:
    async_engine = AsyncEngine(Engine(model, CONFIG))
    async_engine.start()

    async def main() -> FinishReason | None:
        stream = async_engine.submit(prompt(2), SamplingParams(max_tokens=60))
        await anext(stream)
        await asyncio.to_thread(async_engine.stop)
        _, reason = await collect(stream)
        return reason

    assert asyncio.run(main()) == FinishReason.ABORT
    assert not async_engine.is_alive
