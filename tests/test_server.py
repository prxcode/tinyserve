import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import torch
import uvicorn

from tinyserve.async_engine import AsyncEngine
from tinyserve.engine import Engine, EngineConfig
from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.sequence import FinishReason, SamplingParams
from tinyserve.server import UNHEALTHY_AFTER_FAILURES, create_app

CONFIG = EngineConfig(
    num_blocks=64, block_size=4, max_num_seqs=8, max_prefill_tokens=64, max_model_len=64
)


class AsciiTokenizer:
    """One token per ASCII character, matching the tiny test model's 128-token vocabulary."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@dataclass
class Server:
    url: str
    engine: AsyncEngine


@pytest.fixture
def model(tiny_config: LlamaConfig) -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_config).double().eval()


@pytest.fixture
def server(model: LlamaForCausalLM) -> Iterator[Server]:
    engine = AsyncEngine(Engine(model, CONFIG), max_pending=4)
    app = create_app(engine, AsciiTokenizer())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    uvicorn_server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=uvicorn_server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    wait_until(lambda: uvicorn_server.started)

    yield Server(f"http://127.0.0.1:{sock.getsockname()[1]}", engine)

    uvicorn_server.should_exit = True
    thread.join(timeout=30)


def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.01)


def expected_text(model: LlamaForCausalLM, prompt: str, max_tokens: int) -> str:
    tokenizer = AsciiTokenizer()
    (seq,) = Engine(model, CONFIG).generate(
        [tokenizer.encode(prompt)], SamplingParams(max_tokens=max_tokens)
    )
    return tokenizer.decode(seq.output_ids)


def read_events(response: httpx.Response) -> list[Any]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def block_forward(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Hold every forward pass until the returned event is set."""
    release = threading.Event()
    inner = engine.engine.model.model
    original = inner.forward

    def gated(*args: Any, **kwargs: Any) -> torch.Tensor:
        release.wait()
        return original(*args, **kwargs)

    monkeypatch.setattr(inner, "forward", gated)
    return release


def test_streaming_sends_text_then_finish_then_done(
    server: Server, model: LlamaForCausalLM
) -> None:
    body = {"prompt": "hello", "max_tokens": 10}
    with httpx.stream("POST", f"{server.url}/v1/generate", json=body, timeout=30) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = read_events(response)

    *chunks, final, done = events
    assert done == "[DONE]"
    assert "".join(chunk["text"] for chunk in chunks) == expected_text(model, "hello", 10)
    assert final["finish_reason"] in ("stop", "length")
    assert final["usage"]["prompt_tokens"] == 5


def test_non_streaming_returns_json(server: Server, model: LlamaForCausalLM) -> None:
    body = {"prompt": "hello", "max_tokens": 10, "stream": False}

    response = httpx.post(f"{server.url}/v1/generate", json=body, timeout=30)

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == expected_text(model, "hello", 10)
    assert data["usage"]["completion_tokens"] == len(data["text"])


def test_concurrent_clients_get_their_own_correct_output(
    server: Server, model: LlamaForCausalLM
) -> None:
    prompts = [f"request {i}" for i in range(12)]

    def call(prompt: str) -> str:
        body = {"prompt": prompt, "max_tokens": 12, "stream": False}
        response = httpx.post(f"{server.url}/v1/generate", json=body, timeout=60)
        response.raise_for_status()
        text: str = response.json()["text"]
        return text

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, prompts))

    assert results == [expected_text(model, prompt, 12) for prompt in prompts]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": "hi", "temperature": -1},
        {"prompt": "hi", "top_p": 0},
        {"prompt": "hi", "max_tokens": 0},
    ],
    ids=["missing-prompt", "empty-prompt", "negative-temperature", "zero-top-p", "zero-max-tokens"],
)
def test_malformed_requests_get_422(server: Server, body: dict[str, Any]) -> None:
    assert httpx.post(f"{server.url}/v1/generate", json=body).status_code == 422


@pytest.mark.parametrize(
    "body",
    [{"prompt": "hi", "max_tokens": 1000}, {"prompt": "café 日本"}],
    ids=["longer-than-model", "outside-vocabulary"],
)
def test_requests_that_can_never_run_get_400(server: Server, body: dict[str, Any]) -> None:
    response = httpx.post(f"{server.url}/v1/generate", json=body)

    assert response.status_code == 400
    assert server.engine.num_pending == 0


def test_overload_returns_503_with_retry_after(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = block_forward(server.engine, monkeypatch)
    body = {"prompt": "hi", "max_tokens": 4, "stream": False}

    with ThreadPoolExecutor(max_workers=4) as pool:
        held = [
            pool.submit(httpx.post, f"{server.url}/v1/generate", json=body, timeout=30)
            for _ in range(4)
        ]
        wait_until(lambda: server.engine.num_pending == 4)

        rejected = httpx.post(f"{server.url}/v1/generate", json=body)
        release.set()
        statuses = [future.result().status_code for future in held]

    assert rejected.status_code == 503
    assert rejected.headers["retry-after"] == "1"
    assert statuses == [200] * 4


def test_client_disconnect_aborts_the_request(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = server.engine.engine
    aborted = []
    original_abort = engine.abort
    monkeypatch.setattr(engine, "abort", lambda seq: (aborted.append(seq), original_abort(seq)))
    original_forward = engine.model.model.forward

    def slow_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        time.sleep(0.02)  # keep generation running long enough to disconnect mid-stream
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(engine.model.model, "forward", slow_forward)
    engine.eos_token_id = None
    body = {"prompt": "hi", "max_tokens": 60}

    with httpx.stream("POST", f"{server.url}/v1/generate", json=body, timeout=30) as response:
        next(response.iter_lines())  # read one event, then hang up

    wait_until(lambda: server.engine.num_pending == 0)
    blocks = engine.scheduler.blocks
    wait_until(lambda: blocks.num_free_blocks == blocks.num_blocks)
    assert len(aborted) == 1
    assert aborted[0].finish_reason == FinishReason.ABORT
    assert len(aborted[0].output_ids) < 60


def test_failed_generation_reports_error_and_health_degrades(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("simulated CUDA failure")

    assert httpx.get(f"{server.url}/health").status_code == 200
    monkeypatch.setattr(server.engine.engine.model.model, "forward", broken_forward)

    with httpx.stream(
        "POST", f"{server.url}/v1/generate", json={"prompt": "hi", "max_tokens": 4}, timeout=30
    ) as response:
        final, done = read_events(response)
    assert final["finish_reason"] == "error" and "error" in final and done == "[DONE]"

    response = httpx.post(
        f"{server.url}/v1/generate", json={"prompt": "hi", "max_tokens": 4, "stream": False}
    )
    assert response.status_code == 500

    for _ in range(UNHEALTHY_AFTER_FAILURES - 2):
        body = {"prompt": "hi", "max_tokens": 4, "stream": False}
        assert httpx.post(f"{server.url}/v1/generate", json=body).status_code == 500
    health = httpx.get(f"{server.url}/health")
    assert health.status_code == 503
    assert health.json()["status"] == "unhealthy"


def test_health_reports_capacity(server: Server) -> None:
    response = httpx.get(f"{server.url}/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pending": 0,
        "running": 0,
        "waiting": 0,
        "free_blocks": 64,
        "total_blocks": 64,
    }
