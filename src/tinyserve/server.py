"""HTTP API for the inference engine.

Usage: python -m tinyserve.server [--model-dir DIR] [--host HOST] [--port PORT] [...]
"""

import argparse
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from tinyserve.async_engine import AsyncEngine, EngineStoppedError, OverloadedError, TokenEvent
from tinyserve.detokenizer import IncrementalDecoder
from tinyserve.engine import Engine, EngineConfig
from tinyserve.metrics import RequestTimer
from tinyserve.model.loader import load_model
from tinyserve.sequence import FinishReason, SamplingParams
from tinyserve.tokenizer import Tokenizer

# One failed batch can be a transient OOM. Several in a row usually mean a broken CUDA context.
UNHEALTHY_AFTER_FAILURES = 3


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    # Keep generating past the end-of-sequence token. Used by benchmarks to fix output length.
    ignore_eos: bool = False
    stream: bool = True


def create_app(engine: AsyncEngine, tokenizer: TextTokenizer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="tinyserve", lifespan=lifespan)
    metrics = engine.metrics

    @app.post("/v1/generate", response_model=None)
    async def generate(body: GenerateRequest) -> StreamingResponse | dict[str, Any]:
        timer = RequestTimer(metrics)
        prompt_ids = tokenizer.encode(body.prompt)
        params = SamplingParams(
            body.max_tokens, body.temperature, body.top_p, ignore_eos=body.ignore_eos
        )
        try:
            events = engine.submit(prompt_ids, params)
        except ValueError as exc:
            metrics.rejected.labels("invalid").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OverloadedError, EngineStoppedError) as exc:
            metrics.rejected.labels("unavailable").inc()
            raise HTTPException(
                status_code=503, detail=str(exc), headers={"Retry-After": "1"}
            ) from exc
        metrics.prompt_tokens.inc(len(prompt_ids))

        decoder = IncrementalDecoder(tokenizer, prompt_ids)
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": 0}
        if body.stream:
            return StreamingResponse(
                stream_events(events, decoder, usage, timer),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        pieces = []
        reason = None
        async with aclosing(events):
            async for event in events:
                pieces.append(decode_event(event, decoder, usage, timer))
                reason = event.finish_reason
        if reason is FinishReason.ERROR:
            raise HTTPException(status_code=500, detail="generation failed")
        return {"text": "".join(pieces), "finish_reason": reason, "usage": usage}

    @app.get("/health")
    def health() -> JSONResponse:
        healthy = engine.is_alive and engine.consecutive_failures < UNHEALTHY_AFTER_FAILURES
        scheduler = engine.engine.scheduler
        # Read without locking from the event loop thread; len() of a list or deque is
        # atomic, and slightly stale numbers are fine for a health check.
        body = {
            "status": "ok" if healthy else "unhealthy",
            "pending": engine.num_pending,
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting),
            "free_blocks": scheduler.blocks.num_free_blocks,
            "total_blocks": scheduler.blocks.num_blocks,
        }
        return JSONResponse(body, status_code=200 if healthy else 503)

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def decode_event(
    event: TokenEvent, decoder: IncrementalDecoder, usage: dict[str, int], timer: RequestTimer
) -> str:
    text = ""
    if event.token_id is not None:
        timer.token()
        usage["completion_tokens"] += 1
        text = decoder.push(event.token_id)
    if event.finish_reason is not None:
        timer.finish(event.finish_reason)
    return text


async def stream_events(
    events: AsyncGenerator[TokenEvent, None],
    decoder: IncrementalDecoder,
    usage: dict[str, int],
    timer: RequestTimer,
) -> AsyncIterator[str]:
    # When the client disconnects, Starlette cancels this generator; aclosing then closes
    # the engine stream, which aborts the request and frees its cache blocks.
    finished = False
    try:
        async with aclosing(events):
            async for event in events:
                text = decode_event(event, decoder, usage, timer)
                if text:
                    yield sse({"text": text})
                if event.finish_reason is not None:
                    finished = True
                    final: dict[str, Any] = {"finish_reason": event.finish_reason, "usage": usage}
                    if event.finish_reason is FinishReason.ERROR:
                        final["error"] = "generation failed"
                    yield sse(final)
        yield "data: [DONE]\n\n"
    finally:
        if not finished:
            timer.finish(FinishReason.ABORT)  # the client went away before the final event


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=Path("models/TinyLlama-1.1B-Chat-v1.0"))
    # Loopback by default: the API has no authentication.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--num-blocks", type=int, help="required on CPU; profiled on GPU")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--max-pending", type=int, default=256)
    parser.add_argument(
        "--quantize", choices=["none", "int8"], default="none", help="int8 weight-only"
    )
    return parser.parse_args(argv)


def build_engine(args: argparse.Namespace) -> AsyncEngine:
    model = load_model(args.model_dir, args.device, quantize=args.quantize == "int8")
    config = EngineConfig(
        num_blocks=args.num_blocks,
        gpu_memory_fraction=args.gpu_memory_fraction,
        max_num_seqs=args.max_num_seqs,
        max_prefill_tokens=args.max_prefill_tokens,
        max_model_len=args.max_model_len,
    )
    return AsyncEngine(Engine(model, config), args.max_pending)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    app = create_app(build_engine(args), Tokenizer.from_model_dir(args.model_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
