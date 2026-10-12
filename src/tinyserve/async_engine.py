import asyncio
import contextlib
import logging
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from tinyserve.engine import Engine
from tinyserve.sequence import FinishReason, SamplingParams, Sequence

logger = logging.getLogger(__name__)


class OverloadedError(RuntimeError):
    """Too many requests are already in flight; the caller should retry later."""


class EngineStoppedError(RuntimeError):
    """The engine thread is not running, so the request could never complete."""


@dataclass
class TokenEvent:
    token_id: int | None  # None for events that only carry a finish reason
    finish_reason: FinishReason | None


@dataclass(eq=False)
class _Request:
    prompt_ids: list[int]
    params: SamplingParams
    events: asyncio.Queue[TokenEvent]
    loop: asyncio.AbstractEventLoop
    seq: Sequence | None = None  # assigned on the engine thread

    def send(self, event: TokenEvent) -> None:
        # RuntimeError means the caller's event loop has already closed; nobody is listening.
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.events.put_nowait, event)


class AsyncEngine:
    """Runs an Engine on its own thread and streams tokens to asyncio callers.

    Only the engine thread touches the Engine. Callers send it commands through a
    thread-safe queue, and tokens come back through each request's asyncio.Queue.
    The forward pass releases the GIL inside its kernels, so the event loop stays
    responsive while the engine computes.
    """

    def __init__(self, engine: Engine, max_pending: int = 256) -> None:
        self.engine = engine
        self.max_pending = max_pending
        self.consecutive_failures = 0  # written by the engine thread only
        self._commands: queue.SimpleQueue[tuple[Literal["add", "abort"], _Request]] = (
            queue.SimpleQueue()
        )
        self._wakeup = threading.Event()
        self._stopping = threading.Event()
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._requests: dict[int, _Request] = {}  # engine thread only
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)

    @property
    def num_pending(self) -> int:
        return self._pending

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stopping.set()
        self._wakeup.set()
        self._thread.join(timeout)

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> AsyncIterator[TokenEvent]:
        """Queue a request and return its token stream.

        Raises ValueError or OverloadedError right away, before any response has started.
        Closing the stream early aborts the request and frees its cache blocks.
        """
        if self._stopping.is_set() or not self.is_alive:
            raise EngineStoppedError("engine is not running")
        self.engine.validate_request(prompt_ids, params)
        with self._pending_lock:
            if self._pending >= self.max_pending:
                raise OverloadedError(f"{self._pending} requests already in flight")
            self._pending += 1

        request = _Request(list(prompt_ids), params, asyncio.Queue(), asyncio.get_running_loop())
        self._send("add", request)
        return self._stream(request)

    async def _stream(self, request: _Request) -> AsyncIterator[TokenEvent]:
        finished = False
        try:
            while not finished:
                event = await request.events.get()
                finished = event.finish_reason is not None
                yield event
        finally:
            if not finished:
                self._send("abort", request)

    def _send(self, command: Literal["add", "abort"], request: _Request) -> None:
        self._commands.put((command, request))
        self._wakeup.set()

    def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                self._handle_commands()
                if not self.engine.has_unfinished():
                    # Clearing after the wait can drop a wakeup, but the next loop drains the
                    # command queue anyway, so no command is ever missed.
                    self._wakeup.wait()
                    self._wakeup.clear()
                    continue
                self._step()
        except Exception:
            # A bug outside the forward pass. Engine state can no longer be trusted, so end
            # every request instead of leaving its caller waiting forever.
            logger.exception("engine thread crashed")
            self._stopping.set()
            self._end_all(FinishReason.ERROR)
        else:
            self._end_all(FinishReason.ABORT)

    def _step(self) -> None:
        outputs = self.engine.step()
        failed = False
        for output in outputs:
            request = self._requests[output.seq.seq_id]
            if output.finish_reason is not None:
                self._release(request)
            failed |= output.finish_reason is FinishReason.ERROR
            request.send(TokenEvent(output.token_id, output.finish_reason))
        if outputs:
            self.consecutive_failures = self.consecutive_failures + 1 if failed else 0

    def _handle_commands(self) -> None:
        while True:
            try:
                command, request = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "add":
                try:
                    request.seq = self.engine.add_request(request.prompt_ids, request.params)
                except ValueError:
                    # Already validated in submit(); only reachable if limits changed since.
                    logger.exception("request rejected on the engine thread")
                    self._release(request)
                    request.send(TokenEvent(None, FinishReason.ERROR))
                    continue
                self._requests[request.seq.seq_id] = request
            elif request.seq is not None and request.seq.seq_id in self._requests:
                self.engine.abort(request.seq)
                self._release(request)

    def _release(self, request: _Request) -> None:
        if request.seq is not None:
            self._requests.pop(request.seq.seq_id, None)
        with self._pending_lock:
            self._pending -= 1

    def _end_all(self, reason: FinishReason) -> None:
        """Finish every request the engine thread knows about, including unread commands."""
        unstarted = []
        while True:
            try:
                command, request = self._commands.get_nowait()
            except queue.Empty:
                break
            if command == "add":
                unstarted.append(request)
        for request in [*self._requests.values(), *unstarted]:
            self._release(request)
            request.send(TokenEvent(None, reason))
        logger.info("engine thread stopped")
