# Architecture

tinyserve serves one decoder-only model (TinyLlama-1.1B by default) over HTTP. The goal is high
token throughput under many concurrent requests on a single GPU, without letting time-to-first-token
blow up. Most of the work is in three places: how KV-cache memory is allocated, how requests are
scheduled into batches, and how the HTTP layer talks to a blocking GPU loop.

This document describes the design. Each section states whether it is implemented yet; the
status is updated in the same commit that lands the component.

| Component | Module | Status |
|---|---|---|
| Model (Llama forward pass, weight loading) | `tinyserve.model` | implemented, no KV cache yet |
| Paged KV cache | `tinyserve.kv_cache` | planned |
| Scheduler | `tinyserve.scheduler` | planned |
| Sampling (temperature, top-p) | `tinyserve.sampling` | implemented |
| Engine (step loop) | `tinyserve.engine` | planned |
| HTTP server | `tinyserve.server` | planned |
| Metrics | `tinyserve.metrics` | planned |
| Int8 weight quantization | `tinyserve.model.quant` | planned |

## Request lifecycle

```
client ──HTTP──▶ FastAPI handler ──submit──▶ engine thread
                     ▲                          │
                     │  tokens (SSE)            │ loop:
                     └──── per-request queue ◀──┤   schedule() → forward() → sample()
                                                │   append tokens, free finished sequences
```

1. The handler validates the request, tokenizes the prompt and rejects it early if
   `prompt_tokens + max_tokens` can never fit in the cache.
2. The request goes into the scheduler's waiting queue. If that queue is full the server
   returns 503 instead of queueing unbounded work.
3. On each engine step the scheduler picks a batch. New requests get their prompt processed
   (prefill); running requests produce one token each (decode).
4. Each generated token is pushed to the request's asyncio queue and streamed to the client.
5. When a sequence hits EOS, `max_tokens`, or the client disconnects, its cache blocks are
   returned to the free pool on the next step.

## Model

A plain PyTorch implementation of the Llama architecture: RMSNorm, rotary position embeddings,
grouped-query attention, SwiGLU MLP. Weights are read directly from the Hugging Face
safetensors checkpoint. Hugging Face `transformers` is used only in tests as a reference to
check that our logits match.

Weights are loaded in fp16 on GPU and fp32 on CPU. fp16 is used instead of bf16 because the
T4 targeted for deployment has no fast bf16 path. The model is built on PyTorch's `meta`
device and the checkpoint tensors are assigned directly, so parameters are never allocated twice.
For now the forward pass takes a padded `[batch, seq_len]` batch and recomputes attention over
the whole sequence. Reading keys and values from the paged cache replaces this in the engine
phase.

The reason for writing the forward pass instead of calling `transformers` is the attention
call: continuous batching needs every sequence in a batch to read keys and values from its own
non-contiguous cache blocks, at its own position. That does not fit the padded
`past_key_values` interface.

## Paged KV cache

KV memory is one preallocated tensor per layer, split into fixed-size blocks
(e.g. 16 tokens each):

```
k_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]   (same for v_cache)
```

Each sequence owns a block table, a list of block ids in token order. Token `t` of a sequence
lives in block `table[t // block_size]` at offset `t % block_size`.

- `num_blocks` is computed at startup from free GPU memory after the weights are loaded,
  times a configurable utilization fraction. Memory never grows after startup, so an
  out-of-memory error during serving means a bug, not load.
- Allocation is per block, so the only waste is the unfilled tail of each sequence's last
  block. A contiguous per-request buffer would have to reserve `max_tokens` up front.
- The allocator is a free list. Allocating and freeing are O(1) and touch no GPU memory.

## Scheduler

The scheduler has no torch dependency, so its policy can be unit-tested deterministically.
It keeps two queues, `waiting` (FCFS) and `running`, and runs this on each step:

1. If there are waiting requests and enough free blocks for their prompts, admit them (up to
   a token budget per step) and run a prefill batch.
2. Otherwise run one decode step for every running sequence. A sequence that crosses a block
   boundary needs a new block first.
3. If a decode step cannot get a block, preempt the most recently admitted running sequence:
   free its blocks and put it back at the front of `waiting`. It is recomputed from its prompt
   plus the tokens generated so far. Recompute is simpler than swapping to CPU memory, and
   cheap for short sequences.

Prioritizing prefill keeps time-to-first-token low but stalls decodes while a large prefill
runs. This trade-off is measured in the benchmarks rather than assumed.

## Engine and threading

The engine runs in one dedicated thread. The model forward pass blocks, but PyTorch releases
the GIL inside kernels, so the asyncio event loop that serves HTTP stays responsive.

- Requests go in through a thread-safe queue that the engine drains at the start of each step.
- Tokens come out via `loop.call_soon_threadsafe` into a per-request `asyncio.Queue`.
- Cancellation sets a flag that the engine checks before scheduling. The sequence is dropped
  and its blocks are freed.

## Failure handling

| Situation | Behavior |
|---|---|
| Prompt plus `max_tokens` larger than the model context or the whole cache | 400 at admission |
| Waiting queue full | 503 with `Retry-After` |
| Client disconnects mid-stream | Request aborted, blocks freed on the next step |
| Cache exhausted during decode | Preemption and recompute, invisible to the client except for latency |
| Exception inside a forward pass | Requests in that batch fail with an error event and their blocks are freed; the loop keeps running and the error is counted in metrics |

## Metrics

Exposed in Prometheus format on `/metrics`: time to first token, inter-token latency and
end-to-end latency (histograms), generated tokens, running and waiting queue sizes, and KV
cache block utilization.
