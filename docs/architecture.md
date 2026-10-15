# Architecture

tinyserve serves one decoder-only model (TinyLlama-1.1B by default) over HTTP. The goal is high
token throughput under many concurrent requests on a single GPU, without letting time-to-first-token
blow up. Most of the work is in three places: how KV-cache memory is allocated, how requests are
scheduled into batches, and how the HTTP layer talks to a blocking GPU loop.

This document describes the design. Each section states whether it is implemented yet; the
status is updated in the same commit that lands the component.

| Component | Module | Status |
|---|---|---|
| Model (Llama forward pass, weight loading) | `tinyserve.model` | implemented |
| Paged KV cache (allocator, block tables, storage) | `tinyserve.block_manager`, `tinyserve.kv_cache` | implemented |
| Scheduler | `tinyserve.scheduler` | implemented |
| Sampling (temperature, top-p) | `tinyserve.sampling` | implemented |
| Engine (step loop, cache sizing) | `tinyserve.engine` | implemented |
| Engine thread and async bridge | `tinyserve.async_engine` | implemented |
| Streaming detokenizer | `tinyserve.detokenizer` | implemented |
| HTTP server | `tinyserve.server` | implemented |
| Metrics | `tinyserve.metrics` | implemented |
| Int8 weight quantization | `tinyserve.model.quant` | planned |

## Request lifecycle

```
client ──HTTP──▶ FastAPI handler ──submit──▶ engine thread
                     ▲                          │
                     │  tokens (SSE)            │ loop:
                     └──── per-request queue ◀──┤   schedule() → forward() → sample()
                                                │   append tokens, free finished sequences
```

1. The handler tokenizes the prompt and validates the request on the event loop thread. It
   returns 400 if `prompt_tokens + max_tokens` can never fit in the model context or the
   cache, and 503 if too many requests are already in flight. Both happen before the
   response starts, so a client never gets a 200 that fails right away.
2. The request is handed to the engine thread through a command queue and lands in the
   scheduler's waiting queue.
3. On each engine step the scheduler picks a batch. New requests get their prompt processed
   (prefill); running requests produce one token each (decode).
4. Each generated token id is pushed to the request's asyncio queue. The handler turns it into
   text and streams it as a server-sent event.
5. When a sequence hits EOS or `max_tokens`, or the client disconnects, its cache blocks are
   returned to the free pool.

## Model

A plain PyTorch implementation of the Llama architecture: RMSNorm, rotary position embeddings,
grouped-query attention, SwiGLU MLP. Weights are read directly from the Hugging Face
safetensors checkpoint. Hugging Face `transformers` is used only in tests as a reference to
check that our logits match.

Weights are loaded in fp16 on GPU and fp32 on CPU. fp16 is used instead of bf16 because the
T4 targeted for deployment has no fast bf16 path. The model is built on PyTorch's `meta`
device and the checkpoint tensors are assigned directly, so parameters are never allocated twice.
Each tensor is converted to the target dtype on CPU before it is moved to the GPU. Converting
on the GPU briefly held two copies of every weight and peaked at 4.2 GB for 2.1 GB of weights.
The forward pass takes a padded `[batch, seq_len]` batch of new tokens. Given a `PagedBatch`,
each attention layer writes the new keys and values into their cache slots and then attends
over the sequence's whole cached context. The mask is built from absolute positions:
`key_position <= query_position`. Without a `PagedBatch` it runs plain causal attention over
the input, and that path is kept as the reference the cached path is tested against.

Prefill, decode and recompute after preemption are the same operation: process tokens
`[num_cached, length)` of each sequence. A prefill has many new tokens per sequence and a
decode has one.

The reason for writing the forward pass instead of calling `transformers` is the attention
call: continuous batching needs every sequence in a batch to read keys and values from its own
non-contiguous cache blocks, at its own position. That does not fit the padded
`past_key_values` interface.

## Paged KV cache

The cache is split into two parts:

- **`BlockManager`** (pure Python, no torch) decides which blocks belong to which sequence.
- **`KVCache`** owns the GPU tensors and only reads and writes slots it is given.

Keys and values are each one preallocated tensor covering every layer. Each layer's storage is
a flat array of token slots:

```
keys, values: [num_layers, num_blocks * block_size, num_kv_heads, head_dim]
```

Each sequence owns a block table, a list of block ids in token order. Token `t` of a sequence
lives in block `table[t // block_size]` at offset `t % block_size`, which is the flat slot
`block * block_size + offset`.

- Allocation is per block, so the only waste is the unfilled tail of each sequence's last
  block. A contiguous per-request buffer would have to reserve `max_tokens` up front.
- The allocator is a free list used as a stack, so allocating and freeing are O(1). Double
  frees raise instead of silently corrupting the free list.
- `reserve()` is all-or-nothing. A failed reservation leaves the sequence exactly as it was,
  so the scheduler can preempt and retry without cleanup.
- For attention, padded block tables are turned into `[batch, context_len]` slot indices with
  tensor ops. A Python loop over every context token would cost milliseconds per step.
- Unused slots are zero-initialized, not left empty. Padded reads land on them, and NaN garbage
  would survive a zero attention weight (`0 * NaN = NaN`).
- Memory never grows after startup, so an out-of-memory error during serving means a bug,
  not load.

**Sizing.** One TinyLlama block of 16 tokens costs
`2 (K,V) × 22 layers × 4 kv heads × 64 dims × 2 bytes × 16 = 352 KiB`, or 22 KiB per token.
On the development GPU (RTX 3050 Ti, 4 GB), 1098 MiB are free after loading the 2100 MiB of
fp16 weights. A 1 GiB cache holds 2978 blocks (47,648 tokens), and allocating it uses exactly
the predicted amount.

That 1 GiB leaves nothing for activations, so the engine sizes the cache by profiling. It runs
the largest prefill (`max_prefill_tokens` in one sequence) and the largest decode
(`max_num_seqs` sequences at `max_model_len` context) against a throwaway cache. It records
peak activation memory, frees the probe, and gives `free × gpu_memory_fraction − activations`
to KV blocks. With default settings on the development GPU, two runs gave 1246 blocks
(19,936 tokens) and 2345 blocks (37,520 tokens). The budget is taken from currently free
memory, and on a laptop the desktop and other applications use a varying share of VRAM. On a
dedicated server GPU it would be stable.

## Scheduler

The scheduler has no torch dependency, so its policy can be unit-tested deterministically.
It keeps two queues, `waiting` (FCFS) and `running`, and runs this on each step:

1. Admit waiting sequences in arrival order and run them as one prefill batch. Admission
   stops at the first sequence that does not fit, so a long prompt is never starved by short
   ones behind it. A sequence fits when:
   - the running set stays within `max_num_seqs`;
   - the batch stays within `max_prefill_tokens`, counted after padding to the longest
     sequence in the batch (one 1000-token prompt with 63 short ones is 64,000 padded tokens);
   - its blocks are free while still leaving one block per running sequence, so admitting it
     cannot force a preemption on the very next decode step.
2. Otherwise run one decode step for every running sequence. A sequence that crosses a block
   boundary gets a new block first.
3. If a decode step cannot get a block, preempt the most recently admitted running sequence:
   free its blocks and put it back at the front of `waiting`. It is later recomputed from its
   prompt plus the tokens generated so far. Recompute is simpler than swapping to CPU memory,
   and cheap for short sequences.

`max_prefill_tokens` must be at least `max_model_len`, because a preempted sequence can grow
to the full model length and must still fit in one prefill.

Prioritizing prefill keeps time-to-first-token low but stalls decodes while a large prefill
runs. This trade-off is measured in the benchmarks rather than assumed.

## Engine

`Engine.step()` runs one scheduling decision and one forward pass:

1. Ask the scheduler for a batch.
2. Pad each sequence's uncached tokens into `[batch, width]` and build the `PagedBatch`.
3. Run the model and compute logits only for each sequence's last token. The 32,000-wide vocab
   projection for the rest of a prefill is skipped.
4. Sample one token per sequence with that sequence's own temperature and top-p.
5. Append it, and finish the sequence on EOS or `max_tokens`, which frees its blocks.

Requests are validated when they are added. Empty prompts, token ids outside the vocabulary
(which would trigger a CUDA device assert and kill the process), and requests that could never
fit in the model context or the cache are rejected with `ValueError`.

**Numerics.** In float64 the batched engine is token-identical to recomputing every sequence
alone, and the tests rely on that. In fp16 on GPU, batch size and padding change which kernels
run and the order their sums are accumulated in, so greedy outputs can diverge at near-ties.
In one run of six prompts, five matched exactly and one diverged at token 25.

## Engine thread and async bridge

`AsyncEngine` runs the engine on one dedicated thread. The forward pass blocks, but PyTorch
releases the GIL inside its kernels, so the asyncio event loop that serves HTTP stays
responsive while the GPU works.

- **Only the engine thread touches the engine.** The event loop sends it `add` and `abort`
  commands through a `queue.SimpleQueue`, which it drains before every step. The engine and
  scheduler therefore need no locks.
- **Tokens come back** through `loop.call_soon_threadsafe` into a per-request `asyncio.Queue`.
- **Idle waiting.** With no work, the thread sleeps on a `threading.Event` that every command
  sets. It clears the event after waking, which can drop a wakeup, but the next loop drains
  the command queue anyway, so no command is missed.
- **`submit()` is synchronous.** Validation and the in-flight limit are checked before
  anything is queued, which is what lets the server answer 400 or 503 instead of a broken
  stream. The in-flight count is the only state shared across threads, and it has a lock.
- **Cancellation.** Each request's token stream is an async generator. If it is closed before
  the final event, its `finally` block sends `abort`, and the engine finishes the sequence and
  frees its blocks on the next loop. Starlette cancels the response generator when the client
  disconnects, so this covers dropped connections.

### Streaming detokenization

Turning one token at a time into text is wrong in two ways:
- SentencePiece drops the leading space of a word decoded on its own (`"▁Paris"` alone decodes
  to `"Paris"`).
- A multi-byte character can be split across byte tokens and decode to `U+FFFD`.

`IncrementalDecoder` re-decodes a window of up to five earlier tokens plus the new one and
emits only the text the window adds. Output ending in `U+FFFD` is held back until the
character is complete. The tests check this against both a byte-level BPE and TinyLlama's
SentencePiece tokenizer.

## Failure handling

| Situation | Behavior |
|---|---|
| Malformed body (missing prompt, `temperature` out of range, ...) | 422 from schema validation |
| Prompt plus `max_tokens` larger than the model context or the whole cache, or a prompt with token ids outside the vocabulary | 400 before anything is queued |
| More than `max_pending` requests in flight | 503 with `Retry-After: 1` |
| Client disconnects mid-stream | Request aborted and its blocks freed on the engine's next loop |
| Cache exhausted during decode | Preemption and recompute, invisible to the client except for latency |
| Exception inside a forward pass | Only that batch's requests end with `finish_reason: "error"` (SSE) or 500 (JSON), and their blocks are freed. Everything else keeps running |
| Repeated failed batches | `/health` returns 503 after 3 consecutive failed steps and recovers after a successful one. One transient OOM does not flip it, but a dead CUDA context does |
| Bug that crashes the engine thread | Every in-flight request ends with an error instead of hanging, new requests get 503, and `/health` reports unhealthy |

Non-streaming requests are not cancelled when the client disconnects. They run until EOS or
`max_tokens`, which bounds the wasted work.

## Metrics

Exposed in Prometheus format on `/metrics`.

| Metric | Type | Where it is measured |
|---|---|---|
| `tinyserve_time_to_first_token_seconds` | histogram | HTTP layer, from request arrival to first token |
| `tinyserve_inter_token_latency_seconds` | histogram | HTTP layer, between consecutive tokens of a request |
| `tinyserve_request_latency_seconds` | histogram | HTTP layer, from arrival to the final token |
| `tinyserve_requests_total{finish_reason}` | counter | HTTP layer, including `abort` for clients that hung up |
| `tinyserve_rejected_requests_total{reason}` | counter | HTTP layer: `invalid` (400) or `unavailable` (503) |
| `tinyserve_prompt_tokens_total`, `tinyserve_generated_tokens_total` | counter | HTTP layer |
| `tinyserve_engine_step_seconds{kind}` | histogram | Engine thread, per prefill or decode step |
| `tinyserve_engine_batch_size{kind}` | histogram | Engine thread, sequences per step |
| `tinyserve_preemptions_total` | counter | Engine thread |
| `tinyserve_running_requests`, `tinyserve_waiting_requests`, `tinyserve_pending_requests`, `tinyserve_kv_cache_usage_ratio` | gauge | Read from the scheduler when Prometheus scrapes |

Request latencies are measured where tokens reach the HTTP layer, so they include the hop
from the engine thread and match what a client sees, minus the network. Step times are
measured around `Engine.step()`, which ends by copying sampled tokens to the CPU, so the GPU
work is finished when the timer stops. Each server has its own Prometheus registry rather than
the process-wide default, so tests can build many servers in one process.
