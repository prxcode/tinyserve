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
| Engine (step loop, cache sizing) | `tinyserve.engine` | implemented, synchronous; the server thread is not built yet |
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
to KV blocks. With default settings on the development GPU this came to 1246 blocks
(19,936 tokens, 428 MiB), with about 300 MiB left free.

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

## Server threading (planned)

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
