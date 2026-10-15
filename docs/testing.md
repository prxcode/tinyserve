# Testing

```bash
uv run pytest                  # everything that can run on this machine
uv run pytest -m "not model"   # skip tests that need downloaded weights
```

## What is tested where

| Suite | Needs | Runs in CI | What it shows |
|---|---|---|---|
| `test_config.py` | nothing | yes | Config parsing, including the transformers 5 `rope_parameters` layout, and rejection of unsupported variants |
| `test_llama.py` | nothing | yes | Output shapes, causality (earlier logits ignore later tokens), RoPE invariants, tied embeddings |
| `test_loader.py` | nothing | yes | Logits match `transformers` on a small random Llama (tied and untied, single-file and sharded checkpoints), and bad checkpoints fail loudly |
| `test_block_manager.py` | nothing | yes | Allocation, exhaustion, double free, all-or-nothing reservation, slot mapping across non-contiguous blocks, and a 2000-step randomized invariant check (no block owned twice, none leaked) |
| `test_kv_cache.py` | nothing | yes | Write/read round trips through interleaved block tables, tensor slot lookup matching `BlockManager`, layer isolation, and memory size matching the formula |
| `test_sequence.py` | nothing | yes | Which tokens still need computing after partial caching, and sampling-parameter validation |
| `test_scheduler.py` | nothing | yes | Prefill priority, FCFS without skipping, padded prefill budget, decode headroom on admission, preemption of the newest sequence and its reset state, aborts, impossible requests, and a 60-request randomized workload that must finish with preemptions and no leaked blocks |
| `test_paged_attention.py` | nothing | yes | Paged prefill and decode over interleaved blocks give the same logits as recomputing without a cache (float64), and padding never writes to the cache |
| `test_engine.py` | nothing | yes | Batched greedy output is token-identical to a full-recompute reference: all at once, with requests joining mid-run, under heavy preemption, and after aborts. Also EOS and `max_tokens` handling, seeded sampling and prompt validation |
| `test_detokenizer.py` | nothing | yes | Streamed text deltas join to the full decode, including characters split across byte tokens (`°`, CJK, emoji). A guard test proves naive per-token decoding really breaks on the same input |
| `test_async_engine.py` | nothing | yes | Concurrent requests through the engine thread match the synchronous engine. Closing a stream aborts the request and frees its blocks, `max_pending` is enforced and recovers, invalid requests fail before queueing, a failed batch reports an error and the engine recovers, a crashed engine thread fails requests instead of hanging, and `stop()` ends in-flight streams |
| `test_server.py` | nothing | yes | Against a real uvicorn server on a local socket: SSE format and content, JSON mode, 12 concurrent clients each getting their own correct text, 422 and 400 cases, 503 with `Retry-After` under a held engine, a client hanging up mid-stream aborting its request, error events and `/health` degrading after repeated failures, and `/metrics` values matching exactly what a request did (token counts, one TTFT sample, one prefill and N-1 decode steps, aborts and rejections) |
| `test_quant.py` | nothing | yes | Round-trip error at most half a quantization step per row, extremes mapping to ±127, zero rows, `Int8Linear` within 1% of the float layer, only decoder linears replaced, memory matching the formula exactly, a quantized model staying within 2% of the original, and the engine running a quantized model token-identically to its own full-recompute reference |
| `test_evaluation.py` | nothing | yes | Sliding-window perplexity equals direct cross-entropy for one window, scores every token exactly once for several window and stride sizes, and actually uses the overlap as context |
| `test_sampling.py` | nothing | yes | Greedy, nucleus cut-off, empirical distribution, per-row settings, seeded reproducibility |
| `test_tinyllama.py` | downloaded weights | no | fp32 logit parity with `transformers` on the real model, a greedy sanity check, bounded fp16 GPU drift, GPU load peak memory not exceeding one copy of the weights, the engine matching full recompute on the real weights, the profiled GPU cache surviving a full `max_num_seqs` batch, streaming detokenization keeping SentencePiece spaces and multi-byte characters, and the int8 model loading without the fp16 copy reaching the GPU and still completing the Paris prompt |

The parity tests are what catch real bugs. Flipping the sign in the RoPE rotation makes the
loader parity test fail, and changing the attention mask from `<=` to `<` makes the paged
attention test fail. The real-model tests confirm that the same code holds at
full size and in fp16.

The thread and socket tests use explicit synchronization (a gate on the forward pass, polling
with a deadline) rather than sleeps that hope for the right timing. Before they were committed,
the async engine suite was run 25 times and the server suite 10 times without a failure.

## Running the real-model tests

```bash
uv run python -m tinyserve.download     # 2.2 GB into models/, ignored by git
uv run pytest -m model
```

Set `TINYSERVE_MODEL_DIR` to use weights stored elsewhere. Both models must fit in RAM
(about 9 GB) because the fp32 comparison runs on CPU. The fp16 test is skipped without a CUDA
GPU.

Measured fp16 drift (RTX 3050 Ti Laptop, torch 2.14.0+cu126): the max absolute logit difference
from fp32 was 0.12 to 0.22 across three prompts, with identical argmax at every position.
