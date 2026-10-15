# Design decisions

Short records of choices that shaped the code: what was decided, why, and what was given up.
New entries are added at the bottom when a decision is made, not after the fact.

## 1. Write the Llama forward pass instead of wrapping `transformers`

**Decision.** Implement the model in plain PyTorch and load the safetensors checkpoint
directly. `transformers` is a test-only dependency used as a numerical reference.

**Why.** Continuous batching needs an attention call where each sequence in the batch reads
its own non-contiguous cache blocks at its own position. The `transformers` cache interface
assumes padded, contiguous per-batch tensors. Owning the forward pass also keeps every
line of the hot path readable and testable.

**Cost.** More code to maintain. Only the Llama architecture is supported.

## 2. Gather-then-SDPA attention instead of a custom kernel

**Decision.** For decode, gather each sequence's K/V blocks into a padded tensor and call
`torch.nn.functional.scaled_dot_product_attention` with a mask.

**Why.** It can be verified against a reference and needs no CUDA or Triton code. Profiling will
show how much time the gather costs before any kernel work is considered.

**Cost.** Extra memory traffic for the gather and wasted compute on padding. Expected to be
slower than vLLM's PagedAttention.

## 3. Preempt by recompute, not swap

**Decision.** When the cache runs out during decode, the newest running sequence gives up its
blocks and is later recomputed from its prompt plus the tokens generated so far.

**Why.** No CPU-side block pool or copy streams are needed. For a 1.1B model with short
sequences, a prefill is cheap compared with the complexity of swapping.

**Cost.** Wasted compute under sustained memory pressure, visible as higher latency for the
preempted request.

## 4. uv with a locked dependency set and a resolution cutoff date

**Decision.** Dependencies are resolved by uv into `uv.lock`, only from releases older than a
cutoff date set in `pyproject.toml`. PyTorch comes from either the CPU or the CUDA 12.6 wheel
index through mutually exclusive extras.

**Why.** CI and local machines install exactly the same versions. CI can use the small CPU wheel
while GPU machines get CUDA. The cutoff avoids picking up a release published hours earlier.

**Cost.** Upgrades are a deliberate step: move the cutoff date, run `uv lock`, rerun the tests.

## 5. CUDA 12.6 wheels

**Decision.** The GPU extra targets the `cu126` PyTorch build.

**Why.** It runs on both the development GPU (RTX 3050 Ti laptop) and the NVIDIA T4 on AWS
`g4dn` instances with common driver versions. Newer CUDA builds need newer drivers and bring
no feature this project uses.

## 6. 16-token blocks and a flat slot layout

**Decision.** KV blocks hold 16 tokens. Cache storage is indexed by flat slot
(`block * block_size + offset`) rather than as a `[block, offset]` pair.

**Why.** Smaller blocks waste less memory in each sequence's last block, but they make block
tables longer and allocation more frequent. 16 is the common default. The unfilled tail
averages about 8 tokens per sequence whatever its length, about 176 KiB for TinyLlama.
Flat slots make writing a batch of new tokens a single `index_copy_` and reading a context a
single gather.

**Cost.** Block size is not tuned for this workload yet. It is a parameter, so it can be
measured once the benchmark exists.

## 7. Padded batches, with prefill and decode in separate steps

**Decision.** Each step is either a prefill of newly admitted sequences or a decode of all
running ones, never a mix. Inputs are padded to `[batch, longest]` rather than packed into one
flat token stream.

**Why.** Padding keeps every tensor in the familiar `[batch, heads, seq, dim]` layout that
SDPA expects, and makes the attention mask a single comparison of positions. Decode batches,
which are most steps, have one token per sequence and so no padding at all.

**Cost.** Mixed-length prefills waste compute on padding. The scheduler budgets padded tokens
so the waste is bounded, but it is not removed. A long prefill also pauses every running
decode for one step. Chunked prefill or a variable-length attention kernel would fix both,
at the cost of a more complex attention path.

## 8. Size the KV cache by profiling, not by a fixed fraction

**Decision.** At startup the engine runs the largest possible prefill and decode against a
throwaway cache, measures peak activation memory, and gives the rest of free GPU memory
(times `gpu_memory_fraction`) to KV blocks.

**Why.** A fixed split either wastes memory or runs out of it on the first large batch. Peak
activations depend on `max_prefill_tokens`, `max_num_seqs`, the attention kernel and the
model. Measuring them is more reliable than modelling them.

**Cost.** About a second and a half of extra startup time. The probe only covers batch shapes
the scheduler can produce, so changing scheduler limits means the cache is re-profiled on
the next start.

## 9. One engine thread, driven by a command queue

**Decision.** The engine and scheduler are touched only from one dedicated thread. The
asyncio side sends `add` and `abort` commands through a queue, and receives tokens through
`call_soon_threadsafe`.

**Why.** The forward pass blocks for milliseconds and cannot run on the event loop. A single
owner thread means the scheduler, block manager and cache need no locks, and every state
change happens between steps, never in the middle of one. The alternative, calling into the
engine from request handlers under a lock, would hold that lock for a whole forward pass.

**Cost.** Commands wait for the current step to finish before they take effect, so a
cancelled request can still be computed for one more step.

## 10. Validate before streaming

**Decision.** `submit()` validates the request and checks the in-flight limit synchronously,
and only then returns the token stream.

**Why.** Once a streaming response has sent its 200 status, the server can no longer report an
error through the HTTP status. Checking first means a bad or rejected request always gets a
proper 400 or 503.

**Cost.** Validation runs on the event loop thread while the engine thread is running, so it
may only read limits that never change after startup (model length, vocabulary size, cache
size). Anything that depends on live scheduler state, such as free blocks, cannot be checked
there.

## 11. Weight-only int8 with per-channel scales, written by hand

**Decision.** Quantize the decoder's linear weights to int8 with one symmetric scale per
output channel. Keep activations, embeddings and `lm_head` in fp16, and dequantize inside the
forward pass. No `bitsandbytes`.

**Why.** Per-channel scales follow the large differences in magnitude between rows, which a
single per-tensor scale would crush. Weight-only quantization keeps the activation path,
which has outliers, in fp16. Writing it by hand keeps it to about 40 lines that can be
explained and tested completely, and avoids a native dependency that is awkward on Windows.
The measured cost on WikiText-2 is +0.03% perplexity for 44% less weight memory.

**Cost.** No speedup, and probably a slowdown: the fp16 copy made for each matmul adds memory
traffic to a step that is already bandwidth-bound. Getting a speedup needs an int8 matmul
kernel, which is out of scope. The benefit is capacity: on a small GPU the saved memory
becomes KV cache, which means more sequences can be batched at once.
