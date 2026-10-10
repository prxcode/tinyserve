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
tables longer and allocation more frequent. 16 is the common default. The unfilled tail averages about 8 tokens per
sequence whatever its length, about 176 KiB for TinyLlama.
Flat slots make writing a batch of new tokens a single `index_copy_` and reading a context a
single gather.

**Cost.** Block size is not tuned for this workload yet. It is a parameter, so it can be
measured once the benchmark exists.
