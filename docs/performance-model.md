# Performance model

This page predicts how fast tinyserve *can* be on the NVIDIA T4 (AWS `g4dn.xlarge`) and on the
development laptop's RTX 3050 Ti, before any of it is measured. Measurements are checked
against these predictions: a result faster than a lower bound means the measurement is wrong,
and a result far slower needs a profile that explains the gap.

The predictions were committed before the measurements they predict. The git history shows
the order. Every count below comes from [`tinyserve.perf_model`](../src/tinyserve/perf_model.py),
whose tests tie it to the real model's tensors, so the tables can be regenerated with:

```bash
uv run python -m tinyserve.perf_model
```

## Hardware

| | NVIDIA T4 | RTX 3050 Ti Laptop |
|---|---|---|
| Memory bandwidth | 300 GB/s | 192 GB/s |
| FP16 tensor FLOPS (dense, FP32 accumulate) | 65 TFLOPS | 21.5 TFLOPS |
| Ridge point (FLOPS ÷ bandwidth) | 216.7 FLOP/byte | 112.0 FLOP/byte |
| Memory | 16 GB | 4 GB |

- **T4:** [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
  (March 2019): "Mixed-Precision (FP16/FP32) 65 TFLOPS" and "16 GB GDDR6, 300 GB/sec". Some
  third-party pages say 320 GB/s; this model uses the datasheet. The T4 is capped at 70 W
  and can run below its boost clock under sustained load. That makes 65 TFLOPS
  an upper limit, and so the compute bounds below are, if anything, too optimistic.
- **RTX 3050 Ti Laptop:** NVIDIA publishes no tensor FLOPS for this part, so the figure is
  derived:
  - Ampere GA10x does 512 dense FP16 tensor FLOP per clock per SM with FP32 accumulate. The
    [GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
    lists the RTX 3080 at 59.5 TFLOPS from 68 SMs at 1710 MHz, which is 512 per SM per clock.
  - The chip has 20 SMs, and the laptop's `nvidia-smi` reports a 2100 MHz maximum SM clock.
  - Memory is a 128-bit GDDR6 bus at the 6001 MHz maximum memory clock `nvidia-smi` reports
    (12 Gbps effective): 128 / 8 × 12 Gbps = 192 GB/s.
  - Laptop clocks depend on the power limit, so these are peak figures.

PyTorch's fp16 matmuls accumulate in FP32, so the FP32-accumulate rates are the right ones.

## What one step moves and computes

TinyLlama-1.1B: hidden size 2048, 22 layers, 32 query heads and 4 KV heads of 64 dimensions,
MLP width 5632, vocabulary 32,000.

**Weights read per forward step.** Every decoder linear layer, `lm_head` and the norms are
streamed once per step. The embedding table is only indexed, so it is not streamed.

```
per layer:  q, o     2 × 2048 × 2048            =  8,388,608
            k, v     2 × 2048 × (4 × 64)        =  1,048,576
            mlp      3 × 2048 × 5632            = 34,603,008
                                                  44,040,192 × 22 layers = 968,884,224
lm_head     32,000 × 2048                       =     65,536,000
norms       (2 × 22 + 1) × 2048                 =         92,160
fp16 bytes  2 × 1,034,512,384                   =  2,069,024,768 B  (W)
```

**KV cache per cached token:** 2 (K and V) × 22 layers × 4 heads × 64 dims × 2 bytes = **22,528 B**.

**FLOPs per generated token** with `c` tokens of context:

```
linear      2 × (968,884,224 + 65,536,000)      = 2,068,840,448
attention   4 × c × 32 heads × 64 dims × 22     =       180,224 × c
```

A step takes at least `max(bytes ÷ bandwidth, FLOPs ÷ tensor FLOPS)`.

## Prediction 1: batch-1 decode

One sequence, one new token: 2.07 GB to read, 2.07 GFLOP to compute. That is about 1 FLOP per
byte, far below either ridge point, so this step is bound by memory bandwidth.

| | Lower bound per step | Upper bound on tokens/s |
|---|---|---|
| T4 | 2,069,024,768 B ÷ 300 GB/s = **6.90 ms** | **145** |
| RTX 3050 Ti Laptop | 2,069,024,768 B ÷ 192 GB/s = **10.78 ms** | **93** |

Context barely matters at batch 1: 1,024 cached tokens add 23 MB to a 2,069 MB read.

## Prediction 2: batch-1 decode on the T4 is host-bound, not GPU-bound

A profile of one tinyserve decode step (`torch.profiler`, fp16, batch 1) shows **1,375 CUDA
kernel launches**, about 62 per layer. Most are small elementwise kernels: RMSNorm, rotary
embeddings, dtype casts and mask construction. The matmuls are a minority. At batch 32 the
count is 1,441.

PyTorch launches these kernels one at a time from Python. If the host takes `t` per launch,
the step cannot finish in less than `1,375 × t`. The break-even point with the GPU bound is

```
6.90 ms ÷ 1,375 kernels = 5.0 µs per kernel
```

**Hypothesis:** eager PyTorch on the `g4dn.xlarge` CPU needs more than 5 µs per kernel, so the
batch-1 decode step will be set by host overhead. Nsight Systems will show it as gaps between
kernels on the GPU timeline. This is a hypothesis, not a measured fact. It will be tested by
measuring host time per kernel on that instance. If it holds, it is the gap that removing
per-kernel launches (CUDA graphs) would close. If it does not hold, that optimization is not
justified.

The kernel count does not depend on clock speed, so it was taken on the development laptop.
It does depend on the PyTorch version (2.14.0+cu126 here).

## Prediction 3: batched decode stays memory-bound

With `B` sequences that each have `c` cached tokens, one step reads the weights once plus
every sequence's KV cache:

```
bytes  = W + B × c × 22,528 × kv_passes
FLOPs  = B × (2,068,840,448 + 180,224 × c)
```

`kv_passes` is 1 for an ideal attention kernel that reads the cache in place. tinyserve
gathers each sequence's blocks into a padded tensor first: it reads the cache, writes the
copy, and SDPA reads the copy again. That makes about **3 passes**, so tinyserve's bound uses
`kv_passes = 3`.

Lower bounds on T4 step time, in ms, with the resulting tokens/s:

| Batch | Context | Ideal kernel | tinyserve (3 KV passes) |
|---|---|---|---|
| 1 | 512 | 6.94 ms (144 tok/s) | 7.01 ms (143 tok/s) |
| 8 | 512 | 7.20 ms (1,110) | 7.82 ms (1,023) |
| 32 | 512 | 8.13 ms (3,937) | 10.59 ms (3,022) |
| 64 | 512 | 9.36 ms (6,840) | 14.28 ms (4,482) |
| 64 | 1024 | 11.82 ms (5,415) | 21.66 ms (2,955) |
| 128 | 512 | 11.82 ms (10,831) | 21.66 ms (5,909) |

**When does decode become compute-bound?** Setting the compute time equal to the memory time
and solving for `B` (ideal kernel):

```
B × (2.069e9 + 180,224 c) / 65e12  =  (2.069e9 + B × 22,528 c) / 300e9
```

| Context `c` | T4: compute-bound from batch | RTX 3050 Ti Laptop |
|---|---|---|
| 0 | 217 | 112 |
| 128 | 306 | 131 |
| 256 | 518 | 158 |
| 512 | **never** | 267 |
| 1024 | never | never |

Above about 440 tokens of context on the T4, every extra sequence adds more bytes of KV cache
than its FLOPs can cover, so **decode is never compute-bound at any batch size**. Batching
amortizes the weight reads, but not the KV reads. tinyserve's `max_num_seqs` of 64 is far below
every crossover in the table, so batched decode here is a memory-bandwidth problem throughout.
The gap between the two columns, 3 KV passes against 1, is the cost of gather-then-SDPA. It
grows with batch and context: 53% more time at batch 64 and context 512.

## Prediction 4: int8 decode is slower than fp16

tinyserve's int8 layers read 1 byte per weight, write a 2-byte fp16 copy, and read that copy
in the matmul. The largest copies (11.5 M parameters per MLP matrix, 23 MB in fp16) are far
bigger than the T4's 4 MB L2 cache, so the copy goes through DRAM. That is about 5 bytes per
decoder weight instead of 2:

| Batch-1 decode lower bound | T4 | RTX 3050 Ti Laptop |
|---|---|---|
| fp16 | 6.90 ms | 10.78 ms |
| int8, with a real int8 kernel (1 byte/weight) | 3.67 ms | 5.73 ms |
| int8 as tinyserve implements it (5 bytes/weight) | **16.59 ms** | 25.91 ms |

**Prediction:** tinyserve's int8 batch-1 decode is at least about 2.4× slower than fp16 *on the
GPU*. If Prediction 2 holds, host overhead hides part of that, because int8 also launches more
kernels (1,682 against 1,375) and the gap at the end-to-end level will be smaller. The latency
benchmark will show which one dominates.

## Prediction 5: prefill and time to first token

Prefill of an `n`-token prompt computes `2 × 968,884,224 × n` linear FLOPs plus a full
`n × n` attention score matrix. SDPA with an explicit mask does not skip the masked half.
It turns compute-bound once `n` passes the ridge point, about 217 tokens on the T4:

| Prompt tokens | T4 lower bound | RTX 3050 Ti Laptop |
|---|---|---|
| 128 | 6.91 ms (memory) | 11.68 ms (compute) |
| 512 | 15.99 ms (compute) | 48.34 ms (compute) |
| 1024 | 33.44 ms (compute) | 101.07 ms (compute) |
| 2048 | 72.69 ms (compute) | 219.71 ms (compute) |

These are floors for time to first token with an empty queue. Under load, time to first
token also includes waiting for admission and for a decode step to finish. The open-loop
benchmark measures that.

## Prediction 6: KV cache capacity on the T4

The engine sizes the cache from free memory after the weights and the profiled activations.
As an estimate: (16 GB − 2.2 GB of weights − about 1 GB of activations and CUDA context) ×
0.9 ÷ 22,528 B ≈ **500,000 cached tokens**, which is far more than 64 sequences × 2,048 tokens
= 131,072. On the T4, `max_num_seqs` should therefore limit batch size before the cache does,
and preemption should not occur. The engine's profiled block count and the
`tinyserve_preemptions_total` counter will confirm or refute this.

## Check against an earlier measurement

The discarded laptop run (see [benchmarks.md](benchmarks.md#first-attempt-development-laptop-discarded))
was throttled to an 810 MHz memory clock: 810 / 6001 × 192 GB/s = 25.9 GB/s. At that
bandwidth, reading 2.07 GB of weights takes at least **80 ms**. The profiler measured **104 ms**
of matmul kernel time per decode step. That is 77% of the throttled bandwidth bound: slower
than the bound, as it must be, and close enough that the matmuls were clearly
bandwidth-limited. The model explains the slow run instead of only describing it.

## What would prove this model wrong

| Observation | What it would mean |
|---|---|
| Any measured step faster than its lower bound | A measurement or timing bug, or a wrong spec figure |
| Batch-1 decode on the T4 well above 6.90 ms, with a busy GPU timeline | Kernels running far below peak bandwidth, not host overhead (refutes Prediction 2) |
| Throughput scaling with batch beyond what the memory-bound rows allow | The KV-pass count or the bytes model is wrong |
| int8 decode no slower than fp16 at large batch | Host overhead or KV traffic dominates; to be explained from the profile |
| Preemptions on the T4 at `max_num_seqs` 64 | Activations or fragmentation use far more memory than estimated |
