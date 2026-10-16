# Benchmarks

> **Status: throughput and latency results pending.** The load-test tooling is finished, and
> those numbers will be taken on an AWS `g4dn.xlarge` (NVIDIA T4). Int8 memory and perplexity
> results are below; they are correctness numbers and do not depend on GPU speed.

Every result here is checked against the lower bounds in
[performance-model.md](performance-model.md), which were written before the measurements.

## What is measured

`bench/load_test.py` is a closed-loop load generator. For each concurrency level `N` it
starts `N` clients, and each one sends streaming requests back to back until the run's
request count is reached. Every request has:

- a prompt of about 100 words, built from a fixed word list with a fixed seed;
- exactly 128 output tokens, because `ignore_eos` stops generation from ending early.

Fixing both lengths makes runs comparable across commits and machines. It is a synthetic
workload: real traffic has varied lengths, and those change both batching efficiency and the
cost of padded prefills.

All timings are taken on the client, so they include HTTP, SSE parsing and queueing:

| Metric | Definition |
|---|---|
| Output throughput | Generated tokens across all requests ÷ wall time of the run |
| Time to first token (TTFT) | From sending the request to receiving the first streamed text |
| Inter-token latency (ITL) | Gap between consecutive streamed chunks of one request |
| Request latency | From sending the request to receiving its final event |

Percentiles use the nearest-rank method. Before the first run, two short warm-up requests
load CUDA kernels and fill the allocator's cache.

The server reports the same quantities from its own side on `/metrics` (see
[architecture.md](architecture.md#metrics)), plus engine step time and batch size per step
kind. Those are the numbers to look at when explaining the client-side ones.

## Running it

```bash
# terminal 1: the server under test
uv run python -m tinyserve.server

# terminal 2: sweep concurrency 1 to 64 and save the results
uv run python bench/load_test.py --concurrency 1 2 4 8 16 32 64 \
    --label "default settings" --out bench/results/<name>.json
```

To measure what continuous batching contributes, rerun the same sweep against a server that
can only run one sequence at a time:

```bash
uv run python -m tinyserve.server --max-num-seqs 1
```

At concurrency 1 the two setups are identical. Any gap at higher concurrency comes from
batching.

## Guarding against bad measurements

Two seconds into each run, the load generator samples the local GPU with `nvidia-smi`: power
state, SM and memory clocks, utilization, and every process using the GPU. The snapshot is
stored with the results, and the run is flagged when:

- the GPU is not in power state P0 under load;
- the memory clock is below 90% of its maximum;
- any process other than Python is using the GPU.

The check exists because of what happened on the first attempt.

### First attempt: development laptop (discarded)

The first sweep on the development laptop (RTX 3050 Ti Laptop, 4 GB) ran about ten times
slower than expected: around 1.2 s per batch-1 decode step. Profiling five decode steps with
`torch.profiler` showed 96% of GPU time in the linear layers' matrix-vector products. That
work reads every weight once per step, about 2.1 GB, and it ran at about 20 GB/s against a
hardware limit near 192 GB/s.

The code was not the cause:
- The laptop was on battery. The driver held the GPU in power state P5 with its memory clock
  at 810 of 6001 MHz.
- Another project's `llama-server` processes were using the GPU at the same time.

Those numbers were discarded. They describe a throttled, shared GPU, not this server.

The profile does confirm one point that does not depend on clock speed: batch-1 decode is
bound by memory bandwidth, because each step reads all the weights to produce one token per
sequence. So batching more sequences into each step is expected to raise throughput substantially,
until the step becomes compute-bound or the KV cache runs out. That expectation is what the
concurrency sweep will test.

## Int8 quantization

Command: `uv run python bench/perplexity.py --out bench/results/perplexity.json`
(raw output: [bench/results/perplexity.json](../bench/results/perplexity.json)).

- Data: WikiText-2 raw test split, `Salesforce/wikitext` at revision `b08601e`. Lines are joined
  with `"\n\n"`, which gives 341,469 TinyLlama tokens.
- Windows: 1024 tokens with a stride of 512, so every token is scored exactly once with at
  least 512 tokens of context after the first window.
- Both variants ran in one process on the same tokens (RTX 3050 Ti Laptop, torch 2.14.0+cu126).

| Weights | Perplexity | Weight memory |
|---|---|---|
| fp16 | 7.5685 | 2,200,096,768 B (2098 MiB) |
| int8 (decoder linears), fp16 embeddings and `lm_head` | 7.5707 | 1,232,001,024 B (1175 MiB) |
| Change | +0.0022 (+0.03%) | −44.0% |

The memory figure counts weights only. It does not depend on the GPU and can be checked
from the layer shapes: 969M decoder-linear parameters go from 2 bytes to 1, plus one 2-byte
scale per output channel. Perplexity was measured on a throttled, shared GPU. That makes the
run slower but does not change its result.

Speed is not reported here. int8 is expected to make decode slower, because each matmul
converts the weights back to fp16 first (see [architecture.md](architecture.md#int8-weight-only-quantization)).
Its latency will be measured with the throughput benchmarks.

## Throughput and latency results

Pending: to be measured on AWS `g4dn.xlarge` (T4, 16 GB) in the deployment phase.
