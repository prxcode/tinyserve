# tinyserve

[![CI](https://github.com/prxcode/tinyserve/actions/workflows/ci.yml/badge.svg)](https://github.com/prxcode/tinyserve/actions/workflows/ci.yml)

A small LLM inference server written from the model up: a plain PyTorch Llama implementation, a
paged KV cache, a continuous-batching scheduler and a streaming HTTP API. It serves
[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on a single GPU.

> **Status: early development.** The model runs and matches the reference implementation, but
> there is no cache, scheduler or server yet. The roadmap below tracks what is implemented;
> nothing unchecked should be assumed to work.

## Why

Serving one request at a time leaves most of a GPU idle: decoding a token for one sequence is
a memory-bound matrix-vector product. Batching many sequences turns it into a matrix-matrix
product with far better utilization, but sequences start and finish at different times and
each holds a growing KV cache. tinyserve explores the two standard answers to that:

- **Continuous batching**: sequences join and leave the batch at every decode step instead of
  waiting for the whole batch to finish.
- **Paged KV cache**: cache memory is allocated in fixed-size blocks on demand, so memory is
  not reserved for tokens a request may never generate.

The design is described in [docs/architecture.md](docs/architecture.md), and the reasoning
behind the main choices is in [docs/decisions.md](docs/decisions.md).

## Roadmap

- [x] Project scaffold, tooling and CI
- [x] Llama forward pass with logits checked against Hugging Face `transformers`
- [ ] Paged KV cache block allocator
- [ ] Scheduler with continuous batching and preemption
- [ ] Streaming HTTP API with backpressure and cancellation
- [ ] Prometheus metrics and a reproducible load benchmark
- [ ] Int8 weight-only quantization with a perplexity evaluation
- [ ] Docker image and an AWS EC2 deployment guide

Benchmark results will be published with the exact hardware, model and workload used. No
numbers are listed until they have been measured.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). PyTorch comes from one of two
extras:

```bash
uv sync --extra cu126   # NVIDIA GPU with a CUDA 12.6 compatible driver
uv sync --extra cpu     # CPU only (used by CI)
```

Download the model (2.2 GB, pinned to a fixed Hub revision, stored in the git-ignored `models/`):

```bash
uv run python -m tinyserve.download
```

Run the same checks as CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Tests that need the real weights skip themselves when the weights are missing. See
[docs/testing.md](docs/testing.md) for what each suite covers.

## Limitations

- Single GPU, single model per process.
- Attention gathers cache blocks and calls PyTorch SDPA instead of using a fused paged-attention
  kernel, so it trades speed for code that is easy to read and verify.
- No authentication. Do not expose the server to the public internet without a proxy in front
  of it (see [SECURITY.md](SECURITY.md)).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE). Model weights are not part of this repository. They are
downloaded from Hugging Face at runtime and keep their own license (TinyLlama is Apache-2.0).
