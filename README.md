# tinyserve

[![CI](https://github.com/prxcode/tinyserve/actions/workflows/ci.yml/badge.svg)](https://github.com/prxcode/tinyserve/actions/workflows/ci.yml)

A small LLM inference server written from the model up: a plain PyTorch Llama implementation, a
paged KV cache, a continuous-batching scheduler and a streaming HTTP API. It serves
[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on a single GPU.

> **Status: early development.** The server streams generations from a continuous-batching
> engine over a paged KV cache, exports Prometheus metrics, and can load int8 weights.
> Throughput benchmarks and deployment are not done yet. The roadmap below tracks what is implemented;
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
- [x] Paged KV cache: block allocator, block tables and slot storage
- [x] Scheduler with continuous batching and preemption
- [x] Streaming HTTP API with backpressure and cancellation
- [x] Prometheus metrics and a reproducible load generator
- [ ] Benchmark results (pending; to be measured on AWS `g4dn.xlarge`, see
  [docs/benchmarks.md](docs/benchmarks.md))
- [x] Int8 weight-only quantization with a perplexity evaluation
- [ ] Docker image and an AWS EC2 deployment guide

Benchmark results will be published with the exact hardware, model and workload used. No
numbers are listed until they have been measured. [docs/benchmarks.md](docs/benchmarks.md)
describes the method, and why a first attempt on a laptop GPU was discarded.

Measured so far: int8 weight-only quantization cuts TinyLlama's weight memory by 44.0%
(2098 → 1175 MiB) and changes WikiText-2 perplexity from 7.5685 to 7.5707 (+0.03%)
([details](docs/benchmarks.md#int8-quantization)).

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

## Usage

```bash
uv run python -m tinyserve.server              # GPU if available, http://127.0.0.1:8000
uv run python -m tinyserve.server --help       # scheduler and cache limits
```

The KV cache is sized automatically on GPU. On CPU, pass `--num-blocks`. `--quantize int8`
loads the decoder's weights as int8, which leaves more GPU memory for the KV cache.

```bash
curl -N http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "Water boils at", "max_tokens": 20}'
```

```
data: {"text": " "}
data: {"text": "2"}
...
data: {"text": "°"}
data: {"text": "F"}
...
data: {"finish_reason": "length", "usage": {"prompt_tokens": 5, "completion_tokens": 20}}
data: [DONE]
```

Request fields: `prompt` (required), `max_tokens` (default 128), `temperature` (0 to 2,
default 0 = greedy), `top_p` (above 0 and at most 1, default 1) and `stream` (default true).
With `"stream": false` the response is one JSON object with `text`, `finish_reason` and
`usage`. `GET /health` reports queue sizes and free cache blocks, and `GET /metrics`
exports Prometheus metrics. `ignore_eos: true` keeps generating past the end-of-sequence token,
which benchmarks use to fix output length. Error codes are listed in
[docs/architecture.md](docs/architecture.md#failure-handling).

The prompt is sent to the model as raw text. TinyLlama-Chat's chat template is not applied.

## Limitations

- Single GPU, single model per process.
- Attention gathers cache blocks and calls PyTorch SDPA instead of using a fused paged-attention
  kernel, so it trades speed for code that is easy to read and verify.
- Prefill inputs are padded to the longest prompt in the batch, not packed, so mixed-length
  prefills waste compute on padding. The scheduler's token budget counts that padding.
- In fp16, greedy output can depend on which other requests share the batch (see
  [architecture.md](docs/architecture.md#engine)).
- No authentication, and the server binds to `127.0.0.1` by default. Do not expose it to the
  public internet without a proxy in front of it (see [SECURITY.md](SECURITY.md)).
- No chat template: prompts are completed as raw text.
- int8 saves memory but not time: weights are converted back to fp16 for every matmul, so
  decode is expected to be somewhat slower, not faster.
- A non-streaming request keeps running after its client disconnects, up to `max_tokens`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE). Model weights are not part of this repository. They are
downloaded from Hugging Face at runtime and keep their own license (TinyLlama is Apache-2.0).
