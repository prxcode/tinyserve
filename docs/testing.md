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
| `test_sampling.py` | nothing | yes | Greedy, nucleus cut-off, empirical distribution, per-row settings, seeded reproducibility |
| `test_tinyllama.py` | downloaded weights | no | fp32 logit parity with `transformers` on the real model, a greedy sanity check, bounded fp16 GPU drift, and GPU load peak memory not exceeding one copy of the weights |

The parity test on the random model is what catches real bugs. Flipping the sign in the RoPE
rotation, for example, makes it fail. The real-model tests confirm that the same code holds at
full size and in fp16.

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
