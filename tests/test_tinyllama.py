"""Checks against the real TinyLlama weights. Skipped unless they have been downloaded.

Run `python -m tinyserve.download` first, or point TINYSERVE_MODEL_DIR at a local copy.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import transformers

from tinyserve.detokenizer import IncrementalDecoder
from tinyserve.engine import Engine, EngineConfig
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.model.loader import load_model
from tinyserve.sequence import SamplingParams
from tinyserve.tokenizer import Tokenizer

MODEL_DIR = Path(os.environ.get("TINYSERVE_MODEL_DIR", "models/TinyLlama-1.1B-Chat-v1.0"))

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not (MODEL_DIR / "config.json").exists(), reason="TinyLlama weights not downloaded"
    ),
]

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def cpu_model() -> Iterator[LlamaForCausalLM]:
    model = load_model(MODEL_DIR, "cpu", torch.float32)
    yield model
    del model


@pytest.fixture(scope="module")
def input_ids(tokenizer: Tokenizer) -> torch.Tensor:
    return torch.tensor([tokenizer.encode(PROMPT)])


def test_tokenizer_round_trip(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode(PROMPT)

    assert ids[0] == 1  # BOS is added by the tokenizer's post-processor
    assert tokenizer.decode(ids) == PROMPT


def test_streaming_decode_keeps_spaces_and_unicode(tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode(PROMPT)
    output = tokenizer.encode("Paris, at 100°C. Naïve café, 日本語!")[1:]  # drop BOS
    decoder = IncrementalDecoder(tokenizer, prompt)

    streamed = "".join(decoder.push(token_id) for token_id in output)

    # Compare with the continuation as it reads after the prompt; decoding the output alone
    # would strip the leading space that separates it from the prompt.
    full = tokenizer.decode(prompt + output)
    assert streamed == full[len(tokenizer.decode(prompt)) :]
    assert streamed.startswith(" Paris")
    # Decoding token by token drops SentencePiece's word-leading spaces.
    assert "".join(tokenizer.decode([t]) for t in output) != streamed


def test_fp32_logits_match_transformers(
    cpu_model: LlamaForCausalLM, input_ids: torch.Tensor
) -> None:
    reference = transformers.LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)

    with torch.no_grad():
        expected = reference.eval()(input_ids).logits
        actual = cpu_model(input_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_greedy_continuation_is_sensible(
    cpu_model: LlamaForCausalLM, tokenizer: Tokenizer, input_ids: torch.Tensor
) -> None:
    ids = input_ids
    with torch.no_grad():
        for _ in range(6):
            next_id = cpu_model(ids)[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)

    assert "Paris" in tokenizer.decode(ids[0, input_ids.shape[1] :].tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_fp16_gpu_logits_stay_close_to_fp32(
    cpu_model: LlamaForCausalLM, input_ids: torch.Tensor
) -> None:
    gpu_model = load_model(MODEL_DIR, "cuda", torch.float16)

    with torch.no_grad():
        expected = cpu_model(input_ids)
        actual = gpu_model(input_ids.cuda()).float().cpu()

    # Measured max differences were 0.12-0.22 on logits up to ~25; 0.5 leaves headroom.
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    assert (actual - expected).abs().max() < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_load_peaks_at_one_copy_of_the_weights() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    model = load_model(MODEL_DIR, "cuda", torch.float16)

    weights = sum(p.nbytes for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() - baseline
    # Converting dtype on the GPU used to double the peak (4.2 GB for 2.1 GB of weights).
    assert peak < weights * 1.1


def test_engine_matches_full_recompute_on_real_weights(
    cpu_model: LlamaForCausalLM, tokenizer: Tokenizer
) -> None:
    prompts = [tokenizer.encode(text) for text in (PROMPT, "def add(a, b):", "Water boils at")]
    config = EngineConfig(num_blocks=64, max_model_len=256, max_prefill_tokens=256)

    seqs = Engine(cpu_model, config).generate(prompts, SamplingParams(max_tokens=8))

    with torch.no_grad():
        for seq, prompt in zip(seqs, prompts, strict=True):
            ids = torch.tensor([prompt])
            for _ in range(len(seq.output_ids)):
                ids = torch.cat([ids, cpu_model(ids)[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
            assert seq.output_ids == ids[0, len(prompt) :].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_cache_is_sized_to_fit_free_memory(tokenizer: Tokenizer) -> None:
    torch.cuda.empty_cache()
    engine = Engine(load_model(MODEL_DIR, "cuda", torch.float16))

    assert engine.cache.num_blocks * engine.config.block_size >= engine.max_model_len
    # A full-size batch must run after the cache took its share of memory.
    prompts = [tokenizer.encode(PROMPT)] * engine.config.max_num_seqs
    seqs = engine.generate(prompts, SamplingParams(max_tokens=4))
    assert all(len(seq.output_ids) == 4 for seq in seqs)
