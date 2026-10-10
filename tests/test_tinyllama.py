"""Checks against the real TinyLlama weights. Skipped unless they have been downloaded.

Run `python -m tinyserve.download` first, or point TINYSERVE_MODEL_DIR at a local copy.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import transformers

from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.model.loader import load_model
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
