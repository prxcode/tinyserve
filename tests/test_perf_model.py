import pytest
import torch
from torch import nn

from tinyserve.kv_cache import KVCache
from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.model.quant import weight_bytes
from tinyserve.perf_model import T4, ModelCost


def test_counts_match_the_real_module(tiny_config: LlamaConfig) -> None:
    model = LlamaForCausalLM(tiny_config).half()
    cost = ModelCost(tiny_config)
    linears = [m for m in model.model.layers.modules() if isinstance(m, nn.Linear)]

    assert cost.decoder_linear_params == sum(m.weight.numel() for m in linears)
    assert cost.lm_head_params == model.lm_head.weight.numel()
    # Everything streamed per step is every weight except the embedding table.
    embedding = model.model.embed_tokens.weight.nbytes
    assert cost.weight_bytes_per_step() == weight_bytes(model) - embedding


def test_kv_bytes_match_the_cache(tiny_config: LlamaConfig) -> None:
    per_token = KVCache.bytes_per_block(tiny_config, 1, torch.float16)

    assert ModelCost(tiny_config).kv_bytes_per_token == per_token


def test_tinyllama_batch_one_decode_bound_on_t4(tinyllama: LlamaConfig) -> None:
    cost = ModelCost(tinyllama)

    bound = cost.decode_step(batch=1, context=0, gpu=T4)

    assert cost.weight_bytes_per_step() == 2_069_024_768
    assert bound.limit == "memory"
    assert bound.seconds == pytest.approx(2_069_024_768 / 300e9)


def test_compute_bound_batch_is_where_the_two_limits_meet(tinyllama: LlamaConfig) -> None:
    cost = ModelCost(tinyllama)

    batch = cost.compute_bound_batch(context=128, gpu=T4)

    assert batch is not None
    step = cost.decode_step(round(batch), 128, T4)
    assert step.compute_seconds == pytest.approx(step.memory_seconds, rel=0.01)


def test_long_contexts_never_become_compute_bound_on_t4(tinyllama: LlamaConfig) -> None:
    cost = ModelCost(tinyllama)

    assert cost.compute_bound_batch(context=1024, gpu=T4) is None
    assert cost.decode_step(4096, 1024, T4).limit == "memory"


def test_prefill_turns_compute_bound_past_the_ridge_point(tinyllama: LlamaConfig) -> None:
    cost = ModelCost(tinyllama)

    assert cost.prefill(64, T4).limit == "memory"
    assert cost.prefill(512, T4).limit == "compute"
