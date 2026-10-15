import pytest
import torch
from torch import nn

from tinyserve.engine import Engine, EngineConfig
from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.model.quant import Int8Linear, quantize_model, quantize_weight, weight_bytes
from tinyserve.sequence import SamplingParams


def test_round_trip_error_is_at_most_half_a_step_per_row() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 256) * torch.logspace(-3, 1, 64).unsqueeze(1)  # rows of varied scale

    values, scale = quantize_weight(weight)
    restored = values.float() * scale[:, None]

    assert values.dtype == torch.int8
    assert (restored - weight).abs().amax(dim=1).le(scale / 2 + 1e-7).all()


def test_largest_value_in_each_row_maps_to_127() -> None:
    weight = torch.tensor([[0.5, -2.0, 1.0], [3.0, 0.0, -1.5]])

    values, scale = quantize_weight(weight)

    assert values.abs().amax(dim=1).tolist() == [127, 127]
    torch.testing.assert_close(scale, torch.tensor([2.0, 3.0]) / 127)


def test_all_zero_row_stays_zero() -> None:
    values, scale = quantize_weight(torch.zeros(2, 4))

    assert values.eq(0).all()
    assert torch.isfinite(scale).all()


def test_int8_linear_matches_float_linear_closely() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(512, 256, bias=False)
    x = torch.randn(8, 512)

    quantized = Int8Linear.from_linear(linear)

    expected = linear(x)
    error = (quantized(x) - expected).norm() / expected.norm()
    assert error < 0.01


def test_rejects_linear_with_bias() -> None:
    with pytest.raises(ValueError):
        Int8Linear.from_linear(nn.Linear(4, 4, bias=True))


def test_quantizes_decoder_linears_only(tiny_config: LlamaConfig) -> None:
    model = quantize_model(LlamaForCausalLM(tiny_config))

    layer = model.model.layers[0]
    quantized = [layer.self_attn.q_proj, layer.self_attn.o_proj, layer.mlp.down_proj]
    assert all(isinstance(m, Int8Linear) for m in quantized)
    assert isinstance(model.lm_head, nn.Linear)
    assert not any(isinstance(m, nn.Linear) for m in model.model.layers.modules())


def test_weight_memory_matches_expected_size(tiny_config: LlamaConfig) -> None:
    model = LlamaForCausalLM(tiny_config).half()
    before = weight_bytes(model)
    linear_params = sum(
        m.weight.numel() for m in model.model.layers.modules() if isinstance(m, nn.Linear)
    )
    scales = sum(m.out_features for m in model.model.layers.modules() if isinstance(m, nn.Linear))

    after = weight_bytes(quantize_model(model))

    # Each linear weight drops from 2 bytes to 1, plus one fp16 scale per output channel.
    assert after == before - linear_params + 2 * scales


def test_quantized_model_stays_close_to_original(tiny_config: LlamaConfig) -> None:
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (4, 16))
    with torch.no_grad():
        expected = model(input_ids)
        actual = quantize_model(model)(input_ids)

    error = (actual - expected).norm() / expected.norm()
    agreement = (actual.argmax(-1) == expected.argmax(-1)).float().mean()
    assert error < 0.02
    assert agreement > 0.95


def test_engine_runs_quantized_model_exactly(tiny_config: LlamaConfig) -> None:
    torch.manual_seed(0)
    # Quantize first, then switch the scales to float64 for an exact comparison.
    model = quantize_model(LlamaForCausalLM(tiny_config)).double().eval()
    config = EngineConfig(
        num_blocks=32, block_size=4, max_num_seqs=4, max_prefill_tokens=64, max_model_len=64
    )
    prompts = [[5, 9, 30], [70, 3, 3, 8, 100]]

    seqs = Engine(model, config).generate(prompts, SamplingParams(max_tokens=8, ignore_eos=True))

    with torch.no_grad():
        for seq, prompt in zip(seqs, prompts, strict=True):
            ids = torch.tensor([prompt])
            for _ in range(8):
                ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
            assert seq.output_ids == ids[0, len(prompt) :].tolist()
