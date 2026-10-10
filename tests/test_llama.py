from dataclasses import replace

import torch

from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM, apply_rotary, rotary_cos_sin


def test_logits_shape(tiny_config: LlamaConfig) -> None:
    model = LlamaForCausalLM(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (3, 7))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (3, 7, tiny_config.vocab_size)


def test_earlier_logits_ignore_later_tokens(tiny_config: LlamaConfig) -> None:
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config).eval()
    prefix = torch.randint(0, tiny_config.vocab_size, (1, 5))
    longer = torch.cat([prefix, torch.randint(0, tiny_config.vocab_size, (1, 4))], dim=1)

    with torch.no_grad():
        short_logits = model(prefix)
        long_logits = model(longer)[:, :5]

    torch.testing.assert_close(short_logits, long_logits)


def test_rotary_preserves_norm_and_is_identity_at_position_zero() -> None:
    x = torch.randn(2, 4, 6, 16)
    positions = torch.arange(6).expand(2, -1)
    cos, sin = rotary_cos_sin(positions, head_dim=16, theta=10000.0, dtype=torch.float32)

    rotated = apply_rotary(x, cos.unsqueeze(1), sin.unsqueeze(1))

    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])


def test_tied_embeddings_share_storage(tiny_config: LlamaConfig) -> None:
    config = replace(tiny_config, tie_word_embeddings=True)
    model = LlamaForCausalLM(config)

    assert model.lm_head.weight is model.model.embed_tokens.weight
