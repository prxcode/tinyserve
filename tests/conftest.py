import pytest

from tinyserve.model.config import LlamaConfig


@pytest.fixture
def tiny_config() -> LlamaConfig:
    """A randomly initialized Llama small enough to run on CPU in milliseconds."""
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=160,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.fixture
def tinyllama() -> LlamaConfig:
    """The real TinyLlama-1.1B configuration, for size and roofline arithmetic."""
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=22,
        num_heads=32,
        num_kv_heads=4,
        head_dim=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
