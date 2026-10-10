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
