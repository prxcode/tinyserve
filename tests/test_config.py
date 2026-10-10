from typing import Any

import pytest

from tinyserve.model.config import LlamaConfig


def tinyllama_config() -> dict[str, Any]:
    return {
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def test_parses_tinyllama_config() -> None:
    config = LlamaConfig.from_dict(tinyllama_config())

    assert config.num_layers == 22
    assert config.num_kv_heads == 4
    assert config.head_dim == 64
    assert config.eos_token_id == 2


def test_kv_heads_default_to_attention_heads() -> None:
    raw = tinyllama_config()
    del raw["num_key_value_heads"]

    assert LlamaConfig.from_dict(raw).num_kv_heads == 32


def test_first_eos_id_is_used_when_given_a_list() -> None:
    raw = tinyllama_config() | {"eos_token_id": [2, 7]}

    assert LlamaConfig.from_dict(raw).eos_token_id == 2


@pytest.mark.parametrize(
    "override",
    [
        {"model_type": "mistral"},
        {"rope_scaling": {"type": "linear", "factor": 2.0}},
        {"attention_bias": True},
        {"num_key_value_heads": 5},
    ],
)
def test_rejects_unsupported_configs(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LlamaConfig.from_dict(tinyllama_config() | override)
