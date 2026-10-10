import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_id: int | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LlamaConfig":
        if raw.get("model_type") != "llama":
            raise ValueError(f"expected model_type 'llama', got {raw.get('model_type')!r}")
        # transformers 5 moved rope_theta into rope_parameters; older configs keep it top-level.
        rope = raw.get("rope_parameters") or {}
        if raw.get("rope_scaling") or rope.get("rope_type", "default") != "default":
            raise ValueError("only default rotary embeddings are supported, not rope scaling")
        if raw.get("attention_bias") or raw.get("mlp_bias"):
            raise ValueError("bias in attention or MLP layers is not supported")

        num_heads = raw["num_attention_heads"]
        num_kv_heads = raw.get("num_key_value_heads") or num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"{num_heads} heads cannot be grouped over {num_kv_heads} kv heads")

        eos = raw.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0]

        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_layers=raw["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=raw.get("head_dim") or raw["hidden_size"] // num_heads,
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=rope.get("rope_theta", raw.get("rope_theta", 10000.0)),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=eos,
        )

    @classmethod
    def from_file(cls, path: Path) -> "LlamaConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
