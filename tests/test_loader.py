import json
from pathlib import Path

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file

from tinyserve.model.loader import load_model
from tinyserve.model.quant import Int8Linear


def save_reference_model(
    path: Path, tie_word_embeddings: bool = False
) -> transformers.PreTrainedModel:
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=500.0,
        tie_word_embeddings=tie_word_embeddings,
    )
    reference = transformers.LlamaForCausalLM(config).eval()
    reference.save_pretrained(path)
    return reference


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_logits_match_transformers(tmp_path: Path, tie_word_embeddings: bool) -> None:
    reference = save_reference_model(tmp_path, tie_word_embeddings)
    model = load_model(tmp_path)
    input_ids = torch.randint(0, 128, (2, 12))

    with torch.no_grad():
        expected = reference(input_ids).logits
        actual = model(input_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_loads_sharded_checkpoint(tmp_path: Path) -> None:
    reference = save_reference_model(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors").unlink()

    names = sorted(state)
    shards = {
        "model-00001-of-00002.safetensors": names[:10],
        "model-00002-of-00002.safetensors": names[10:],
    }
    weight_map = {}
    for shard, shard_names in shards.items():
        save_file({name: state[name] for name in shard_names}, tmp_path / shard)
        weight_map |= dict.fromkeys(shard_names, shard)
    index = {"metadata": {}, "weight_map": weight_map}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    input_ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            load_model(tmp_path)(input_ids), reference(input_ids).logits, rtol=1e-4, atol=1e-5
        )


def test_missing_tensor_is_an_error(tmp_path: Path) -> None:
    save_reference_model(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    del state["model.layers.1.mlp.up_proj.weight"]
    save_file(state, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError, match="up_proj"):
        load_model(tmp_path)


def test_rejects_directory_without_safetensors(tmp_path: Path) -> None:
    save_reference_model(tmp_path)
    (tmp_path / "model.safetensors").unlink()

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


def test_converts_to_requested_dtype(tmp_path: Path) -> None:
    save_reference_model(tmp_path)

    model = load_model(tmp_path, dtype=torch.bfloat16)

    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_quantized_load_replaces_decoder_linears(tmp_path: Path) -> None:
    reference = save_reference_model(tmp_path)
    input_ids = torch.randint(0, 128, (2, 10))

    model = load_model(tmp_path, quantize=True)

    assert isinstance(model.model.layers[0].mlp.up_proj, Int8Linear)
    with torch.no_grad():
        expected = reference(input_ids).logits
        error = (model(input_ids) - expected).norm() / expected.norm()
    assert error < 0.02
