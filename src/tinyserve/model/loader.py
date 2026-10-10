import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM


def checkpoint_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [model_dir / name for name in sorted(set(weight_map.values()))]

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        # Pickle checkpoints (.bin/.pt) can run arbitrary code on load, so they are not supported.
        raise FileNotFoundError(f"no .safetensors weights found in {model_dir}")
    return files


def default_dtype(device: torch.device) -> torch.dtype:
    # fp16 rather than bf16 so the same setting works on pre-Ampere GPUs like the T4.
    return torch.float16 if device.type == "cuda" else torch.float32


def load_model(
    model_dir: Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> LlamaForCausalLM:
    device = torch.device(device)
    dtype = dtype or default_dtype(device)
    config = LlamaConfig.from_file(model_dir / "config.json")

    state: dict[str, torch.Tensor] = {}
    for path in checkpoint_files(model_dir):
        for name, tensor in load_file(path, device=str(device)).items():
            # Some older checkpoints store the rotary frequencies; they are recomputed instead.
            if name.endswith("rotary_emb.inv_freq"):
                continue
            state[name] = tensor.to(dtype)

    if config.tie_word_embeddings:
        state.setdefault("lm_head.weight", state["model.embed_tokens.weight"])

    # Build on the meta device so parameters are never allocated twice.
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model.load_state_dict(state, strict=True, assign=True)

    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
