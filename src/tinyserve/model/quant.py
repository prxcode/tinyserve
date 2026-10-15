"""Weight-only int8 quantization with one scale per output channel.

Each weight row is scaled so its largest magnitude maps to 127 and then rounded, so the error
of any single weight is at most half a quantization step of its own row. Activations stay in
fp16. The matmul runs in fp16 on a temporary copy of the weight, which saves memory but not
time: this halves the weights' footprint without making decode faster.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyserve.model.llama import LlamaForCausalLM

INT8_MAX = 127


def quantize_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Return int8 values and per-row scales such that weight ≈ values * scale[:, None]."""
    weight32 = weight.float()
    scale = weight32.abs().amax(dim=1) / INT8_MAX
    # An all-zero row would divide by zero; any scale represents it exactly.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    values = torch.round(weight32 / scale[:, None]).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return values, scale.to(weight.dtype)


class Int8Linear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear with int8 weights."""

    weight: Tensor  # int8 [out_features, in_features]
    scale: Tensor  # [out_features], same dtype as the activations

    def __init__(self, values: Tensor, scale: Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", values)
        self.register_buffer("scale", scale)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        if linear.bias is not None:
            raise ValueError("only bias-free linear layers are supported")
        return cls(*quantize_weight(linear.weight.detach()))

    def forward(self, x: Tensor) -> Tensor:
        # Scaling each output column afterwards equals dequantizing the weight first, but
        # touches [tokens, out] values instead of [out, in].
        return F.linear(x, self.weight.to(x.dtype)) * self.scale.to(x.dtype)


def quantize_model(model: LlamaForCausalLM) -> LlamaForCausalLM:
    """Replace every linear layer inside the decoder layers with Int8Linear, in place.

    Embeddings and lm_head keep their precision: together they are about 12% of TinyLlama's
    weights, and the output projection is the layer most sensitive to rounding.
    """
    for module in list(model.model.layers.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, Int8Linear.from_linear(child))
    return model


def weight_bytes(model: nn.Module) -> int:
    """Bytes held by parameters and buffers, counting tied tensors once."""
    tensors = {t.data_ptr(): t for t in [*model.parameters(), *model.buffers()]}
    return sum(t.nbytes for t in tensors.values())
