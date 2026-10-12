import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyserve.kv_cache import PagedBatch
from tinyserve.model.config import LlamaConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Squaring fp16 activations can overflow, so normalize in float32.
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


def rotary_cos_sin(
    positions: Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    exponents = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (exponents / head_dim))
    angles = positions.float()[..., None] * inv_freq
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Llama pairs dimension i with i + head_dim/2 (not adjacent dims) when rotating.
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, paged: PagedBatch | None) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if paged is None:
            # No cache: plain causal attention over the given tokens. Used as the reference.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            k, v = paged.update(self.layer_idx, k, v)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=paged.mask, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, paged: PagedBatch | None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, paged)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, i) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, input_ids: Tensor, positions: Tensor, paged: PagedBatch | None = None
    ) -> Tensor:
        x = self.embed_tokens(input_ids)
        cos, sin = rotary_cos_sin(positions, self.config.head_dim, self.config.rope_theta, x.dtype)
        # Add a head axis so cos/sin broadcast over [batch, heads, seq, head_dim].
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, cos, sin, paged)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    """Module names match Hugging Face checkpoints so weights load without renaming."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: Tensor, positions: Tensor | None = None) -> Tensor:
        """Return logits of shape [batch, seq_len, vocab] for a [batch, seq_len] input."""
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            positions = positions.expand(input_ids.shape[0], -1)
        return self.lm_head(self.model(input_ids, positions))
