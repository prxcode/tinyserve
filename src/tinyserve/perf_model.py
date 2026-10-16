"""Roofline lower bounds for TinyLlama-style inference, derived from the model config.

Every step of a decoder has to move its bytes through GPU memory and do its FLOPs on the
tensor cores. Whichever takes longer at the spec-sheet peak is a lower bound on the step's
time; nothing measured can beat it. Run `python -m tinyserve.perf_model` to print the tables
used in docs/performance-model.md.
"""

from dataclasses import dataclass
from pathlib import Path

from tinyserve.model.config import LlamaConfig

FP16_BYTES = 2


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_bandwidth: float  # bytes per second
    fp16_tensor_flops: float  # FLOP per second, dense, FP32 accumulate
    memory_bytes: int
    source: str

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte above which a kernel is compute-bound rather than memory-bound."""
        return self.fp16_tensor_flops / self.memory_bandwidth


T4 = Gpu(
    name="NVIDIA T4",
    memory_bandwidth=300e9,
    fp16_tensor_flops=65e12,
    memory_bytes=16 * 1024**3,
    source="NVIDIA T4 datasheet (Mar 2019): 300 GB/s GDDR6, 65 TFLOPS FP16/FP32 mixed precision",
)

# NVIDIA does not publish tensor FLOPS for this laptop part. 512 FLOP/clock/SM is GA10x's dense
# FP16 tensor rate with FP32 accumulate (GA102 whitepaper: RTX 3080 = 59.5 TFLOPS from 68 SMs at
# 1710 MHz). 2100 MHz is the max SM clock reported by nvidia-smi on the development laptop, and
# 6001 MHz the max memory clock (GDDR6, 12 Gbps effective on a 128-bit bus).
RTX_3050_TI_LAPTOP = Gpu(
    name="RTX 3050 Ti Laptop",
    memory_bandwidth=128 / 8 * 12.0e9,
    fp16_tensor_flops=20 * 512 * 2.1e9,
    memory_bytes=4 * 1024**3,
    source="20 SMs x 512 FLOP/clk x 2.1 GHz; 128-bit x 12 Gbps GDDR6",
)


@dataclass(frozen=True)
class ModelCost:
    """Parameter, byte and FLOP counts for one decoder-only Llama model."""

    config: LlamaConfig

    @property
    def decoder_linear_params(self) -> int:
        c = self.config
        kv_width = c.num_kv_heads * c.head_dim
        attention = 2 * c.hidden_size * c.hidden_size + 2 * c.hidden_size * kv_width
        mlp = 3 * c.hidden_size * c.intermediate_size
        return c.num_layers * (attention + mlp)

    @property
    def lm_head_params(self) -> int:
        return self.config.vocab_size * self.config.hidden_size

    @property
    def norm_params(self) -> int:
        return (2 * self.config.num_layers + 1) * self.config.hidden_size

    @property
    def kv_bytes_per_token(self) -> int:
        c = self.config
        return 2 * c.num_layers * c.num_kv_heads * c.head_dim * FP16_BYTES

    def weight_bytes_per_step(self, weight_bytes_per_param: float = FP16_BYTES) -> float:
        """Weights read by one forward step. Embedding rows are looked up, not streamed, and
        lm_head and norms stay fp16 even when the decoder linears are quantized."""
        streamed_fp16 = (self.lm_head_params + self.norm_params) * FP16_BYTES
        return self.decoder_linear_params * weight_bytes_per_param + streamed_fp16

    def flops_per_token(self, context: int) -> float:
        """Matmul FLOPs for one new token attending to `context` earlier tokens."""
        linear = 2 * (self.decoder_linear_params + self.lm_head_params)
        # QK^T and PV: two matmuls of `context` x head_dim per query head, per layer.
        attention = 4 * context * self.config.num_heads * self.config.head_dim
        return linear + attention * self.config.num_layers

    def decode_step(
        self, batch: int, context: int, gpu: Gpu, kv_passes: int = 1, weight_bytes: float = 2.0
    ) -> "Bound":
        """One decode step of `batch` sequences, each with `context` tokens already cached.

        kv_passes counts how many times the step moves each cached byte: 1 for an ideal fused
        kernel, 3 for tinyserve's gather (read cache, write a copy, read the copy in SDPA).
        """
        moved = self.weight_bytes_per_step(weight_bytes) + (
            kv_passes * batch * context * self.kv_bytes_per_token
        )
        flops = batch * self.flops_per_token(context)
        return Bound(moved, flops, gpu)

    def prefill(self, tokens: int, gpu: Gpu) -> "Bound":
        """Prefill of one `tokens`-long prompt with logits for the last token only.

        Attention is counted over the full tokens x tokens score matrix, which is what SDPA
        computes when it is given an explicit mask instead of is_causal.
        """
        linear = 2 * self.decoder_linear_params * tokens + 2 * self.lm_head_params
        c = self.config
        attention = 4 * tokens * tokens * c.num_heads * c.head_dim * c.num_layers
        moved = self.weight_bytes_per_step() + tokens * self.kv_bytes_per_token
        return Bound(moved, linear + attention, gpu)

    def compute_bound_batch(self, context: int, gpu: Gpu, kv_passes: int = 1) -> float | None:
        """Smallest batch at which decode turns compute-bound, or None if it never does.

        Solve batch * flops / F >= (W + batch * kv * context) / BW for batch.
        """
        per_seq_compute = self.flops_per_token(context) / gpu.fp16_tensor_flops
        per_seq_memory = kv_passes * context * self.kv_bytes_per_token / gpu.memory_bandwidth
        margin = per_seq_compute - per_seq_memory
        if margin <= 0:
            return None
        return self.weight_bytes_per_step() / gpu.memory_bandwidth / margin


@dataclass(frozen=True)
class Bound:
    bytes_moved: float
    flops: float
    gpu: Gpu

    @property
    def memory_seconds(self) -> float:
        return self.bytes_moved / self.gpu.memory_bandwidth

    @property
    def compute_seconds(self) -> float:
        return self.flops / self.gpu.fp16_tensor_flops

    @property
    def seconds(self) -> float:
        return max(self.memory_seconds, self.compute_seconds)

    @property
    def limit(self) -> str:
        return "memory" if self.memory_seconds >= self.compute_seconds else "compute"


def main() -> None:
    cost = ModelCost(LlamaConfig.from_file(Path("models/TinyLlama-1.1B-Chat-v1.0/config.json")))
    print(f"decoder linear params {cost.decoder_linear_params:,}")
    print(f"lm_head params        {cost.lm_head_params:,}")
    print(f"norm params           {cost.norm_params:,}")
    print(f"weight bytes / step   {cost.weight_bytes_per_step():,.0f} (fp16)")
    print(f"                      {cost.weight_bytes_per_step(1):,.0f} (int8 stored)")
    print(f"kv bytes / token      {cost.kv_bytes_per_token:,}")
    print(f"flops / token (c=0)   {cost.flops_per_token(0):,.0f}")

    for gpu in (T4, RTX_3050_TI_LAPTOP):
        print(f"\n== {gpu.name}: ridge {gpu.ridge_point:.1f} FLOP/byte")
        one = cost.decode_step(1, 0, gpu)
        print(f"batch-1 decode >= {one.seconds * 1e3:.2f} ms ({1 / one.seconds:.1f} tok/s)")
        print("decode, ideal attention (kv_passes=1) and tinyserve's gather (kv_passes=3):")
        print("  batch context   ideal ms  limit    tok/s  | gather ms   tok/s")
        for batch in (1, 8, 32, 64, 128, 256):
            for context in (128, 512, 1024):
                ideal = cost.decode_step(batch, context, gpu)
                gather = cost.decode_step(batch, context, gpu, kv_passes=3)
                print(
                    f"  {batch:5d} {context:7d} {ideal.seconds * 1e3:10.2f}  {ideal.limit:7s}"
                    f" {batch / ideal.seconds:8.0f}  | {gather.seconds * 1e3:9.2f}"
                    f" {batch / gather.seconds:7.0f}"
                )
        for context in (0, 128, 256, 512, 1024):
            b = cost.compute_bound_batch(context, gpu)
            print(f"compute-bound from batch {b:.0f}" if b else "never compute-bound", end="")
            print(f" at context {context}")
        # tinyserve's int8 reads 1 byte, writes a 2-byte fp16 copy and reads it again; an int8
        # kernel would read only the 1 byte.
        for label, weight_bytes in (("int8, ideal kernel", 1.0), ("int8, tinyserve", 5.0)):
            q = cost.decode_step(1, 0, gpu, weight_bytes=weight_bytes)
            print(f"batch-1 decode {label}: >= {q.seconds * 1e3:.2f} ms")
        print("prefill of one prompt:")
        for tokens in (32, 128, 217, 512, 1024, 2048):
            p = cost.prefill(tokens, gpu)
            print(f"  {tokens:5d} tokens >= {p.seconds * 1e3:7.2f} ms ({p.limit})")


if __name__ == "__main__":
    main()
