import itertools
import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from tinyserve.block_manager import BlockManager
from tinyserve.kv_cache import KVCache, PagedBatch
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.sampling import sample
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import FinishReason, SamplingParams, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    block_size: int = 16
    # None sizes the cache from free GPU memory; it must be set when running on CPU.
    num_blocks: int | None = None
    gpu_memory_fraction: float = 0.9
    max_num_seqs: int = 64
    max_prefill_tokens: int = 2048
    # None uses the model's max_position_embeddings.
    max_model_len: int | None = None
    seed: int | None = None


@dataclass
class StepOutput:
    seq: Sequence
    token_id: int | None  # None when the step failed for this sequence
    finish_reason: FinishReason | None


class Engine:
    """Runs continuous batching: each step() schedules a batch, runs one forward pass and
    appends one token to every sequence in it. Not thread-safe; one thread drives it."""

    def __init__(self, model: LlamaForCausalLM, config: EngineConfig | None = None) -> None:
        config = config or EngineConfig()
        self.model = model
        self.config = config
        self.device = model.lm_head.weight.device
        self.dtype = model.lm_head.weight.dtype
        self.max_model_len = config.max_model_len or model.config.max_position_embeddings
        self.eos_token_id = model.config.eos_token_id

        num_blocks = config.num_blocks
        if num_blocks is None:
            num_blocks = self._profile_num_blocks()
        self.cache = KVCache(model.config, num_blocks, config.block_size, self.dtype, self.device)
        self.scheduler = Scheduler(
            BlockManager(num_blocks, config.block_size),
            config.max_num_seqs,
            config.max_prefill_tokens,
            self.max_model_len,
        )
        self.generator = None
        if config.seed is not None:
            self.generator = torch.Generator(self.device).manual_seed(config.seed)
        self._next_id = itertools.count()

    def validate_request(self, prompt_ids: list[int], params: SamplingParams) -> None:
        """Raise ValueError for a request that can never run. Reads only fixed limits, so
        the server may call it from outside the engine thread."""
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        # An out-of-range id would trigger a device-side assert that kills the CUDA context.
        if min(prompt_ids) < 0 or max(prompt_ids) >= self.model.config.vocab_size:
            raise ValueError("prompt contains token ids outside the vocabulary")
        self.scheduler.check(len(prompt_ids), params.max_tokens)

    def add_request(self, prompt_ids: list[int], params: SamplingParams) -> Sequence:
        self.validate_request(prompt_ids, params)
        seq = Sequence(next(self._next_id), list(prompt_ids), params)
        self.scheduler.add(seq)
        return seq

    def abort(self, seq: Sequence) -> None:
        self.scheduler.abort(seq)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    @torch.inference_mode()
    def step(self) -> list[StepOutput]:
        batch = self.scheduler.schedule()
        if batch is None:
            return []
        seqs = batch.sequences

        try:
            token_ids = self._forward(seqs)
        except Exception:
            # Fail only this batch. Its sequences give back their blocks, and every other
            # request keeps running on the next step.
            logger.exception("forward pass failed for a batch of %d sequences", len(seqs))
            for seq in seqs:
                self.scheduler.finish(seq, FinishReason.ERROR)
            return [StepOutput(seq, None, FinishReason.ERROR) for seq in seqs]

        outputs = []
        for seq, token_id in zip(seqs, token_ids, strict=True):
            seq.num_cached = seq.length
            seq.output_ids.append(token_id)
            reason = self._finish_reason(seq, token_id)
            if reason is not None:
                self.scheduler.finish(seq, reason)
            outputs.append(StepOutput(seq, token_id, reason))
        return outputs

    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[Sequence]:
        """Run prompts to completion. For offline use and tests; the server calls step()."""
        seqs = [self.add_request(prompt, params) for prompt in prompts]
        while self.has_unfinished():
            self.step()
        return seqs

    def _forward(self, seqs: list[Sequence]) -> list[int]:
        input_ids, positions, paged, last_index = self._prepare(seqs)
        hidden = self.model.model(input_ids, positions, paged)
        # Only the last new token of each sequence needs logits; skip the rest of the vocab matmul.
        logits = self.model.lm_head(hidden[torch.arange(len(seqs), device=self.device), last_index])
        temperatures = torch.tensor([s.params.temperature for s in seqs], device=self.device)
        top_ps = torch.tensor([s.params.top_p for s in seqs], device=self.device)
        token_ids: list[int] = sample(logits, temperatures, top_ps, self.generator).tolist()
        return token_ids

    def _finish_reason(self, seq: Sequence, token_id: int) -> FinishReason | None:
        if token_id == self.eos_token_id:
            return FinishReason.STOP
        if len(seq.output_ids) >= seq.params.max_tokens:
            return FinishReason.LENGTH
        return None

    def _prepare(self, seqs: list[Sequence]) -> tuple[Tensor, Tensor, PagedBatch, Tensor]:
        """Pad every sequence's uncached tokens into one [batch, width] input."""
        chunks = [seq.uncached_token_ids() for seq in seqs]
        lengths = [len(chunk) for chunk in chunks]
        width = max(lengths)
        input_ids = [chunk + [0] * (width - len(chunk)) for chunk in chunks]
        positions = [
            list(range(seq.num_cached, seq.length)) + [0] * (width - n)
            for seq, n in zip(seqs, lengths, strict=True)
        ]
        tables = [self.scheduler.blocks.block_table(seq.seq_id) for seq in seqs]
        max_blocks = max(len(table) for table in tables)
        tables = [table + [0] * (max_blocks - len(table)) for table in tables]

        def to_device(rows: list[list[int]] | list[int]) -> Tensor:
            return torch.tensor(rows, dtype=torch.long, device=self.device)

        position_tensor = to_device(positions)
        length_tensor = to_device(lengths)
        valid = torch.arange(width, device=self.device) < length_tensor.unsqueeze(1)
        paged = self.cache.batch_view(position_tensor, valid, to_device(tables))
        return to_device(input_ids), position_tensor, paged, length_tensor - 1

    def _profile_num_blocks(self) -> int:
        """Give the KV cache whatever GPU memory is left after the weights and the activations
        of the largest prefill and decode batches the scheduler can produce."""
        if self.device.type != "cuda":
            raise ValueError("num_blocks must be set explicitly when not running on a GPU")
        config = self.config
        model_config = self.model.config
        longest = max(config.max_prefill_tokens, self.max_model_len)
        probe_blocks = -(-longest // config.block_size)
        # Probe batches share one block table; reading the same slots costs the same memory.
        probe = KVCache(model_config, probe_blocks, config.block_size, self.dtype, self.device)
        table = torch.arange(probe_blocks, device=self.device)

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline = torch.cuda.memory_allocated(self.device)
        with torch.inference_mode():
            self._probe(probe, table, batch=1, new_tokens=config.max_prefill_tokens)
            self._probe(probe, table, batch=config.max_num_seqs, new_tokens=1)
        activations = torch.cuda.max_memory_allocated(self.device) - baseline
        del probe
        torch.cuda.empty_cache()

        free, _ = torch.cuda.mem_get_info(self.device)
        budget = int(free * config.gpu_memory_fraction) - activations
        num_blocks = KVCache.blocks_that_fit(budget, model_config, config.block_size, self.dtype)
        if num_blocks * config.block_size < self.max_model_len:
            raise RuntimeError(
                f"only {num_blocks} KV cache blocks fit in GPU memory, not enough for one "
                f"{self.max_model_len}-token sequence; lower max_prefill_tokens or max_num_seqs"
            )
        return num_blocks

    def _probe(self, cache: KVCache, table: Tensor, batch: int, new_tokens: int) -> None:
        context = table.numel() * cache.block_size
        positions = torch.arange(context - new_tokens, context, device=self.device)
        positions = positions.expand(batch, -1)
        valid = torch.ones(batch, new_tokens, dtype=torch.bool, device=self.device)
        paged = cache.batch_view(positions, valid, table.expand(batch, -1))
        input_ids = torch.zeros(batch, new_tokens, dtype=torch.long, device=self.device)
        hidden = self.model.model(input_ids, positions, paged)
        self.model.lm_head(hidden[:, -1])
