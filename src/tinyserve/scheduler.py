from collections import deque
from dataclasses import dataclass

from tinyserve.block_manager import BlockManager
from tinyserve.sequence import FinishReason, Sequence, Status


@dataclass
class ScheduledBatch:
    sequences: list[Sequence]
    is_prefill: bool


class Scheduler:
    """Decides which sequences run on each engine step. Has no torch dependency.

    A step is either a prefill of newly admitted sequences or one decode token for every
    running sequence. Waiting sequences are admitted first-come first-served; when a decode
    step runs out of cache blocks, the most recently admitted sequence is preempted and later
    recomputed from scratch.
    """

    def __init__(
        self,
        blocks: BlockManager,
        max_num_seqs: int,
        max_prefill_tokens: int,
        max_model_len: int,
    ) -> None:
        # A preempted sequence can be up to max_model_len long and must fit in one prefill.
        if max_prefill_tokens < max_model_len:
            raise ValueError("max_prefill_tokens must be at least max_model_len")
        self.blocks = blocks
        self.max_num_seqs = max_num_seqs
        self.max_prefill_tokens = max_prefill_tokens
        self.max_model_len = max_model_len
        self.waiting: deque[Sequence] = deque()
        # Ordered oldest admission first, so the newest is always running[-1].
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def check(self, prompt_len: int, max_tokens: int) -> None:
        """Raise ValueError for a request that could never run. Only reads fixed limits."""
        total = prompt_len + max_tokens
        if total > self.max_model_len:
            raise ValueError(
                f"prompt ({prompt_len}) + max_tokens ({max_tokens}) "
                f"exceeds the model limit of {self.max_model_len} tokens"
            )
        if self.blocks.blocks_for(total) > self.blocks.num_blocks:
            raise ValueError(f"request needs {total} tokens, more than the whole KV cache")

    def add(self, seq: Sequence) -> None:
        self.check(len(seq.prompt_ids), seq.params.max_tokens)
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> ScheduledBatch | None:
        admitted = self._admit_waiting()
        if admitted:
            return ScheduledBatch(admitted, is_prefill=True)
        decodes = self._reserve_decode_blocks()
        return ScheduledBatch(decodes, is_prefill=False) if decodes else None

    def finish(self, seq: Sequence, reason: FinishReason) -> None:
        self.blocks.free(seq.seq_id)
        self.running.remove(seq)
        seq.status = Status.FINISHED
        seq.finish_reason = reason

    def abort(self, seq: Sequence) -> None:
        if seq.status is Status.RUNNING:
            self.finish(seq, FinishReason.ABORT)
        elif seq.status is Status.WAITING:
            self.waiting.remove(seq)
            seq.status = Status.FINISHED
            seq.finish_reason = FinishReason.ABORT

    def _admit_waiting(self) -> list[Sequence]:
        admitted: list[Sequence] = []
        longest = 0
        while self.waiting:
            seq = self.waiting[0]
            if len(self.running) >= self.max_num_seqs:
                break
            # Prefill inputs are padded to the longest sequence, so the budget counts padding.
            padded_tokens = (len(admitted) + 1) * max(longest, seq.length)
            if padded_tokens > self.max_prefill_tokens:
                break
            # Leave one free block per running sequence for its next decode step, so admitting
            # this sequence cannot immediately force a preemption.
            needed = self.blocks.blocks_needed(seq.seq_id, seq.length) + len(self.running)
            if needed > self.blocks.num_free_blocks:
                break

            self.waiting.popleft()
            self.blocks.reserve(seq.seq_id, seq.length)
            seq.status = Status.RUNNING
            self.running.append(seq)
            admitted.append(seq)
            longest = max(longest, seq.length)
        return admitted

    def _reserve_decode_blocks(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        for seq in list(self.running):
            if seq.status is not Status.RUNNING:
                continue  # preempted earlier in this loop
            while not self.blocks.can_reserve(seq.seq_id, seq.length):
                self._preempt(self.running[-1])
                if seq.status is not Status.RUNNING:
                    break
            else:
                self.blocks.reserve(seq.seq_id, seq.length)
                scheduled.append(seq)
        return scheduled

    def _preempt(self, seq: Sequence) -> None:
        self.blocks.free(seq.seq_id)
        self.running.remove(seq)
        seq.status = Status.WAITING
        seq.num_cached = 0
        # Front of the queue, so preempted work resumes before anything newer.
        self.waiting.appendleft(seq)
        self.num_preemptions += 1
