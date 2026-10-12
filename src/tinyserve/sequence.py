from dataclasses import dataclass, field
from enum import Enum, StrEnum


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(StrEnum):
    STOP = "stop"  # end-of-sequence token
    LENGTH = "length"  # hit max_tokens
    ABORT = "abort"  # cancelled by the caller


@dataclass(eq=False)
class Sequence:
    seq_id: int
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    finish_reason: FinishReason | None = None
    # Leading tokens whose keys and values are already in the cache. Reset to 0 on preemption.
    num_cached: int = 0

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def uncached_token_ids(self) -> list[int]:
        """Tokens the next forward pass must process: [num_cached, length)."""
        num_prompt = len(self.prompt_ids)
        if self.num_cached >= num_prompt:
            return self.output_ids[self.num_cached - num_prompt :]
        return self.prompt_ids[self.num_cached :] + self.output_ids
