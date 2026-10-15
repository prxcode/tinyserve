import math

import torch
import torch.nn.functional as F

from tinyserve.model.llama import LlamaForCausalLM


@torch.inference_mode()
def perplexity(
    model: LlamaForCausalLM,
    token_ids: list[int],
    max_length: int,
    stride: int,
    max_windows: int | None = None,
) -> tuple[float, int]:
    """Sliding-window perplexity over one long token sequence. Returns (perplexity, scored).

    Each window of up to `max_length` tokens scores only the tokens the previous window did not,
    so every token is predicted exactly once, with at least `max_length - stride` tokens of
    context once the first window is done.
    """
    if not 0 < stride <= max_length:
        raise ValueError("stride must be in (0, max_length]")
    device = model.lm_head.weight.device
    total_nll = 0.0
    scored = 0
    scored_until = 1  # the first token has nothing before it to be predicted from
    for window_index, begin in enumerate(range(0, len(token_ids), stride)):
        if max_windows is not None and window_index >= max_windows:
            break
        end = min(begin + max_length, len(token_ids))
        window = torch.tensor([token_ids[begin:end]], device=device)
        logits = model(window)[0, :-1].float()
        targets = window[0, 1:]
        # Prediction i in this window is for token begin + i + 1; keep only new ones.
        first_new = scored_until - (begin + 1)
        total_nll += F.cross_entropy(
            logits[first_new:], targets[first_new:], reduction="sum"
        ).item()
        scored += len(targets) - first_new
        scored_until = end
        if end == len(token_ids):
            break
    return math.exp(total_nll / scored), scored
