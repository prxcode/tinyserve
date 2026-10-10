import torch
from torch import Tensor


def sample(
    logits: Tensor,
    temperatures: Tensor,
    top_ps: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pick one token id per row of [batch, vocab] logits.

    Each row has its own temperature and top-p so one batch can mix requests with different
    settings. A temperature of 0 means greedy decoding for that row.
    """
    logits = logits.float()
    greedy = logits.argmax(dim=-1)
    sampling_rows = temperatures > 0
    if not sampling_rows.any():
        return greedy

    # Greedy rows still go through the math below; clamping avoids a division by zero there.
    scaled = logits / temperatures.clamp(min=1e-5).unsqueeze(-1)
    probs = torch.softmax(scaled, dim=-1)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)

    # Drop a token once the tokens ranked above it already cover top_p; the first always stays.
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs[mass_before >= top_ps.unsqueeze(-1)] = 0.0

    choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    sampled = sorted_ids.gather(-1, choice).squeeze(-1)
    return torch.where(sampling_rows, sampled, greedy)
