import math

import pytest
import torch
import torch.nn.functional as F

from tinyserve.evaluation import perplexity
from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM


@pytest.fixture
def model(tiny_config: LlamaConfig) -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_config).double().eval()


@pytest.fixture
def tokens(tiny_config: LlamaConfig) -> list[int]:
    generator = torch.Generator().manual_seed(1)
    return torch.randint(0, tiny_config.vocab_size, (100,), generator=generator).tolist()


def test_single_window_equals_direct_cross_entropy(
    model: LlamaForCausalLM, tokens: list[int]
) -> None:
    ids = torch.tensor([tokens])
    with torch.no_grad():
        expected = math.exp(F.cross_entropy(model(ids)[0, :-1], ids[0, 1:]).item())

    value, scored = perplexity(model, tokens, max_length=128, stride=128)

    assert scored == len(tokens) - 1
    # perplexity() computes the loss in float32, so agreement is to float32 precision.
    assert value == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(("max_length", "stride"), [(32, 32), (32, 16), (32, 7), (40, 1)])
def test_every_token_is_scored_exactly_once(
    model: LlamaForCausalLM, tokens: list[int], max_length: int, stride: int
) -> None:
    _, scored = perplexity(model, tokens, max_length=max_length, stride=stride)

    assert scored == len(tokens) - 1


def test_overlapping_windows_score_with_more_context_than_disjoint_ones(
    model: LlamaForCausalLM, tokens: list[int]
) -> None:
    # With no overlap, the first tokens of each window are predicted from almost nothing.
    # Overlap gives them context, so the two must differ; a mistake that ignored the overlap
    # and scored identically would be caught here.
    disjoint, _ = perplexity(model, tokens, max_length=20, stride=20)
    overlapping, _ = perplexity(model, tokens, max_length=20, stride=5)

    assert disjoint != pytest.approx(overlapping, rel=1e-6)


def test_max_windows_limits_the_work(model: LlamaForCausalLM, tokens: list[int]) -> None:
    _, scored = perplexity(model, tokens, max_length=20, stride=10, max_windows=2)

    assert scored == 19 + 10


def test_rejects_stride_longer_than_window(model: LlamaForCausalLM, tokens: list[int]) -> None:
    with pytest.raises(ValueError):
        perplexity(model, tokens, max_length=10, stride=11)
