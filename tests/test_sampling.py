import torch

from tinyserve.sampling import sample


def logits_from_probs(probs: list[float], rows: int = 1) -> torch.Tensor:
    return torch.tensor(probs).log().expand(rows, -1)


def draw(logits: torch.Tensor, temperature: float, top_p: float, n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    rows = logits.expand(n, -1)
    return sample(rows, torch.full((n,), temperature), torch.full((n,), top_p), generator)


def test_zero_temperature_is_argmax() -> None:
    logits = torch.randn(5, 50)

    tokens = sample(logits, torch.zeros(5), torch.ones(5))

    assert torch.equal(tokens, logits.argmax(dim=-1))


def test_top_p_restricts_to_nucleus() -> None:
    logits = logits_from_probs([0.5, 0.3, 0.15, 0.05])

    tokens = draw(logits, temperature=1.0, top_p=0.75, n=2000)

    assert set(tokens.tolist()) == {0, 1}


def test_tiny_top_p_keeps_only_the_most_likely_token() -> None:
    logits = logits_from_probs([0.1, 0.6, 0.3])

    tokens = draw(logits, temperature=1.0, top_p=1e-6, n=200)

    assert set(tokens.tolist()) == {1}


def test_full_top_p_matches_the_distribution() -> None:
    probs = [0.5, 0.3, 0.15, 0.05]

    tokens = draw(logits_from_probs(probs), temperature=1.0, top_p=1.0, n=20000)

    observed = torch.bincount(tokens, minlength=4).float() / tokens.numel()
    torch.testing.assert_close(observed, torch.tensor(probs), atol=0.02, rtol=0)


def test_mixed_batch_applies_settings_per_row() -> None:
    logits = logits_from_probs([0.4, 0.35, 0.25], rows=2).contiguous()
    generator = torch.Generator().manual_seed(0)

    tokens = [
        sample(logits, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]), generator)
        for _ in range(300)
    ]

    assert all(t[0].item() == 0 for t in tokens)
    assert {t[1].item() for t in tokens} == {0, 1, 2}


def test_seeded_generator_is_reproducible() -> None:
    logits = torch.randn(1, 100)

    first = draw(logits, temperature=0.8, top_p=0.9, n=50)
    second = draw(logits, temperature=0.8, top_p=0.9, n=50)

    assert torch.equal(first, second)
