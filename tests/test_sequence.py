import pytest

from tinyserve.sequence import SamplingParams, Sequence


def test_uncached_tokens_cover_prompt_then_outputs() -> None:
    seq = Sequence(seq_id=0, prompt_ids=[1, 2, 3], params=SamplingParams())
    seq.output_ids = [4, 5]

    assert seq.uncached_token_ids() == [1, 2, 3, 4, 5]

    seq.num_cached = 2
    assert seq.uncached_token_ids() == [3, 4, 5]

    seq.num_cached = 4
    assert seq.uncached_token_ids() == [5]


@pytest.mark.parametrize(
    "kwargs",
    [{"max_tokens": 0}, {"temperature": -0.1}, {"top_p": 0.0}, {"top_p": 1.5}],
)
def test_rejects_invalid_sampling_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)  # type: ignore[arg-type]
