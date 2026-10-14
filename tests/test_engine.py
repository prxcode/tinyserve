import pytest
import torch

from tinyserve.engine import Engine, EngineConfig
from tinyserve.model.config import LlamaConfig
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.sequence import FinishReason, SamplingParams, Status

PROMPT_LENGTHS = [5, 1, 12, 7, 3, 9]


@pytest.fixture
def model(tiny_config: LlamaConfig) -> LlamaForCausalLM:
    torch.manual_seed(0)
    # float64 so batched and unbatched runs agree exactly and greedy picks never flip.
    return LlamaForCausalLM(tiny_config).double().eval()


@pytest.fixture
def prompts(tiny_config: LlamaConfig) -> list[list[int]]:
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(3, tiny_config.vocab_size, (n,), generator=generator).tolist()
        for n in PROMPT_LENGTHS
    ]


def make_engine(model: LlamaForCausalLM, **overrides: int) -> Engine:
    config = {
        "num_blocks": 64,
        "block_size": 4,
        "max_num_seqs": 8,
        "max_prefill_tokens": 64,
        "max_model_len": 64,
    }
    return Engine(model, EngineConfig(**(config | overrides)))


def reference_greedy(model: LlamaForCausalLM, prompt: list[int], max_tokens: int) -> list[int]:
    """Recompute the whole sequence for every token: slow, but obviously correct."""
    tokens = list(prompt)
    output: list[int] = []
    with torch.no_grad():
        while len(output) < max_tokens:
            next_id = int(model(torch.tensor([tokens]))[0, -1].argmax())
            output.append(next_id)
            tokens.append(next_id)
            if next_id == model.config.eos_token_id:
                break
    return output


def test_batched_greedy_matches_reference(
    model: LlamaForCausalLM, prompts: list[list[int]]
) -> None:
    engine = make_engine(model)

    seqs = engine.generate(prompts, SamplingParams(max_tokens=10))

    for seq, prompt in zip(seqs, prompts, strict=True):
        assert seq.output_ids == reference_greedy(model, prompt, 10)
    assert engine.scheduler.blocks.num_free_blocks == engine.scheduler.blocks.num_blocks


def test_decode_steps_batch_all_running_sequences(
    model: LlamaForCausalLM, prompts: list[list[int]]
) -> None:
    engine = make_engine(model)
    for prompt in prompts:
        engine.add_request(prompt, SamplingParams(max_tokens=10))

    # The padded prefill budget (64) admits five prompts padded to 12 tokens, then the sixth.
    assert len(engine.step()) == 5
    assert len(engine.step()) == 1
    assert len(engine.step()) == len(prompts)


def test_requests_joining_mid_generation_match_reference(
    model: LlamaForCausalLM, prompts: list[list[int]]
) -> None:
    engine = make_engine(model)
    params = SamplingParams(max_tokens=10)
    seqs = []

    for prompt in prompts:
        seqs.append(engine.add_request(prompt, params))
        engine.step()
        engine.step()
    while engine.has_unfinished():
        engine.step()

    for seq, prompt in zip(seqs, prompts, strict=True):
        assert seq.output_ids == reference_greedy(model, prompt, 10)


def test_preemption_and_recompute_do_not_change_outputs(
    model: LlamaForCausalLM, prompts: list[list[int]]
) -> None:
    # 12 blocks of 4 tokens cannot hold all six sequences at full length at once.
    engine = make_engine(model, num_blocks=12, max_model_len=32, max_prefill_tokens=32)

    seqs = engine.generate(prompts, SamplingParams(max_tokens=12))

    assert engine.scheduler.num_preemptions > 0
    for seq, prompt in zip(seqs, prompts, strict=True):
        assert seq.output_ids == reference_greedy(model, prompt, 12)
    assert engine.scheduler.blocks.num_free_blocks == engine.scheduler.blocks.num_blocks


def test_abort_frees_blocks_and_leaves_others_untouched(
    model: LlamaForCausalLM, prompts: list[list[int]]
) -> None:
    engine = make_engine(model)
    seqs = [engine.add_request(p, SamplingParams(max_tokens=10)) for p in prompts]
    engine.step()
    engine.step()

    engine.abort(seqs[2])
    while engine.has_unfinished():
        engine.step()

    assert seqs[2].finish_reason == FinishReason.ABORT
    assert len(seqs[2].output_ids) < 10
    for i in (0, 1, 3, 4, 5):
        assert seqs[i].output_ids == reference_greedy(model, prompts[i], 10)
    assert engine.scheduler.blocks.num_free_blocks == engine.scheduler.blocks.num_blocks


def test_stops_at_eos(model: LlamaForCausalLM, prompts: list[list[int]]) -> None:
    engine = make_engine(model)
    # Make the first greedy token of this prompt the end-of-sequence token.
    first_token = reference_greedy(model, prompts[0], 1)[0]
    engine.eos_token_id = first_token

    (seq,) = engine.generate([prompts[0]], SamplingParams(max_tokens=10))

    assert seq.output_ids == [first_token]
    assert seq.finish_reason == FinishReason.STOP
    assert seq.status is Status.FINISHED


def test_ignore_eos_runs_to_max_tokens(model: LlamaForCausalLM, prompts: list[list[int]]) -> None:
    engine = make_engine(model)
    engine.eos_token_id = reference_greedy(model, prompts[0], 1)[0]

    (seq,) = engine.generate([prompts[0]], SamplingParams(max_tokens=5, ignore_eos=True))

    assert len(seq.output_ids) == 5
    assert seq.output_ids[0] == engine.eos_token_id
    assert seq.finish_reason == FinishReason.LENGTH


def test_max_tokens_finishes_with_length_reason(
    model: LlamaForCausalLM, prompts: list[list[int]]
) -> None:
    engine = make_engine(model)
    engine.eos_token_id = None

    seqs = engine.generate(prompts, SamplingParams(max_tokens=3))

    assert all(len(seq.output_ids) == 3 for seq in seqs)
    assert all(seq.finish_reason == FinishReason.LENGTH for seq in seqs)


def test_seeded_sampling_is_reproducible(model: LlamaForCausalLM, prompts: list[list[int]]) -> None:
    params = SamplingParams(max_tokens=8, temperature=1.0, top_p=0.9)

    first = make_engine(model, seed=123).generate(prompts, params)
    second = make_engine(model, seed=123).generate(prompts, params)

    assert [s.output_ids for s in first] == [s.output_ids for s in second]


def test_failed_forward_fails_only_its_batch(
    model: LlamaForCausalLM, prompts: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(model)
    params = SamplingParams(max_tokens=6)
    running = [engine.add_request(p, params) for p in prompts[:2]]
    engine.step()  # prefill the first two

    late = engine.add_request(prompts[2], params)
    original = engine.model.model.forward

    def fail_once(*args: object, **kwargs: object) -> torch.Tensor:
        monkeypatch.setattr(engine.model.model, "forward", original)
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(engine.model.model, "forward", fail_once)
    outputs = engine.step()  # the prefill of `late` fails

    assert [(o.seq, o.token_id, o.finish_reason) for o in outputs] == [
        (late, None, FinishReason.ERROR)
    ]
    while engine.has_unfinished():
        engine.step()
    for seq, prompt in zip(running, prompts, strict=False):
        assert seq.output_ids == reference_greedy(model, prompt, 6)
    assert engine.scheduler.blocks.num_free_blocks == engine.scheduler.blocks.num_blocks


@pytest.mark.parametrize("prompt", [[], [-1, 5], [5, 10_000]], ids=["empty", "negative", "too-big"])
def test_rejects_invalid_prompts(model: LlamaForCausalLM, prompt: list[int]) -> None:
    engine = make_engine(model)

    with pytest.raises(ValueError):
        engine.add_request(prompt, SamplingParams())


def test_cpu_engine_requires_explicit_cache_size(model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        Engine(model, EngineConfig(max_prefill_tokens=256))
