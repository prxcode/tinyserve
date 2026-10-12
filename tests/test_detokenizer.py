import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tinyserve.detokenizer import IncrementalDecoder

CORPUS = ["the quick brown fox jumps over the lazy dog", "hello world", "water boils at 100"]
TEXTS = [
    "hello world, the quick fox",
    "water boils at 100°C",
    "naïve café 日本語 🙂 done",
]


class ByteLevelTokenizer:
    """A small byte-level BPE trained in the test. Characters it never saw, like 日 or 🙂,
    become several byte tokens, which is exactly the case streaming has to get right."""

    def __init__(self) -> None:
        self._tokenizer = Tokenizer(models.BPE())
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self._tokenizer.train_from_iterator(CORPUS, trainer)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelTokenizer:
    return ByteLevelTokenizer()


@pytest.mark.parametrize("text", TEXTS)
def test_deltas_join_to_full_decode(tokenizer: ByteLevelTokenizer, text: str) -> None:
    prompt = tokenizer.encode("prompt: ")
    output = tokenizer.encode(text)
    decoder = IncrementalDecoder(tokenizer, prompt)

    deltas = [decoder.push(token_id) for token_id in output]

    assert "".join(deltas) == tokenizer.decode(output)
    assert not any("�" in delta for delta in deltas)


def test_split_character_is_held_until_complete(tokenizer: ByteLevelTokenizer) -> None:
    output = tokenizer.encode("日")
    assert len(output) == 3, "expected one byte token per UTF-8 byte"
    decoder = IncrementalDecoder(tokenizer, [])

    deltas = [decoder.push(token_id) for token_id in output]

    assert deltas == ["", "", "日"]


def test_naive_per_token_decoding_is_actually_broken(tokenizer: ByteLevelTokenizer) -> None:
    # Guards the test above: if single-token decoding worked, it would not prove anything.
    output = tokenizer.encode("日")

    assert "".join(tokenizer.decode([token_id]) for token_id in output) != "日"
