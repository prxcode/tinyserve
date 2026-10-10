from pathlib import Path

import tokenizers


class Tokenizer:
    """Thin wrapper over a Hugging Face tokenizer.json file."""

    def __init__(self, path: Path) -> None:
        self._tokenizer = tokenizers.Tokenizer.from_file(str(path))

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "Tokenizer":
        return cls(model_dir / "tokenizer.json")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        # The tokenizer's own post-processor decides whether a BOS token is prepended.
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
