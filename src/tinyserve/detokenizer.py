from typing import Protocol

# How many earlier tokens to re-decode for context. Enough for SentencePiece to place the
# leading space of a word and for a multi-byte character split across byte tokens.
CONTEXT_TOKENS = 5


class Decoder(Protocol):
    def decode(self, token_ids: list[int]) -> str: ...


class IncrementalDecoder:
    """Turns generated tokens into text deltas that can be streamed one at a time.

    Decoding each token alone loses SentencePiece's leading spaces and can cut a UTF-8
    character in half. Instead, re-decode a short window ending at the new token and emit
    only the text that window adds. Text ending in U+FFFD is held back until the
    character is complete.
    """

    def __init__(self, tokenizer: Decoder, prompt_ids: list[int]) -> None:
        self._tokenizer = tokenizer
        self._ids = list(prompt_ids[-CONTEXT_TOKENS:])
        self._prefix_offset = 0  # start of the window that gets re-decoded
        self._read_offset = len(self._ids)  # tokens before this have already been emitted

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        emitted = self._tokenizer.decode(self._ids[self._prefix_offset : self._read_offset])
        current = self._tokenizer.decode(self._ids[self._prefix_offset :])
        if len(current) <= len(emitted) or current.endswith("�"):
            return ""
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._ids)
        return current[len(emitted) :]
