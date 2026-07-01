"""Thin wrapper around the provided 8k-vocab tokenizer (``tokenizer.json``)."""

from __future__ import annotations

import os

from tokenizers import Tokenizer

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer.json")


class GraceTokenizer:
    def __init__(self, path: str = _DEFAULT_PATH):
        self.tok = Tokenizer.from_file(path)
        self.bos_id = self.tok.token_to_id("<bos>")
        self.eos_id = self.tok.token_to_id("<eos>")
        self.pad_id = self.tok.token_to_id("<pad>")

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self.tok.encode(text).ids
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)
