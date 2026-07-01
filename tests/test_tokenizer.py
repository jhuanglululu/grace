"""Tokenizer: ids in range and a stable (idempotent) encode/decode round-trip."""

import os

import pytest

from grace.tokenizer import GraceTokenizer

_HAS_TOKENIZER = os.path.exists(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer.json")
)
pytestmark = pytest.mark.skipif(not _HAS_TOKENIZER, reason="tokenizer.json not present")

SAMPLES = ["維基百科是一個自由的百科全書", "台灣位於東亞", "hello world 123"]


def test_ids_in_range():
    tok = GraceTokenizer()
    for s in SAMPLES:
        ids = tok.encode(s)
        assert all(0 <= i < tok.vocab_size for i in ids)


def test_encode_decode_is_stable():
    # decode() may not reproduce whitespace exactly, but re-encoding the decode
    # must reproduce the original ids — a round-trip that doesn't assume lossless
    # text reconstruction and doesn't reimplement tokenization.
    tok = GraceTokenizer()
    for s in SAMPLES:
        ids = tok.encode(s)
        assert tok.encode(tok.decode(ids)) == ids


def test_eos_appended():
    tok = GraceTokenizer()
    if tok.eos_id is not None:
        assert tok.encode("abc", add_eos=True)[-1] == tok.eos_id
