"""Sliding-window dataset: coverage, stride, and next-token round-trip."""

import numpy as np
import torch

from grace.data import WindowedDataset, write_bin


def _make_bin(tmp_path, n=50):
    path = str(tmp_path / "toy.bin")
    write_bin(path, np.arange(n))  # ids are their own index -> easy to verify
    return path


def test_window_covers_stream_with_correct_stride(tmp_path):
    ds = WindowedDataset(_make_bin(tmp_path, 50), seq_len=8, overlap=2)
    assert ds.stride == 6  # seq_len - overlap
    # consecutive windows advance by exactly `stride` (except the tail-aligned last)
    diffs = [b - a for a, b in zip(ds.starts, ds.starts[1:])]
    assert all(d == ds.stride for d in diffs[:-1])
    # the final window ends exactly at the last token -> full coverage, no padding
    assert ds.starts[-1] + ds.row_len == 50


def test_next_token_roundtrip(tmp_path):
    ds = WindowedDataset(_make_bin(tmp_path, 50), seq_len=8, overlap=2)
    for s in ds.starts:
        x, y = ds._xy(s)
        assert x.shape == (8,) and y.shape == (8,)
        # y is x shifted by one token (the training objective), by construction
        assert torch.equal(y[:-1], x[1:])
        # because ids == position, x[i] must equal its absolute stream index
        assert torch.equal(x, torch.arange(s, s + 8))


def test_get_batch_shapes_and_dtype(tmp_path):
    ds = WindowedDataset(_make_bin(tmp_path, 200), seq_len=16, overlap=4)
    g = torch.Generator().manual_seed(0)
    x, y = ds.get_batch(5, generator=g)
    assert x.shape == (5, 16) and y.shape == (5, 16)
    assert x.dtype == torch.int64
