"""Pre-tokenized packed dataset with sliding windows.

``prepare_data.py`` concatenates all documents (separated by ``<eos>``) into one
flat uint16 token stream per split — short docs pack together, so there is no
padding. Training/eval examples are fixed **sliding windows** over that stream:
window size ``seq_len`` (default 1024) with ``overlap`` (default 256), i.e.
stride ``seq_len - overlap``. Windows may straddle document boundaries; for a
throwaway experiment that is fine and avoids padding entirely.

Each window stores ``seq_len + 1`` tokens so the next-token target exists for the
last position: ``x = row[:-1]``, ``y = row[1:]``.
"""

from __future__ import annotations

import numpy as np
import torch

# vocab is 8192 < 2**16, so uint16 is exact and halves the on-disk size.
DTYPE = np.uint16

DEFAULT_WINDOW = 1024
DEFAULT_OVERLAP = 256


def write_bin(path: str, ids) -> int:
    arr = np.asarray(ids, dtype=DTYPE)
    arr.tofile(path)
    return arr.shape[0]


class WindowedDataset:
    """Fixed sliding windows over a flat packed token memmap."""

    def __init__(self, bin_path: str, seq_len: int = DEFAULT_WINDOW, overlap: int = DEFAULT_OVERLAP):
        assert 0 <= overlap < seq_len, "need 0 <= overlap < seq_len"
        self.data = np.memmap(bin_path, dtype=DTYPE, mode="r")
        self.seq_len = seq_len
        self.overlap = overlap
        self.stride = seq_len - overlap
        self.row_len = seq_len + 1  # +1 for the shifted target
        n = len(self.data)
        if n < self.row_len:
            raise ValueError(f"{bin_path} has {n} tokens, need >= {self.row_len}")
        last = n - self.row_len
        starts = list(range(0, last + 1, self.stride))
        if starts[-1] != last:  # cover the tail without a padded remainder
            starts.append(last)
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def _xy(self, start: int):
        row = np.asarray(self.data[start : start + self.row_len], dtype=np.int64)
        row = torch.from_numpy(row)
        return row[:-1], row[1:]

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        ix = torch.randint(len(self.starts), (batch_size,), generator=generator).tolist()
        xs, ys = zip(*(self._xy(self.starts[i]) for i in ix))
        return torch.stack(xs), torch.stack(ys)

    def iter_epoch(self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None):
        """Yield (x, y) batches covering every window once."""
        order = torch.arange(len(self.starts))
        if shuffle:
            order = order[torch.randperm(len(self.starts), generator=generator)]
        order = order.tolist()
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            xs, ys = zip(*(self._xy(self.starts[j]) for j in idx))
            yield torch.stack(xs), torch.stack(ys)


def tokens_in_bin(bin_path: str) -> int:
    return int(np.memmap(bin_path, dtype=DTYPE, mode="r").shape[0])
