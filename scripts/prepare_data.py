"""Download erhwenkuo/wikipedia-zhtw, tokenize with the provided 8k tokenizer,
and write packed uint16 train/val bins.

Usage:
    python -m scripts.prepare_data --out-dir data --val-frac 0.005

Documents are concatenated with <eos> separators. A held-out fraction of
documents forms the validation split.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from grace.data import DTYPE, write_bin
from grace.tokenizer import GraceTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    p.add_argument("--dataset", default="erhwenkuo/wikipedia-zhtw")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument(
        "--val-tokens",
        type=int,
        default=1_000_000,
        help="hold out this many tokens (whole docs) as validation — a fixed size, not a fraction",
    )
    p.add_argument("--limit", type=int, default=None, help="cap #documents (debug)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    from datasets import load_dataset  # imported lazily so tests don't require it

    os.makedirs(args.out_dir, exist_ok=True)
    tok = GraceTokenizer()
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Fill the validation set with whole documents until it reaches --val-tokens,
    # then everything else is training data. Fixed val size, independent of corpus.
    train_ids: list[int] = []
    val_ids: list[int] = []
    for row in tqdm(ds, desc="tokenizing"):
        text = row[args.text_field]
        if not text:
            continue
        ids = tok.encode(text, add_eos=True)
        (val_ids if len(val_ids) < args.val_tokens else train_ids).extend(ids)

    n_train = write_bin(os.path.join(args.out_dir, "train.bin"), train_ids)
    n_val = write_bin(os.path.join(args.out_dir, "val.bin"), val_ids)
    print(f"wrote {n_train:,} train tokens, {n_val:,} val tokens (dtype={np.dtype(DTYPE)})")
    print(f"vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()
