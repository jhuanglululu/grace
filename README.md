# GRACE

A ~50M-parameter **GRACE** model (Grouped Residual Attention over Composed
dEpth) and a size/FLOP-matched **baseline** transformer, trained on
[`erhwenkuo/wikipedia-zhtw`](https://huggingface.co/datasets/erhwenkuo/wikipedia-zhtw)
with the provided 8k-vocab tokenizer (`tokenizer.json`).

See [`claude.md`](claude.md) for the architecture rationale.

## Layout

| path | what |
|------|------|
| `grace/config.py` | model presets + shared `TrainConfig` |
| `grace/modules.py` | RMSNorm, RoPE, SwiGLU, causal attention |
| `grace/model_baseline.py` | modern GPT baseline (RMSNorm + RoPE + SwiGLU) |
| `grace/model_grace.py` | GRACE: pool / depth-query / append DAG |
| `grace/data.py` | sliding-window dataset over a packed uint16 stream |
| `grace/train.py` | epoch-based trainer |
| `scripts/prepare_data.py` | download + tokenize + pack the corpus |
| `tests/` | shape / gradient / round-trip / causality property tests |

Both presets are ~50.4M params (matched to <0.4%); forward matmul FLOPs match to
within ~10% (GRACE's depth-attention overhead).

## Setup

```bash
# CPU (local dev): pull torch from the CPU wheel index
uv add torch --index https://download.pytorch.org/whl/cpu
# on the L40S server instead: uv add torch   (default index = CUDA build)
uv add numpy tokenizers datasets tqdm
uv add --dev pytest
```

`uv` manages the project venv automatically — prefix commands with `uv run`.

## Data

Documents are concatenated (separated by `<eos>`) into one packed uint16 stream
per split; training examples are **sliding windows** of size `max_seq_len`
(1024) with overlap 256. Validation holds out a fixed number of tokens (whole
docs), not a fraction:

```bash
uv run python -m scripts.prepare_data --out-dir data --val-tokens 1000000
```

## Train

Only the model, seed, and (optional) output path are CLI args — every training
hyperparameter lives in `TrainConfig` (`grace/config.py`) so the two runs are
identical (edit it there):

```bash
uv run python -m grace.train --model baseline           # -> ckpt/baseline/0/
uv run python -m grace.train --model grace  --seed 1     # -> ckpt/grace/1/
```

Each run writes to `ckpt/<model>/<seed>/` (override the dir with `--out`; warns
up front if it already contains files):

| file | contents |
|------|----------|
| `metadata.json` | model + train config and param count |
| `record.jsonl` | one line per step: `step`, `epoch`, `train_loss`, `val_loss`, `time` |
| `epoch_{n}.pt` | checkpoint after each epoch (3 by default) |

Progress is a tqdm bar (loss/lr in the postfix). Validation loss is estimated
over a fixed 10 batches each epoch. Defaults: 3 epochs, bf16 on CUDA. Compare
validation loss (`record.jsonl`) and tokens/sec.

**GPU selection (shared servers):** by default the trainer queries `nvidia-smi`
and picks the freest idle GPU (< ~1 GB used), so it won't land on a card someone
else is training on. If every GPU is busy it aborts with a message rather than
interfering. Override by setting `TrainConfig.device` (`"cuda:1"`, `"cpu"`) or
the `CUDA_VISIBLE_DEVICES` env var, both of which are respected as-is.

## Test

```bash
uv run pytest -q
```

Tests assert **properties** — shapes, finite gradients, next-token round-trip,
causality (future tokens don't leak), zero-init query ⇒ uniform pool average,
distinct queries ⇒ distinct group inputs — rather than recomputing any einsum a
second way.
