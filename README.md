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
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build on the server
uv pip install numpy tokenizers pytest tqdm
```

## Data

Documents are concatenated (separated by `<eos>`) into one packed uint16 stream
per split; training examples are **sliding windows** of size `max_seq_len`
(1024) with overlap 256.

```bash
python -m scripts.prepare_data --out-dir data --val-frac 0.005
```

## Train

Only the model, seed, and (optional) output path are CLI args — every training
hyperparameter lives in `TrainConfig` (`grace/config.py`) so the two runs are
identical (edit it there):

```bash
python -m grace.train --model baseline           # -> ckpt/baseline/0/
python -m grace.train --model grace  --seed 1     # -> ckpt/grace/1/
```

Each run writes to `ckpt/<model>/<seed>/` (override the dir with `--out`; warns
up front if it already contains files):

| file | contents |
|------|----------|
| `metadata.json` | model + train config and param count |
| `record.jsonl` | one line per step: `step`, `epoch`, `train_loss`, `val_loss`, `time` |
| `epoch_{n}.pt` | checkpoint after each epoch (3 by default) |

Progress is a tqdm bar (loss/lr in the postfix). Defaults: 3 epochs, bf16 on
CUDA. Compare validation loss (`record.jsonl`) and tokens/sec.

## Test

```bash
pytest -q
```

Tests assert **properties** — shapes, finite gradients, next-token round-trip,
causality (future tokens don't leak), zero-init query ⇒ uniform pool average,
distinct queries ⇒ distinct group inputs — rather than recomputing any einsum a
second way.
