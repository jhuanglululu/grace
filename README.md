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

The presets are **block-matched**: all three models are built from the same 24
blocks (12 attn + 12 mlp, d_model=512, d_ff=1472) and differ only in wiring —
a **flattening sweep** (24 divides by 2 and 3):

| preset | wiring | depth |
|--------|--------|-------|
| `baseline` | 24 blocks chained sequentially | 12 layers × [attn, mlp] |
| `grace2` | flattened 2-wide | 12 alternating layers × 2 groups |
| `grace3` | flattened 3-wide | 8 alternating layers × 3 groups |

Each GRACE variant adds only its zero-init depth-attention queries, so all
three are ~44M params (matched to <0.05%) with FLOPs/token within ~1% — the
sweep isolates how quality and decode speed change as sequential depth is
traded for parallel width. See the `grace/config.py` docstring for the
rationale and the earlier ~50M parameter-matched round.

## Setup

```bash
# CPU (local dev): pull torch from the CPU wheel index
uv add torch --index https://download.pytorch.org/whl/cpu
# on the L40S server instead: uv add torch   (default index = CUDA build)
uv add numpy tokenizers datasets tqdm
uv add torchao   # weight-only int8 for `generate.py --dtype int8`
uv add --dev pytest
```

`uv` manages the project venv automatically — prefix commands with `uv run`.

## Data

Documents are concatenated (separated by `<eos>`) into one packed uint16 stream
per split; training examples are **sliding windows** of size `max_seq_len`
(1024) with overlap 256. Validation holds out a fixed number of tokens (whole
docs), not a fraction:

```bash
uv run scripts/prepare_data.py --out-dir data --val-tokens 1000000
```

## Train

Only the model, seed, and (optional) output path are CLI args — every training
hyperparameter lives in `TrainConfig` (`grace/config.py`) so the two runs are
identical (edit it there):

```bash
uv run python -m grace.train --model baseline            # -> ckpt/baseline/0/
uv run python -m grace.train --model grace2               # -> ckpt/grace2/0/
uv run python -m grace.train --model grace3 --seed 1      # -> ckpt/grace3/1/
```

Each run writes to `ckpt/<model>/<seed>/` (override the dir with `--out`):

| file | contents |
|------|----------|
| `metadata.json` | model + train config and param count |
| `record.jsonl` | one line per step: `step`, `epoch`, `train_loss`, `val_loss`, `time` |
| `step{N}.safetensors` | the **top-3 checkpoints by validation loss** (model weights) |
| `best.json` | the ranked top-3 (`step`, `val_loss`, `file`) |
| `last.safetensors` | always-latest **full state** (model + optimizer + RNG) for resume |

Checkpoints are safetensors (the tied LM head is dropped and re-tied on load).
Resume an interrupted run with `--resume` (loads `last.safetensors`, continues at
epoch granularity):

```bash
uv run python -m grace.train --model grace2 --resume
```

Progress is a tqdm bar (train loss, lr, and latest val loss in the postfix).
Validation runs every `val_every` steps (default 500) and at each epoch end,
estimated over a fixed 10 batches, and is logged to `record.jsonl`. Defaults:
3 epochs, bf16 on CUDA. Compare validation loss and tokens/sec.

**GPU selection (shared servers):** by default the trainer queries `nvidia-smi`
and picks the freest idle GPU (< ~1 GB used), so it won't land on a card someone
else is training on. If every GPU is busy it aborts with a message rather than
interfering. Override by setting `TrainConfig.device` (`"cuda:1"`, `"cpu"`) or
the `CUDA_VISIBLE_DEVICES` env var, both of which are respected as-is.

## Generate

Sample from a checkpoint — the architecture is read from the run's
`metadata.json` (must sit beside the checkpoint), so only the checkpoint path is
needed. Point it at a `best.json`-listed checkpoint or `last.safetensors`:

```bash
uv run scripts/generate.py --ckpt-path ckpt/grace2/0/last.safetensors \
    --prompt "台灣" --rep-pen 1.2 --seed 0
```

Generation uses a **KV cache** (prefill the prompt, then O(1)-attention single-
token steps). It runs an untimed **warmup** pass first (so kernel autotune /
allocation don't skew the numbers), re-seeds, then reports **prefill and decode
tok/s separately** — prefill is parallel prompt ingestion, decode is sequential
single-token throughput:

```
[prefill: 16 tok in 26.1ms = 612.8 tok/s]
[decode:  49 tok in 0.645s = 75.9 tok/s on cuda:0]
```

By default `--device` auto-picks a free GPU (same `nvidia-smi` logic as
training); pass `--device cpu`/`cuda:1` to override.

Flags: `--prompt` (empty starts from a document boundary), `--rep-pen`
(repetition penalty, 1.0 = off), `--seed` (one seed or a comma-separated list
like `0,42,67` — each generates once from the same loaded/quantized/compiled
model, printing per-seed stats and a pooled tok/s average), `--show-output`
(print the generated text too; default is statistics only), plus
`--temperature`, `--top-k`,
`--max-new-tokens`, `--device`, `--compile` (auto/on/off — torch.compile both
models; auto = on for CUDA, and the biggest lever on the decode numbers), and
`--dtype` (`f32` default | `f16` | `bf16` | `int8`). Below-f32 dtypes shrink
weight and KV-cache traffic — the main bottleneck at batch-1 — and sampling
always runs in fp32. `int8` is torchao weight-only quantization over bf16
activations (CUDA recommended; the tied embedding stays bf16). Checkpoints
themselves are always saved fp32.

## Test

```bash
uv run pytest -q
```

Tests assert **properties** — shapes, finite gradients, next-token round-trip,
causality (future tokens don't leak), zero-init query ⇒ uniform pool average,
distinct queries ⇒ distinct group inputs — rather than recomputing any einsum a
second way.
