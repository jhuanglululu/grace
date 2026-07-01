"""Epoch-based trainer for the baseline and GRACE models.

All shared hyperparameters live in ``TrainConfig`` (``grace/config.py``) so the
two models train under identical settings — the only CLI choices are which model
to train, the seed, and (optionally) where to write the run:

    python -m grace.train --model baseline           # -> ckpt/baseline/0/
    python -m grace.train --model grace --seed 1      # -> ckpt/grace/1/

Each run writes to ``ckpt/<model>/<seed>/`` (override the dir with --out):
    metadata.json   train + model config and param count
    record.jsonl    one line per logged step: step, epoch, train/val loss, time
    epoch_{n}.pt    a checkpoint after every epoch (3 by default)

Runs on the remote L40S in bf16; also runs on CPU (fp32) for the tiny configs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, replace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import PRESETS, BaselineConfig, GraceConfig, TrainConfig
from .data import WindowedDataset
from .model_baseline import BaselineTransformer
from .model_grace import GraceTransformer
from .utils import count_params


def resolve_run_dir(model_kind: str, seed: int, out: str | None = None) -> str:
    """Directory holding a run's artifacts. An explicit ``out`` wins, otherwise
    runs are organised as ``ckpt/<model>/<seed>/`` so seeds/models don't clash."""
    if out:
        return out
    return os.path.join("ckpt", model_kind, str(seed))


def build_model(kind: str):
    preset = f"{kind}_50m"
    cfg = PRESETS[preset]
    if kind == "baseline":
        assert isinstance(cfg, BaselineConfig)
        return BaselineTransformer(cfg), cfg
    if kind == "grace":
        assert isinstance(cfg, GraceConfig)
        return GraceTransformer(cfg), cfg
    raise ValueError(kind)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def cosine_lr(step: int, total: int, base: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return base * min_ratio
    frac = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))


# A GPU using less than this many MB is treated as free (idle cards still hold a
# few hundred MB). Below this we assume nobody else is on it.
GPU_FREE_MEM_MB = 1024


def _parse_gpu_stats(csv_text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu
    --format=csv,noheader,nounits` output into a list of dicts."""
    stats = []
    for line in csv_text.strip().splitlines():
        if not line.strip():
            continue
        idx, used, total, util = (p.strip() for p in line.split(","))
        stats.append({"index": int(idx), "mem_used": int(used), "mem_total": int(total), "util": int(util)})
    return stats


def query_gpu_stats() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return _parse_gpu_stats(out)


def pick_free_gpu(stats: list[dict], free_mem_mb: int = GPU_FREE_MEM_MB):
    """Return the index of the freest idle GPU (least memory used, then least
    utilization), or None if every GPU is already in use."""
    free = [g for g in stats if g["mem_used"] < free_mem_mb]
    if not free:
        return None
    free.sort(key=lambda g: (g["mem_used"], g["util"]))
    return free[0]["index"]


def resolve_device(tcfg: TrainConfig) -> str:
    if tcfg.device is not None:  # explicit override wins (e.g. "cpu", "cuda:2")
        return tcfg.device
    if not torch.cuda.is_available():
        return "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):  # respect an externally pinned GPU
        return "cuda"
    stats = query_gpu_stats()
    if not stats:  # CUDA present but nvidia-smi unavailable — let torch choose
        print("nvidia-smi unavailable; using default cuda device")
        return "cuda"
    idx = pick_free_gpu(stats)
    if idx is None:
        busy = ", ".join(f"cuda:{g['index']}({g['mem_used']}MB)" for g in stats)
        raise RuntimeError(
            f"No free GPU: all in use (>= {GPU_FREE_MEM_MB}MB) [{busy}]. "
            f"Set TrainConfig.device to override (e.g. 'cuda:1' or 'cpu')."
        )
    used = next(g["mem_used"] for g in stats if g["index"] == idx)
    print(f"selected free GPU cuda:{idx} ({used}MB used of {len(stats)} GPUs)")
    return f"cuda:{idx}"


# Validation loss is estimated over a fixed number of batches (not a fraction of
# the val set) so eval cost is constant regardless of corpus size.
VAL_BATCHES = 10


@torch.no_grad()
def evaluate(model, val_ds: WindowedDataset, batch_size: int, device: str):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_ds.iter_epoch(batch_size)):
        if i >= VAL_BATCHES:
            break
        x, y = x.to(device), y.to(device)
        losses.append(loss_fn(model(x), y).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(model_kind: str, out: str | None = None, tcfg: TrainConfig | None = None):
    tcfg = tcfg or TrainConfig()
    device = resolve_device(tcfg)
    if device.startswith("cuda") and ":" in device:
        torch.cuda.set_device(device)  # pin the chosen GPU for all allocations
    model, cfg = build_model(model_kind)
    model.to(device)
    n_params = count_params(model)

    run_dir = resolve_run_dir(model_kind, tcfg.seed, out)
    if os.path.isdir(run_dir) and any(os.scandir(run_dir)):  # warn early, before a long run
        print(f"WARNING: run dir {run_dir} already contains files; they may be overwritten")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(
            {"model": model_kind, "params": n_params, "device": device,
             "model_config": asdict(cfg), "train_config": asdict(tcfg)},
            f, indent=2, ensure_ascii=False,
        )
    print(f"model={model_kind} params={n_params:,} device={device} run_dir={run_dir}")

    train_ds = WindowedDataset(os.path.join(tcfg.data_dir, "train.bin"), cfg.max_seq_len, tcfg.overlap)
    val_path = os.path.join(tcfg.data_dir, "val.bin")
    val_ds = WindowedDataset(val_path, cfg.max_seq_len, tcfg.overlap) if os.path.exists(val_path) else None

    n_windows = len(train_ds)
    batches_per_epoch = math.ceil(n_windows / tcfg.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / tcfg.grad_accum)
    total_steps = max(1, steps_per_epoch * tcfg.epochs)
    warmup = tcfg.warmup or max(1, int(0.02 * total_steps))

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, betas=(0.9, 0.95), weight_decay=tcfg.weight_decay)
    use_amp = device.startswith("cuda")
    gen = torch.Generator().manual_seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)

    record_f = open(os.path.join(run_dir, "record.jsonl"), "w")

    def record(**kw):
        record_f.write(json.dumps(kw) + "\n")

    model.train()
    t0 = time.time()
    step = 0
    last_val: float | None = None

    def optimizer_step() -> float:
        nonlocal step
        lr = cosine_lr(step, total_steps, tcfg.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        return lr

    def postfix(pbar, lr):
        p = {"loss": f"{last_loss:.3f}", "lr": f"{lr:.1e}"}
        if last_val is not None:
            p["val"] = f"{last_val:.3f}"
        pbar.set_postfix(**p)

    try:
        for epoch in range(tcfg.epochs):
            micro = 0
            last_loss = float("nan")
            opt.zero_grad(set_to_none=True)
            pbar = tqdm(
                train_ds.iter_epoch(tcfg.batch_size, shuffle=True, generator=gen),
                total=batches_per_epoch,
                desc=f"epoch {epoch + 1}/{tcfg.epochs}",
            )
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
                with ctx:
                    loss = loss_fn(model(x), y) / tcfg.grad_accum
                loss.backward()
                last_loss = loss.item() * tcfg.grad_accum
                micro += 1
                if micro % tcfg.grad_accum == 0:
                    lr = optimizer_step()
                    did_val = val_ds is not None and step % tcfg.val_every == 0
                    if did_val:
                        last_val = evaluate(model, val_ds, tcfg.batch_size, device)
                    record(step=step, epoch=epoch, train_loss=last_loss,
                           val_loss=last_val if did_val else None, time=time.time() - t0)
                    postfix(pbar, lr)
            if micro % tcfg.grad_accum != 0:  # flush trailing partial accumulation
                optimizer_step()
            pbar.close()

            # End-of-epoch validation + checkpoint.
            if val_ds is not None:
                last_val = evaluate(model, val_ds, tcfg.batch_size, device)
            record(step=step, epoch=epoch, train_loss=last_loss, val_loss=last_val, time=time.time() - t0)
            record_f.flush()
            ckpt = os.path.join(run_dir, f"epoch_{epoch + 1}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "step": step}, ckpt)
            msg = f"epoch {epoch + 1}/{tcfg.epochs} done | train {last_loss:.4f}"
            if last_val is not None:
                msg += f" | val {last_val:.4f}"
            tqdm.write(f"{msg} | saved {ckpt}")
    finally:
        record_f.close()
    print(f"done. run_dir={run_dir}")


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    p = argparse.ArgumentParser(description="Train the baseline or GRACE model (shared TrainConfig).")
    p.add_argument("--model", choices=["baseline", "grace"], required=True)
    p.add_argument("--out", default=None, help="run directory (default ckpt/<model>/<seed>/)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (for training multiple models)")
    args = p.parse_args()
    train(args.model, out=args.out, tcfg=replace(TrainConfig(), seed=args.seed))


if __name__ == "__main__":
    main()
