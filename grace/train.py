"""Epoch-based trainer for the baseline and GRACE models.

All shared hyperparameters live in ``TrainConfig`` (``grace/config.py``) so the
two models train under identical settings — the only CLI choices are which model
to train, the seed, and (optionally) where to write the run:

    python -m grace.train --model baseline           # -> ckpt/baseline/0/
    python -m grace.train --model grace --seed 1      # -> ckpt/grace/1/

Each run writes to ``ckpt/<model>/<seed>/`` (override the dir with --out):
    metadata.json       train + model config and param count
    record.jsonl        one line per logged step: step, epoch, train/val loss, time
    step{N}.safetensors the top-3 checkpoints by validation loss (model weights)
    best.json           the ranked top-3 (step, val_loss, file)
    last.safetensors    always-latest full state (model + optimizer + RNG) for resume

Resume an interrupted run with ``--resume`` (loads last.safetensors, continues at
epoch granularity). Runs on the remote L40S in bf16; also runs on CPU (fp32).
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
from safetensors.torch import save_file
from tqdm import tqdm

from .config import PRESETS, BaselineConfig, GraceConfig, TrainConfig
from .data import WindowedDataset
from .model_baseline import BaselineTransformer
from .model_grace import GraceTransformer
from .utils import count_params

# Fixed training mechanics — never swept, so constants rather than TrainConfig
# fields (kept out of metadata.json).
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.02  # warmup steps = 2% of total
VAL_EVERY = 500  # run validation every N optimizer steps
TOP_K_CKPTS = 3  # keep the best-K checkpoints by validation loss


def _model_tensors(model) -> dict:
    """Model weights for a safetensors checkpoint (drop the tied lm_head, which
    duplicates embed.weight and would trip safetensors' shared-tensor check)."""
    return {
        k: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
        if k != "lm_head.weight"
    }


def _optim_tensors(opt) -> dict:
    out = {}
    for i, st in opt.state_dict()["state"].items():
        for name, v in st.items():
            t = v if torch.is_tensor(v) else torch.tensor(v)
            out[f"opt.{i}.{name}"] = t.detach().cpu().contiguous()
    return out


def _rng_tensors(gen, device) -> dict:
    out = {"rng.torch": torch.get_rng_state(), "rng.gen": gen.get_state()}
    if device.startswith("cuda"):
        out["rng.cuda"] = torch.cuda.get_rng_state()
    return out


def load_resume(run_dir: str, model, opt, gen, device: str) -> int:
    """Restore model + optimizer + RNG from last.safetensors; return the epoch to
    resume at. Raises FileNotFoundError if there's nothing to resume from."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = os.path.join(run_dir, "last.safetensors")
    flat = load_file(path, device="cpu")
    model.load_state_dict(
        {k: v for k, v in flat.items() if not k.startswith(("opt.", "rng."))},
        strict=False,
    )
    model.to(device)
    state: dict = {}
    for k, v in flat.items():
        if k.startswith("opt."):
            _, i, name = k.split(".", 2)
            state.setdefault(int(i), {})[name] = v.to(device) if name != "step" else v
    sd = opt.state_dict()
    sd["state"] = state
    opt.load_state_dict(sd)
    torch.set_rng_state(flat["rng.torch"])
    gen.set_state(flat["rng.gen"])
    if device.startswith("cuda") and "rng.cuda" in flat:
        torch.cuda.set_rng_state(flat["rng.cuda"])
    with safe_open(path, framework="pt") as f:
        return int(f.metadata()["resume_epoch"])


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


def build_optimizer(model, tcfg: TrainConfig):
    """AdamW with weight decay only on >=2-D matmul weights. RMSNorm scales
    (1-D) and the zero-init depth queries are excluded: decaying the queries
    toward zero would fight the very differentiation that keeps GRACE's parallel
    groups from collapsing (see claude.md)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("query") or name.endswith("readout_query"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": tcfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tcfg.lr, betas=(0.9, 0.95))


def cosine_lr(
    step: int, total: int, base: float, warmup: int, min_ratio: float = 0.1
) -> float:
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
        stats.append(
            {
                "index": int(idx),
                "mem_used": int(used),
                "mem_total": int(total),
                "util": int(util),
            }
        )
    return stats


def query_gpu_stats() -> list[dict]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
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


def train(
    model_kind: str, tcfg: TrainConfig, out: str | None = None, resume: bool = False
):
    device = resolve_device(tcfg)
    if device.startswith("cuda") and ":" in device:
        torch.cuda.set_device(device)  # pin the chosen GPU for all allocations
    # Seed BEFORE build_model so weight init (nn.init.normal_) is reproducible per seed.
    torch.manual_seed(tcfg.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(tcfg.seed)
    model, cfg = build_model(model_kind)
    model.to(device)
    model.compile()
    n_params = count_params(model)

    run_dir = resolve_run_dir(model_kind, tcfg.seed, out)
    os.makedirs(run_dir, exist_ok=True)
    resuming = resume and os.path.exists(os.path.join(run_dir, "last.safetensors"))
    if os.path.isdir(run_dir) and any(os.scandir(run_dir)) and not resuming:
        print(
            f"WARNING: run dir {run_dir} already contains files; they may be overwritten"
        )
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "model": model_kind,
                "params": n_params,
                "device": device,
                "model_config": asdict(cfg),
                "train_config": asdict(tcfg),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"model={model_kind} params={n_params:,} device={device} run_dir={run_dir}")

    train_ds = WindowedDataset(
        os.path.join(tcfg.data_dir, "train.bin"), cfg.max_seq_len, tcfg.overlap
    )
    val_path = os.path.join(tcfg.data_dir, "val.bin")
    val_ds = (
        WindowedDataset(val_path, cfg.max_seq_len, tcfg.overlap)
        if os.path.exists(val_path)
        else None
    )

    n_windows = len(train_ds)
    steps_per_epoch = math.ceil(
        n_windows / tcfg.batch_size
    )  # one optimizer step per batch
    total_steps = max(1, steps_per_epoch * tcfg.epochs)
    warmup = max(1, int(WARMUP_FRAC * total_steps))

    opt = build_optimizer(model, tcfg)
    use_amp = device.startswith("cuda")
    gen = torch.Generator().manual_seed(
        tcfg.seed
    )  # dataset-shuffle RNG (independent of global seed)

    # best_ckpts: (val_loss, step, path) sorted ascending, len <= TOP_K_CKPTS.
    best_ckpts: list = []
    start_epoch = 0
    if resuming:
        start_epoch = load_resume(run_dir, model, opt, gen, device)
        best_path = os.path.join(run_dir, "best.json")
        if os.path.exists(best_path):
            best_ckpts = [
                (r["val_loss"], r["step"], os.path.join(run_dir, r["file"]))
                for r in json.load(open(best_path))
            ]
        print(f"resumed from last.safetensors at epoch {start_epoch}")

    # Append to the record on resume, truncate on a fresh run.
    record_f = open(os.path.join(run_dir, "record.jsonl"), "a" if resuming else "w")

    def record(**kw):
        record_f.write(json.dumps(kw) + "\n")

    model.train()
    t0 = time.time()
    step = start_epoch * steps_per_epoch  # LR schedule position (epoch-granular resume)
    last_val: float | None = None

    def optimizer_step() -> float:
        nonlocal step
        lr = cosine_lr(step, total_steps, tcfg.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        return lr

    def postfix(pbar, lr):
        p = {"loss": f"{last_loss:.3f}", "lr": f"{lr:.1e}"}
        if last_val is not None:
            p["val"] = f"{last_val:.3f}"
        pbar.set_postfix(**p)

    def save_last(resume_epoch: int):
        """Always-overwritten checkpoint with full state for resume-from-last."""
        tensors = {
            **_model_tensors(model),
            **_optim_tensors(opt),
            **_rng_tensors(gen, device),
        }
        meta = {
            "step": str(step),
            "resume_epoch": str(resume_epoch),
            "model": model_kind,
        }
        if last_val is not None:
            meta["val_loss"] = f"{last_val:.6f}"
        save_file(tensors, os.path.join(run_dir, "last.safetensors"), metadata=meta)

    def save_best(epoch: int):
        """Keep the TOP_K_CKPTS lowest-val-loss checkpoints (model weights only)."""
        if last_val is None:
            return
        if len(best_ckpts) >= TOP_K_CKPTS and last_val >= best_ckpts[-1][0]:
            return  # not better than the worst kept
        path = os.path.join(run_dir, f"step{step}.safetensors")
        if any(p == path for _, _, p in best_ckpts):
            return  # already saved this exact step
        save_file(
            _model_tensors(model),
            path,
            metadata={
                "step": str(step),
                "epoch": str(epoch),
                "val_loss": f"{last_val:.6f}",
                "model": model_kind,
            },
        )
        best_ckpts.append((last_val, step, path))
        best_ckpts.sort(key=lambda r: r[0])
        while len(best_ckpts) > TOP_K_CKPTS:
            _, _, evict = best_ckpts.pop()
            if os.path.exists(evict):
                os.remove(evict)
        with open(os.path.join(run_dir, "best.json"), "w") as f:
            json.dump(
                [
                    {
                        "rank": i + 1,
                        "step": s,
                        "val_loss": vl,
                        "file": os.path.basename(p),
                    }
                    for i, (vl, s, p) in enumerate(best_ckpts)
                ],
                f,
                indent=2,
            )

    def checkpoint(epoch: int, resume_epoch: int):
        save_best(epoch)  # top-K by val loss (model only)
        save_last(resume_epoch)  # always: latest full state for resume
        record_f.flush()

    try:
        for epoch in range(start_epoch, tcfg.epochs):
            last_loss = float("nan")
            opt.zero_grad(set_to_none=True)
            pbar = tqdm(
                train_ds.iter_epoch(tcfg.batch_size, shuffle=True, generator=gen),
                total=steps_per_epoch,
                desc=f"epoch {epoch + 1}/{tcfg.epochs}",
            )
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                ctx = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_amp
                    else _nullctx()
                )
                with ctx:
                    loss = loss_fn(model(x), y)
                loss.backward()
                last_loss = loss.item()
                lr = optimizer_step()
                did_val = val_ds is not None and step % VAL_EVERY == 0
                if did_val:
                    last_val = evaluate(model, val_ds, tcfg.batch_size, device)
                record(
                    step=step,
                    epoch=epoch,
                    train_loss=last_loss,
                    val_loss=last_val if did_val else None,
                    time=time.time() - t0,
                )
                if did_val:
                    checkpoint(
                        epoch, resume_epoch=epoch
                    )  # mid-epoch -> resume restarts this epoch
                postfix(pbar, lr)
            pbar.close()

            # End-of-epoch validation + checkpoint (resume anchor = next epoch).
            if val_ds is not None:
                last_val = evaluate(model, val_ds, tcfg.batch_size, device)
            record(
                step=step,
                epoch=epoch,
                train_loss=last_loss,
                val_loss=last_val,
                time=time.time() - t0,
            )
            checkpoint(epoch, resume_epoch=epoch + 1)
            msg = f"epoch {epoch + 1}/{tcfg.epochs} done | train {last_loss:.4f}"
            if last_val is not None:
                msg += f" | val {last_val:.4f}"
            tqdm.write(f"{msg} | saved last.safetensors + top-{TOP_K_CKPTS}")
    finally:
        record_f.close()
    print(f"done. run_dir={run_dir}")


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    p = argparse.ArgumentParser(
        description="Train the baseline or GRACE model (shared TrainConfig)."
    )
    p.add_argument("--model", choices=["baseline", "grace"], required=True)
    p.add_argument(
        "--out", default=None, help="run directory (default ckpt/<model>/<seed>/)"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed (for training multiple models)"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume from last.safetensors in the run dir",
    )
    args = p.parse_args()
    train(
        args.model,
        out=args.out,
        tcfg=replace(TrainConfig(), seed=args.seed),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
