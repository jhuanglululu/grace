"""Load a trained checkpoint and generate text from a prompt.

The model architecture/config is read from ``metadata.json`` in the checkpoint's
run directory (written by the trainer), so only the checkpoint path is needed.

Usage (either form works):
    uv run scripts/generate.py --ckpt-path ckpt/grace/0/last.safetensors --prompt "台灣" --rep-pen 1.2
    uv run python -m scripts.generate --ckpt-path ckpt/grace/0/last.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running this file directly even when the `grace` package isn't installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from grace.config import BaselineConfig, GraceConfig, TrainConfig
from grace.model_baseline import BaselineTransformer
from grace.model_grace import GraceTransformer
from grace.tokenizer import GraceTokenizer
from grace.train import resolve_device

# Tokens generated in the untimed warmup pass (exercises prefill + decode kernels).
WARMUP_TOKENS = 8


def config_from_metadata(meta: dict):
    """Rebuild the (kind, config) pair from a run's metadata.json."""
    kind = meta["model"]
    mc = meta["model_config"]
    if kind == "baseline":
        return kind, BaselineConfig(**mc)
    if kind == "grace":
        return kind, GraceConfig(**mc)
    raise ValueError(f"unknown model kind {kind!r}")


def load_model(ckpt_path: str, device: str = "cpu"):
    from safetensors.torch import load_file

    meta_path = os.path.join(os.path.dirname(ckpt_path), "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    kind, cfg = config_from_metadata(meta)
    model = BaselineTransformer(cfg) if kind == "baseline" else GraceTransformer(cfg)
    flat = load_file(ckpt_path, device="cpu")
    # strict=False: lm_head is tied (absent from the file), and last.safetensors
    # carries extra optimizer/RNG tensors that are not model params.
    model.load_state_dict({k: v for k, v in flat.items() if not k.startswith(("opt.", "rng."))},
                          strict=False)
    model.to(device).eval()
    return model, cfg


def apply_rep_pen(logits: torch.Tensor, prev_ids, pen: float) -> torch.Tensor:
    """CTRL-style repetition penalty over already-seen tokens: divide positive
    logits by ``pen`` and multiply negative ones (both push the score down)."""
    if pen == 1.0 or not prev_ids:
        return logits
    ids = torch.tensor(sorted(set(prev_ids)), device=logits.device)
    vals = logits[ids]
    logits = logits.clone()
    logits[ids] = torch.where(vals > 0, vals / pen, vals * pen)
    return logits


def _sample(logits: torch.Tensor, prev_ids, temperature: float, top_k: int, rep_pen: float) -> int:
    logits = apply_rep_pen(logits, prev_ids, rep_pen)
    if temperature <= 0:  # greedy
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k)
        return int(idx[torch.multinomial(F.softmax(vals, dim=-1), 1)])
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def generate_ids(
    model,
    prompt_ids,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    rep_pen: float = 1.1,
    eos_id: int | None = None,
    max_seq_len: int = 1024,
    device: str = "cpu",
    return_timing: bool = False,
):
    """Autoregressively sample using a KV cache; returns the new ids (and, if
    ``return_timing``, a stats dict with separate prefill/decode timings).

    The prompt is prefilled in one pass, then each subsequent token is a single-
    token forward against the growing cache (O(1) attention per step). No decode
    forward is run after the final token, so the timing is not skewed by wasted work.
    """
    ids = list(prompt_ids)[-max_seq_len:]
    caches = model.init_kv_cache()

    ctx = torch.tensor([ids], dtype=torch.long, device=device)
    _sync(device)
    t0 = time.perf_counter()
    logits = model(ctx, caches, start_pos=0)[0, -1, :]  # prefill
    _sync(device)
    prefill_time = time.perf_counter() - t0
    prefill_tokens = len(ids)

    pos = len(ids)
    new: list[int] = []
    decode_time = 0.0
    decode_steps = 0
    while len(new) < max_new_tokens and pos < max_seq_len:
        nxt = _sample(logits, ids, temperature, top_k, rep_pen)
        ids.append(nxt)
        new.append(nxt)
        if (eos_id is not None and nxt == eos_id) or len(new) >= max_new_tokens:
            break  # don't run a decode step whose logits we'd never use
        tok = torch.tensor([[nxt]], dtype=torch.long, device=device)
        _sync(device)
        t0 = time.perf_counter()
        logits = model(tok, caches, start_pos=pos)[0, -1, :]  # one-token decode step
        _sync(device)
        decode_time += time.perf_counter() - t0
        decode_steps += 1
        pos += 1

    if return_timing:
        stats = {
            "prefill_tokens": prefill_tokens,
            "prefill_time": prefill_time,
            "decode_tokens": decode_steps,
            "decode_time": decode_time,
        }
        return new, stats
    return new


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    p.add_argument("--ckpt-path", required=True, help="path to a .safetensors checkpoint (metadata.json must sit beside it)")
    p.add_argument("--prompt", default="", help="conditioning text (empty => start from a doc boundary)")
    p.add_argument("--rep-pen", type=float, default=1.1, help="repetition penalty (1.0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 => greedy")
    p.add_argument("--top-k", type=int, default=50, help="0 => no top-k")
    p.add_argument("--device", default=None, help="default: auto-pick a free GPU (see train.resolve_device)")
    p.add_argument("--compile", choices=["auto", "on", "off"], default="auto",
                   help="torch.compile the model to fuse kernels (auto = on for CUDA)")
    args = p.parse_args()

    device = resolve_device(TrainConfig(device=args.device))
    if device.startswith("cuda") and ":" in device:
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    tok = GraceTokenizer()
    model, cfg = load_model(args.ckpt_path, device)

    # Fuse the many small ops (norms, softmax, rope, depth-attention) into few
    # kernels. This is where GRACE's fewer-layers advantage shows up: eager launch
    # overhead otherwise dominates at batch-1. Applied to both models so the
    # comparison is architecture, not compile-vs-eager. The warmup below absorbs
    # the one-time compile cost.
    do_compile = args.compile == "on" or (args.compile == "auto" and device.startswith("cuda"))
    if do_compile:
        # dynamic=True: compile once for a dynamic sequence/position rather than
        # relying on automatic-dynamic to engage as start_pos and the KV-cache
        # length grow each decode step (else the timed run could recompile).
        model = torch.compile(model, dynamic=True)
        print("torch.compile enabled (dynamic=True)")

    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:  # seed from a document boundary the model saw in training
        prompt_ids = [tok.eos_id if tok.eos_id is not None else (tok.bos_id or 0)]

    common = dict(
        temperature=args.temperature, top_k=args.top_k, rep_pen=args.rep_pen,
        eos_id=tok.eos_id, max_seq_len=cfg.max_seq_len, device=device,
    )

    # Warm up (compile + kernel autotune / allocation) so the timed run measures
    # steady state. Force greedy with no EOS stop so warmup ALWAYS runs the full
    # decode path (not cut short by an early EOS), guaranteeing the decode shape
    # is compiled before timing. This is a benchmark, not a chatbot — accuracy first.
    generate_ids(
        model, prompt_ids, max_new_tokens=WARMUP_TOKENS,
        temperature=0.0, top_k=0, rep_pen=1.0, eos_id=None,
        max_seq_len=cfg.max_seq_len, device=device,
    )

    # Re-seed after warmup so --seed still determines the actual output.
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    gen_ids, s = generate_ids(model, prompt_ids, max_new_tokens=args.max_new_tokens, return_timing=True, **common)

    print(args.prompt + tok.decode(gen_ids))
    pt, dt = s["prefill_time"], s["decode_time"]
    print(
        f"\n[prefill: {s['prefill_tokens']} tok in {pt * 1e3:.1f}ms = "
        f"{s['prefill_tokens'] / max(pt, 1e-9):.1f} tok/s]"
    )
    print(
        f"[decode:  {s['decode_tokens']} tok in {dt:.3f}s = "
        f"{s['decode_tokens'] / max(dt, 1e-9):.1f} tok/s on {device}]"
    )


if __name__ == "__main__":
    main()
