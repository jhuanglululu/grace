"""Load a trained checkpoint and generate text from a prompt.

The model architecture/config is read from ``metadata.json`` in the checkpoint's
run directory (written by the trainer), so only the checkpoint path is needed.

Usage (either form works):
    uv run scripts/generate.py --ckpt-path ckpt/grace2/0/last.safetensors --prompt "台灣" --rep-pen 1.2
    uv run python -m scripts.generate --ckpt-path ckpt/grace2/0/last.safetensors
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

from grace.config import TrainConfig
from grace.tokenizer import GraceTokenizer
from grace.train import model_family, resolve_device, seed_all


def config_from_metadata(meta: dict):
    """Rebuild the (ModelClass, config) pair from a run's metadata.json.
    Kind dispatch lives in grace.train.model_family — one source of truth."""
    cfg_cls, model_cls = model_family(meta["model"])
    return model_cls, cfg_cls(**meta["model_config"])


def load_model(ckpt_path: str, device: str = "cpu"):
    from safetensors.torch import load_file

    meta_path = os.path.join(os.path.dirname(ckpt_path), "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    model_cls, cfg = config_from_metadata(meta)
    model = model_cls(cfg)
    flat = load_file(ckpt_path, device="cpu")
    # last.safetensors carries extra optimizer/RNG tensors that are not model params.
    state = {k: v for k, v in flat.items() if not k.startswith(("opt.", "rng."))}
    # Only the tied lm_head may legitimately be absent from the file (dropped on
    # save, re-tied in __init__); anything else means a mismatched checkpoint.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {"lm_head.weight"} or unexpected:
        raise ValueError(
            f"checkpoint {ckpt_path!r} does not match a {meta['model']!r} model: "
            f"missing keys {sorted(set(missing) - {'lm_head.weight'})}, "
            f"unexpected keys {sorted(unexpected)}"
        )
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


def _sample(
    logits: torch.Tensor, prev_ids, temperature: float, top_k: int, rep_pen: float
) -> int:
    logits = (
        logits.float()
    )  # sample in fp32 regardless of model dtype (8k vocab — cheap)
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


def _seed_list(arg: str) -> list[int]:
    """argparse type for --seed: one int or a comma-separated list of ints."""
    try:
        return [int(s) for s in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an int or comma-separated ints (e.g. 0,42,67), got {arg!r}"
        )


def _rate(tokens: int, seconds: float) -> float:
    return tokens / max(seconds, 1e-9)


@torch.no_grad()
def generate_ids(
    model,
    prompt_ids,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    rep_pen: float = 1.1,
    max_seq_len: int = 1024,
    device: str = "cpu",
    return_timing: bool = False,
    warn: bool = True,
):
    """Autoregressively sample using a KV cache; returns the new ids (and, if
    ``return_timing``, a stats dict with separate prefill/decode/sample timings).

    The prompt is prefilled in one pass, then each subsequent token is a single-
    token forward against the growing cache (O(1) attention per step). No decode
    forward is run after the final token, so the timing is not skewed by wasted work.
    A sampled EOS does NOT stop generation — this is a benchmark, and every run
    producing exactly ``max_new_tokens`` keeps tok/s comparable across models/seeds.
    """
    ids = list(prompt_ids)
    if len(ids) >= max_seq_len:  # keep room to generate at least one token
        ids = ids[-(max_seq_len - 1) :]
        if warn:
            print(
                f"warning: prompt truncated to its last {len(ids)} tokens "
                f"(max_seq_len={max_seq_len})",
                file=sys.stderr,
            )
    room = max_seq_len - len(ids)
    if warn and room < max_new_tokens:
        print(
            f"warning: context window leaves room for only {room} of "
            f"{max_new_tokens} requested tokens",
            file=sys.stderr,
        )
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
    sample_time = 0.0
    while len(new) < max_new_tokens and pos < max_seq_len:
        # int() inside _sample forces a device sync, so this timing is honest.
        t0 = time.perf_counter()
        nxt = _sample(logits, ids, temperature, top_k, rep_pen)
        sample_time += time.perf_counter() - t0
        ids.append(nxt)
        new.append(nxt)
        if len(new) >= max_new_tokens:
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
            "new_tokens": len(new),
            "sample_time": sample_time,
        }
        return new, stats
    return new


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    p.add_argument(
        "--ckpt-path",
        required=True,
        help="path to a .safetensors checkpoint (metadata.json must sit beside it)",
    )
    p.add_argument(
        "--prompt",
        default="",
        help="conditioning text (empty => start from a doc boundary)",
    )
    p.add_argument(
        "--rep-pen", type=float, default=1.1, help="repetition penalty (1.0 = off)"
    )
    p.add_argument(
        "--seed",
        type=_seed_list,
        default=[0],
        help="random seed, or comma-separated list (e.g. 0,42,67): each seed "
        "generates once from the same loaded/quantized/compiled model, "
        "with per-seed stats plus a pooled average",
    )
    p.add_argument(
        "--show-output",
        action="store_true",
        help="print the generated text (default: statistics only)",
    )
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 => greedy")
    p.add_argument("--top-k", type=int, default=50, help="0 => no top-k")
    p.add_argument(
        "--device",
        default=None,
        help="default: auto-pick a free GPU (see train.resolve_device)",
    )
    p.add_argument(
        "--compile",
        choices=["auto", "on", "off"],
        default="auto",
        help="torch.compile the model to fuse kernels (auto = on for CUDA)",
    )
    p.add_argument(
        "--dtype",
        choices=["f32", "f16", "bf16", "int8"],
        default="f32",
        help="weight/activation precision; halves KV-cache traffic below f32. "
        "int8 = torchao weight-only quant over bf16 activations (CUDA recommended)",
    )
    args = p.parse_args()

    seeds = args.seed
    device = resolve_device(TrainConfig(device=args.device))
    if device.startswith("cuda") and ":" in device:
        torch.cuda.set_device(device)
    # TF32 tensor cores for fp32 matmuls (the --dtype f32 path; no effect on
    # f16/bf16/int8). Same setting as training, applied to both models equally.
    torch.set_float32_matmul_precision("high")

    tok = GraceTokenizer()
    model, cfg = load_model(args.ckpt_path, device)

    # Cast BEFORE compile so kernels are generated for the final dtype. The KV
    # cache and pool buffer inherit the activation dtype, so f16/bf16 also halve
    # cache traffic. RMSNorm computes in fp32 internally either way.
    if args.dtype in ("f16", "bf16"):
        model = model.to(torch.float16 if args.dtype == "f16" else torch.bfloat16)
    elif args.dtype == "int8":
        # weight-only int8 (per-channel scales) on bf16 activations; the tied
        # embedding stays bf16 (torchao only rewrites nn.Linear weights).
        # version=2 (Int8Tensor): version 1's AffineQuantizedTensor is deprecated
        # (github.com/pytorch/ao/issues/2752); PerRow == v1's per-channel scales.
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        model = model.to(torch.bfloat16)
        quantize_(model, Int8WeightOnlyConfig(version=2))

    # Fuse the many small ops (norms, softmax, rope, depth-attention) into few
    # kernels. This is where GRACE's fewer-layers advantage shows up: eager launch
    # overhead otherwise dominates at batch-1. Applied to both models so the
    # comparison is architecture, not compile-vs-eager. The warmup below absorbs
    # the one-time compile cost.
    do_compile = args.compile == "on" or (
        args.compile == "auto" and device.startswith("cuda")
    )
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
        temperature=args.temperature,
        top_k=args.top_k,
        rep_pen=args.rep_pen,
        max_seq_len=cfg.max_seq_len,
        device=device,
    )

    # Warm up with a full untimed replica of a timed run — identical sampling
    # args and length, not a shortened pass. Every one-time cost (compile,
    # dynamic-shape recompiles at KV-cache lengths a short warmup never reaches,
    # allocator pool growth, kernel autotune) must land here, or it pollutes the
    # first seed's numbers and skews the pooled average. Seeding with seeds[0]
    # makes the warmup a bit-identical rehearsal of the first timed run.
    seed_all(seeds[0], device)
    _ = generate_ids(
        model, prompt_ids, max_new_tokens=args.max_new_tokens, warn=False, **common
    )

    # One generation per seed, re-seeding before each so every --seed value
    # determines its own output; the model stays loaded/quantized/compiled.
    totals = {
        "prefill_tokens": 0,
        "prefill_time": 0.0,
        "decode_tokens": 0,
        "decode_time": 0.0,
        "new_tokens": 0,
        "sample_time": 0.0,
    }
    for i, seed in enumerate(seeds):
        seed_all(seed, device)
        gen_ids, s = generate_ids(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            return_timing=True,
            warn=(i == 0),
            **common,
        )
        for k in totals:
            totals[k] += s[k]

        if len(seeds) > 1:
            if i:
                print()
            print(f"--- seed {seed} ---")
        if args.show_output:
            print(args.prompt + tok.decode(gen_ids) + "\n")
        pt, dt, st = s["prefill_time"], s["decode_time"], s["sample_time"]
        print(
            f"[prefill: {s['prefill_tokens']} tok in {pt * 1e3:.1f}ms = "
            f"{_rate(s['prefill_tokens'], pt):.1f} tok/s]"
        )
        print(
            f"[decode:  {s['decode_tokens']} tok in {dt:.3f}s = "
            f"{_rate(s['decode_tokens'], dt):.1f} tok/s on {device}, {args.dtype} (model forward only)]"
        )
        print(
            f"[end-to-end: {s['new_tokens']} tok in {dt + st:.3f}s = "
            f"{_rate(s['new_tokens'], dt + st):.1f} tok/s (forward + sampling)]"
        )

    if len(seeds) > 1:
        # Pooled (total tokens / total time), not a mean of rates — robust to
        # runs of different lengths (a prompt near max_seq_len leaves less room).
        pt, dt, st = (
            totals["prefill_time"],
            totals["decode_time"],
            totals["sample_time"],
        )
        print(
            f"\n[average over {len(seeds)} seeds (pooled): "
            f"prefill {_rate(totals['prefill_tokens'], pt):.1f} tok/s | "
            f"decode {_rate(totals['decode_tokens'], dt):.1f} tok/s | "
            f"end-to-end {_rate(totals['new_tokens'], dt + st):.1f} tok/s]"
        )


if __name__ == "__main__":
    main()
