"""Parameter counting and a rough matmul-FLOP estimate for the two models."""

from __future__ import annotations

import torch.nn as nn

from .config import BaselineConfig, GraceConfig


def count_params(model: nn.Module, non_embedding: bool = False) -> int:
    total = sum(p.numel() for p in model.parameters())
    if non_embedding:
        # The LM head is always tied to this table, so it's counted once here.
        emb = getattr(model, "embed", None)
        if emb is not None:
            total -= emb.weight.numel()
    return total


def baseline_flops_per_token(cfg: BaselineConfig, seq_len: int) -> int:
    """Forward matmul FLOPs per token (2 per MAC). Ignores norms/softmax."""
    d = cfg.d_model
    per_layer = 2 * 4 * d * d  # qkv + out proj
    per_layer += 2 * 3 * d * cfg.d_ff  # SwiGLU
    per_layer += 2 * 2 * seq_len * d  # attn scores + context (avg over positions ~ seq_len)
    flops = cfg.n_layer * per_layer
    flops += 2 * d * cfg.vocab_size  # lm head
    return flops


def grace_flops_per_token(cfg: GraceConfig, seq_len: int) -> int:
    d, G = cfg.d_model, cfg.groups
    pool = cfg.max_pool
    flops = 0
    for i, t in enumerate(cfg.layer_types):
        p = 1 + G * i  # pool size seen by this layer
        flops += 2 * 2 * G * p * d  # depth-attention (scores + weighted sum)
        if t == "attn":
            flops += 2 * 4 * G * d * d  # per-group qkv + out
            flops += 2 * 2 * G * seq_len * d  # per-group attn scores + context
        else:
            flops += 2 * 3 * G * d * cfg.d_ff  # per-group SwiGLU
    flops += 2 * 2 * pool * d  # read-out depth-attention
    flops += 2 * d * cfg.vocab_size  # lm head
    return flops
