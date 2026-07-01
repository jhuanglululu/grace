"""Modern GPT-style baseline: pre-norm residual, RoPE attention, SwiGLU MLP.

This is the reference point GRACE is compared against at equal params/FLOPs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import BaselineConfig
from .modules import KVCache, RMSNorm, SwiGLU, apply_rope, attend, build_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv: KVCache | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, H, hd) -> (B, H, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv is not None:
            k, v = kv.update(k, v)
        out = attend(q, k, v)  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, kv: KVCache | None = None):
        x = x + self.attn(self.attn_norm(x), cos, sin, kv)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._rope_cache: tuple | None = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, start_pos, T, device, dtype):
        if self._rope_cache is None or self._rope_cache[0].device != device:
            cos, sin = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
            self._rope_cache = (cos, sin)
        cos, sin = self._rope_cache
        return cos[start_pos : start_pos + T], sin[start_pos : start_pos + T]

    def init_kv_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def forward(self, idx: torch.Tensor, caches: list[KVCache] | None = None, start_pos: int = 0) -> torch.Tensor:
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self._rope(start_pos, T, idx.device, x.dtype)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        x = self.final_norm(x)
        return self.lm_head(x)
