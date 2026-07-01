"""GRACE — Grouped Residual Attention over Composed dEpth.

Homogeneous alternating layers (all-attn / all-mlp). Each layer has ``G``
parallel groups fused as a leading axis. Each group owns a zero-init *depth
query* that attends over a **pool** of all previous block outputs (kept
separate, not summed); its composed input is the attention-weighted sum of the
pool. The G group outputs are appended to the pool. The per-group query is what
stops the parallel groups from collapsing into one wide block.

See ``claude.md`` for the full rationale. Shapes:
  B batch, T seq, d d_model, G groups, P current pool size, H heads/group, hd head_dim.

Note on the pool buffer: ``claude.md`` prescribes a preallocated contiguous
buffer with in-place slice writes (an MLX-inference fusion concern). For this
correctness-first PyTorch training reference we grow the pool with ``cat``,
which is autograd-safe and clearer; the preallocated buffer is an inference-time
optimization left for the MLX port.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraceConfig
from .modules import KVCache, RMSNorm, apply_rope, attend, build_rope_cache, rms_normalize


def depth_attention(query: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """Grouped attention over the pool along the *depth* axis (not the seq axis).

    query: (G, d) learned per-group depth queries.
    pool:  (B, T, P, d) all prior block outputs (raw; normalized here as K/V).
    returns composed inputs (B, T, G, d).

    With a zero query the softmax is uniform, so the composed input is the mean
    of the RMS-normalized pool entries.
    """
    d = query.shape[-1]
    kv = rms_normalize(pool)  # (B, T, P, d) — K and V share the normalized pool
    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("gd,btpd->btgp", query, kv) * scale  # (B, T, G, P)
    attn = torch.softmax(scores, dim=-1)
    composed = torch.einsum("btgp,btpd->btgd", attn, kv)  # (B, T, G, d)
    return composed


class GroupedAttnLayer(nn.Module):
    """One homogeneous attention layer: G parallel causal-attention blocks, each
    reading its own composed input from the pool."""

    def __init__(self, cfg: GraceConfig):
        super().__init__()
        self.G, self.H, self.hd = cfg.groups, cfg.n_head, cfg.head_dim
        d = cfg.d_model
        self.query = nn.Parameter(torch.zeros(self.G, d))  # zero-init depth query
        self.in_norm = RMSNorm(d)
        # Per-group (block-diagonal) projections — a leading group axis.
        self.Wq = nn.Parameter(torch.empty(self.G, d, d))
        self.Wk = nn.Parameter(torch.empty(self.G, d, d))
        self.Wv = nn.Parameter(torch.empty(self.G, d, d))
        self.Wo = nn.Parameter(torch.empty(self.G, d, d))
        for w in (self.Wq, self.Wk, self.Wv, self.Wo):
            nn.init.normal_(w, mean=0.0, std=0.02)

    def forward(self, pool, cos, sin, kv: KVCache | None = None):
        B, T = pool.shape[0], pool.shape[1]
        x = self.in_norm(depth_attention(self.query, pool))  # (B, T, G, d)
        q = torch.einsum("btgd,gde->btge", x, self.Wq)
        k = torch.einsum("btgd,gde->btge", x, self.Wk)
        v = torch.einsum("btgd,gde->btge", x, self.Wv)
        # (B, T, G, d) -> (B, G*H, T, hd) for batched attention (group as outer head).
        def heads(t):
            return t.view(B, T, self.G, self.H, self.hd).permute(0, 2, 3, 1, 4).reshape(
                B, self.G * self.H, T, self.hd
            )
        q, k, v = heads(q), heads(k), heads(v)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv is not None:
            k, v = kv.update(k, v)
        out = attend(q, k, v)  # (B, G*H, T, hd)
        out = out.view(B, self.G, self.H, T, self.hd).permute(0, 3, 1, 2, 4).reshape(
            B, T, self.G, self.H * self.hd
        )
        return torch.einsum("btgd,gde->btge", out, self.Wo)  # (B, T, G, d)


class GroupedMlpLayer(nn.Module):
    """One homogeneous MLP layer: G parallel SwiGLU blocks over composed inputs."""

    def __init__(self, cfg: GraceConfig):
        super().__init__()
        self.G = cfg.groups
        d, dff = cfg.d_model, cfg.d_ff
        self.query = nn.Parameter(torch.zeros(self.G, d))  # zero-init depth query
        self.in_norm = RMSNorm(d)
        self.Wg = nn.Parameter(torch.empty(self.G, d, dff))
        self.Wu = nn.Parameter(torch.empty(self.G, d, dff))
        self.Wd = nn.Parameter(torch.empty(self.G, dff, d))
        for w in (self.Wg, self.Wu, self.Wd):
            nn.init.normal_(w, mean=0.0, std=0.02)

    def forward(self, pool, cos, sin, kv: KVCache | None = None):  # kv unused (no cross-token state)
        x = self.in_norm(depth_attention(self.query, pool))  # (B, T, G, d)
        gate = torch.einsum("btgd,gdf->btgf", x, self.Wg)
        up = torch.einsum("btgd,gdf->btgf", x, self.Wu)
        h = F.silu(gate) * up
        return torch.einsum("btgf,gfd->btgd", h, self.Wd)  # (B, T, G, d)


class GraceTransformer(nn.Module):
    def __init__(self, cfg: GraceConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            GroupedAttnLayer(cfg) if t == "attn" else GroupedMlpLayer(cfg)
            for t in cfg.layer_types
        )
        # Learned read-out query composes the full pool into the final hidden.
        self.readout_query = nn.Parameter(torch.zeros(1, cfg.d_model))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if not cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self._rope_cache: tuple | None = None

    def _rope(self, start_pos, T, device, dtype):
        if self._rope_cache is None or self._rope_cache[0].device != device:
            cos, sin = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
            self._rope_cache = (cos, sin)
        cos, sin = self._rope_cache
        return cos[start_pos : start_pos + T], sin[start_pos : start_pos + T]

    def init_kv_cache(self) -> list[KVCache | None]:
        # Only attention layers hold a cache; mlp layers have no cross-token state.
        return [KVCache() if isinstance(layer, GroupedAttnLayer) else None for layer in self.layers]

    def forward(self, idx: torch.Tensor, caches: list | None = None, start_pos: int = 0) -> torch.Tensor:
        B, T = idx.shape
        x = self.embed(idx)  # (B, T, d)
        cos, sin = self._rope(start_pos, T, idx.device, x.dtype)
        # The pool (depth-attention) is per-token, so each forward builds a fresh
        # pool for just this chunk; cross-token state lives only in the attn KVCaches.
        pool = x.unsqueeze(2)  # (B, T, 1, d) — token embedding is the first pool entry
        for i, layer in enumerate(self.layers):
            kv = caches[i] if caches is not None else None
            out = layer(pool, cos, sin, kv)  # (B, T, G, d)
            pool = torch.cat([pool, out], dim=2)  # append G entries
        final = depth_attention(self.readout_query, pool).squeeze(2)  # (B, T, d)
        final = self.final_norm(final)
        return self.lm_head(final)
