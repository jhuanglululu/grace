"""Shared building blocks: RMSNorm, rotary embeddings, SwiGLU, causal attention.

Kept deliberately small and explicit. Grouped variants (a leading ``G`` group
axis) are used by GRACE; the baseline uses the plain variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms over last dim; compute in fp32 for stability then cast back.
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weightless RMS normalization (used for pool keys/values in GRACE)."""
    dtype = x.dtype
    x = x.float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * rms).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Return (cos, sin) each of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "rotary head_dim must be even"
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # (seq_len, half)
    # duplicate each frequency for the interleaved-half layout below
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding. ``x`` is (..., T, head_dim); cos/sin are (T, head_dim).

    This is a per-2D-subspace rotation, so it preserves the vector norm.
    """
    # broadcast cos/sin over all leading dims of x
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class KVCache:
    """Per-attention-layer key/value cache for incremental decoding.

    ``update`` appends the current step's keys/values (B, H, T, hd) and returns
    the full cached (k, v) so the query can attend over all past positions.
    """

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled-dot-product attention over (B, H, Tq, hd) queries and (B, H, Tk, hd)
    keys/values. Handles both a full causal pass (Tq == Tk) and single-step
    decoding against a longer cache (Tq == 1, attend all past)."""
    tq, tk = q.shape[2], k.shape[2]
    if tq == tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if tq == 1:  # decode step: the one new query attends every cached key
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)
    raise NotImplementedError("partial-chunk prefill into a non-empty cache is unsupported")
