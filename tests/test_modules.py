"""Property tests for the shared building blocks (shape / norm / RoPE geometry).

These assert *properties* (unit RMS, rotation preserves norm, RoPE relative-
position invariance) rather than recomputing the ops a second way.
"""

import torch

from grace.modules import RMSNorm, SwiGLU, apply_rope, build_rope_cache

torch.manual_seed(0)


def test_rmsnorm_is_unit_rms():
    d = 64
    norm = RMSNorm(d)  # weight initialised to ones
    x = torch.randn(4, 10, d) * 5.0 + 2.0
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm():
    B, H, T, hd = 2, 3, 16, 32
    cos, sin = build_rope_cache(T, hd, 10000.0, "cpu", torch.float32)
    x = torch.randn(B, H, T, hd)
    xr = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), atol=1e-4)


def test_rope_relative_position_invariance():
    # <rope(q, m), rope(k, n)> depends only on (m - n): shifting both positions
    # by the same amount leaves the score unchanged.
    hd, T = 32, 16
    cos, sin = build_rope_cache(T, hd, 10000.0, "cpu", torch.float32)
    q = torch.randn(hd)
    k = torch.randn(hd)

    def score(m, n):
        qm = apply_rope(q.view(1, 1, hd), cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k.view(1, 1, hd), cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum().item()

    assert abs(score(5, 2) - score(8, 5)) < 1e-4  # both have gap 3
    assert abs(score(10, 4) - score(7, 1)) < 1e-4  # both have gap 6


def test_swiglu_shape():
    mlp = SwiGLU(48, 128)
    x = torch.randn(2, 7, 48)
    assert mlp(x).shape == (2, 7, 48)
