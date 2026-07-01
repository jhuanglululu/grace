"""GRACE: shape, gradient flow (incl. depth queries), pool growth, causality,
and the two mechanism properties from claude.md:

  * zero-init query  => depth-attention is a uniform average of the pool
  * distinct queries => distinct composed inputs (the anti-collapse property)
"""

import torch
import torch.nn.functional as F

from grace.config import PRESETS
from grace.model_grace import GraceTransformer, depth_attention
from grace.modules import rms_normalize

torch.manual_seed(0)


def _model():
    return GraceTransformer(PRESETS["grace_tiny"])


def test_forward_shape():
    m = _model()
    idx = torch.randint(0, m.cfg.vocab_size, (3, 16))
    assert m(idx).shape == (3, 16, m.cfg.vocab_size)


def test_all_params_get_finite_grads_including_queries():
    m = _model()
    idx = torch.randint(0, m.cfg.vocab_size, (2, 16))
    loss = F.cross_entropy(m(idx).reshape(-1, m.cfg.vocab_size), idx.reshape(-1))
    loss.backward()
    query_grad_norm = 0.0
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if name.endswith("query"):
            query_grad_norm += p.grad.norm().item()
    # queries are the mechanism; they must actually receive gradient signal
    # even though they start at zero (uniform softmax still has a Jacobian).
    assert query_grad_norm > 0


def test_pool_grows_by_G_per_layer():
    # The pool starts with the embedding (1 entry) and appends G per layer.
    m = _model()
    cfg = m.cfg
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    x = m.embed(idx)
    pool = rms_normalize(x).unsqueeze(2)  # entries are normalized at append time
    cos, sin = m._rope(0, 12, idx.device, x.dtype)
    sizes = [pool.shape[2]]
    for layer in m.layers:
        out = layer(pool, cos, sin)
        assert out.shape == (2, 12, cfg.groups, cfg.d_model)
        pool = torch.cat([pool, rms_normalize(out)], dim=2)
        sizes.append(pool.shape[2])
    assert sizes == [1 + cfg.groups * i for i in range(cfg.n_layer + 1)]
    assert pool.shape[2] == cfg.max_pool


def test_zero_query_is_uniform_average_of_pool():
    # With a zero query, softmax over the pool is uniform, so the composed input
    # is exactly the mean of the pool entries (which are pre-normalized at append).
    G, d = 4, 32
    pool = torch.randn(2, 5, 7, d)  # (B, T, P, d)
    composed = depth_attention(torch.zeros(G, d), pool)  # (B, T, G, d)
    expected = pool.mean(dim=2, keepdim=True).expand(-1, -1, G, -1)
    assert torch.allclose(composed, expected, atol=1e-5)


def test_distinct_queries_give_distinct_inputs():
    # The anti-collapse property: different group queries compose different
    # inputs from the same pool, so the groups are genuinely different functions.
    d = 32
    pool = torch.randn(1, 3, 9, d)
    q = torch.stack([torch.randn(d) * 3, torch.randn(d) * 3])  # 2 distinct queries
    composed = depth_attention(q, pool)  # (1, 3, 2, d)
    assert not torch.allclose(composed[:, :, 0], composed[:, :, 1])


def test_buffered_infer_matches_cat_path():
    # The no-grad preallocated-buffer path must equal the autograd cat path.
    m = GraceTransformer(PRESETS["grace_tiny"]).eval()
    idx = torch.randint(0, m.cfg.vocab_size, (2, 16))
    with torch.enable_grad():
        cat_logits = m(idx)  # grad enabled -> _forward_cat
    with torch.no_grad():
        buf_logits = m(idx)  # grad disabled -> _forward_buffered
    assert torch.allclose(cat_logits.detach(), buf_logits, atol=1e-5)


def test_causality_future_tokens_do_not_leak():
    m = _model().eval()
    T, j = 16, 8
    idx = torch.randint(0, m.cfg.vocab_size, (1, T))
    with torch.no_grad():
        base = m(idx)
        idx2 = idx.clone()
        idx2[0, j] = (idx2[0, j] + 1) % m.cfg.vocab_size
        perturbed = m(idx2)
    assert torch.allclose(base[:, :j], perturbed[:, :j], atol=1e-5)
    assert not torch.allclose(base[:, j], perturbed[:, j])
