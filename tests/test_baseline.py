"""Baseline transformer: shape, gradient flow, and causality."""

import torch
import torch.nn.functional as F

from grace.config import PRESETS
from grace.model_baseline import BaselineTransformer

torch.manual_seed(0)


def _model():
    return BaselineTransformer(PRESETS["baseline_tiny"])


def test_forward_shape():
    m = _model()
    cfg = m.cfg
    idx = torch.randint(0, cfg.vocab_size, (3, 16))
    logits = m(idx)
    assert logits.shape == (3, 16, cfg.vocab_size)


def test_all_params_get_finite_grads():
    m = _model()
    idx = torch.randint(0, m.cfg.vocab_size, (2, 16))
    logits = m(idx)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_causality_future_tokens_do_not_leak(tmp_path):
    # Perturbing the token at position j must not change logits at positions < j.
    m = _model().eval()
    T, j = 16, 8
    idx = torch.randint(0, m.cfg.vocab_size, (1, T))
    with torch.no_grad():
        base = m(idx)
        idx2 = idx.clone()
        idx2[0, j] = (idx2[0, j] + 1) % m.cfg.vocab_size
        perturbed = m(idx2)
    assert torch.allclose(base[:, :j], perturbed[:, :j], atol=1e-5)
    assert not torch.allclose(base[:, j], perturbed[:, j])  # position j itself changes
