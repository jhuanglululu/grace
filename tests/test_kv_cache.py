"""KV cache correctness: incremental decoding must reproduce the full forward.

This is the key property — rather than re-deriving expected values, we assert
that per-step cached logits equal a single full-sequence forward pass.
"""

import pytest
import torch

from grace.config import PRESETS
from grace.model_baseline import BaselineTransformer
from grace.model_grace import GraceTransformer

torch.manual_seed(0)


def _models():
    return {
        "baseline": BaselineTransformer(PRESETS["baseline_tiny"]).eval(),
        "grace": GraceTransformer(PRESETS["grace_tiny"]).eval(),
    }


@pytest.mark.parametrize("name", ["baseline", "grace"])
def test_incremental_matches_full_forward(name):
    model = _models()[name]
    T = 16
    idx = torch.randint(0, model.cfg.vocab_size, (1, T))
    with torch.no_grad():
        full = model(idx)  # (1, T, vocab)
        caches = model.init_kv_cache()
        for t in range(T):
            step = model(idx[:, t : t + 1], caches, start_pos=t)  # (1, 1, vocab)
            assert torch.allclose(step[0, 0], full[0, t], atol=1e-4), f"mismatch at position {t}"


@pytest.mark.parametrize("name", ["baseline", "grace"])
def test_prefill_then_decode_matches_full(name):
    # Prefill a prompt in one pass, then decode the rest one token at a time.
    model = _models()[name]
    T, prefill = 16, 6
    idx = torch.randint(0, model.cfg.vocab_size, (1, T))
    with torch.no_grad():
        full = model(idx)
        caches = model.init_kv_cache()
        pre = model(idx[:, :prefill], caches, start_pos=0)
        assert torch.allclose(pre[0, -1], full[0, prefill - 1], atol=1e-4)
        for t in range(prefill, T):
            step = model(idx[:, t : t + 1], caches, start_pos=t)
            assert torch.allclose(step[0, 0], full[0, t], atol=1e-4), f"mismatch at position {t}"
