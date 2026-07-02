"""Generation: rep-penalty math, checkpoint save->load round-trip, and
deterministic/length-bounded sampling."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict  # noqa: E402

from grace.config import PRESETS  # noqa: E402
from grace.model_grace import GraceTransformer  # noqa: E402
from scripts.generate import apply_rep_pen, generate_ids, load_model  # noqa: E402


def test_rep_pen_pushes_seen_tokens_down():
    logits = torch.tensor([2.0, -2.0, 0.5, 3.0])
    out = apply_rep_pen(logits, prev_ids=[0, 1, 3], pen=2.0)
    assert out[0].item() == 1.0     # positive: 2 / 2
    assert out[3].item() == 1.5     # positive: 3 / 2
    assert out[1].item() == -4.0    # negative: -2 * 2 (more negative)
    assert out[2].item() == 0.5     # untouched (not in prev_ids)
    # a no-op penalty leaves logits unchanged
    assert torch.equal(apply_rep_pen(logits, [0, 1], 1.0), logits)


def test_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import save_file

    cfg = PRESETS["grace_tiny"]
    model = GraceTransformer(cfg).eval()
    # save model weights as safetensors (drop tied lm_head, like the trainer does)
    sd = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(sd, str(tmp_path / "step1.safetensors"))
    json.dump({"model": "grace", "model_config": asdict(cfg)}, open(tmp_path / "metadata.json", "w"))

    loaded, lcfg = load_model(str(tmp_path / "step1.safetensors"), "cpu")
    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        assert torch.allclose(model(idx), loaded(idx), atol=1e-6)  # same weights -> same logits


def test_generate_is_deterministic_and_bounded():
    model = GraceTransformer(PRESETS["grace_tiny"]).eval()
    prompt = [1, 2, 3]
    kw = dict(max_new_tokens=6, temperature=0.9, top_k=5, rep_pen=1.1, max_seq_len=32)

    torch.manual_seed(0)
    a = generate_ids(model, prompt, **kw)
    torch.manual_seed(0)
    b = generate_ids(model, prompt, **kw)

    assert a == b                      # same seed -> same sample
    assert len(a) == 6                 # exactly max_new_tokens, prompt excluded
    assert all(0 <= t < PRESETS["grace_tiny"].vocab_size for t in a)


def test_greedy_generation_needs_no_seed():
    model = GraceTransformer(PRESETS["grace_tiny"]).eval()
    kw = dict(max_new_tokens=5, temperature=0.0, top_k=0, rep_pen=1.0, max_seq_len=32)
    assert generate_ids(model, [1, 2], **kw) == generate_ids(model, [1, 2], **kw)
