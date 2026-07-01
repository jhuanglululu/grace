"""Optimizer param groups: weight decay must skip the zero-init depth queries
and 1-D norm scales (decaying the queries would fight GRACE's anti-collapse)."""

import torch

from grace.config import PRESETS, TrainConfig
from grace.model_grace import GraceTransformer
from grace.train import build_optimizer


def test_queries_and_norms_are_excluded_from_weight_decay():
    torch.manual_seed(0)
    model = GraceTransformer(PRESETS["grace_tiny"])
    opt = build_optimizer(model, TrainConfig(weight_decay=0.1))

    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    for name, p in model.named_parameters():
        should_skip = p.ndim < 2 or name.endswith("query")
        assert (id(p) in no_decay_ids) == should_skip, name

    # sanity: at least one query and one norm weight actually landed in no-decay
    assert any(n.endswith("query") for n, _ in model.named_parameters())
    assert all(id(p) not in no_decay_ids for n, p in model.named_parameters()
               if n.endswith("Wq") or n.endswith("Wg"))  # matmul weights DO decay
