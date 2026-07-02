"""Sanity: the trio stays block-matched (see grace/config.py docstring) —
identical block dimensions, essentially identical sizes, comparable FLOPs."""

import pytest

from grace.config import PRESETS
from grace.model_baseline import BaselineTransformer
from grace.model_grace import GraceTransformer
from grace.utils import baseline_flops_per_token, count_params, grace_flops_per_token

# Derived from PRESETS so a new grace preset is covered without editing this file.
GRACE_KINDS = tuple(k for k in PRESETS if k.startswith("grace") and not k.endswith("_tiny"))


def test_presets_shape_matched():
    b = PRESETS["baseline"]
    for kind in GRACE_KINDS:
        g = PRESETS[kind]
        assert b.d_model == g.d_model
        assert b.d_ff == g.d_ff
        assert b.n_head == g.n_head
        # same 24 blocks: baseline has 2 sublayers per layer, grace G per layer
        assert g.groups * g.n_layer == 2 * b.n_layer


def test_preset_sizes():
    base = count_params(BaselineTransformer(PRESETS["baseline"]))
    assert 40e6 <= base <= 50e6, f"baseline has {base:,} params, expected ~44M"
    for kind in GRACE_KINDS:
        grace = count_params(GraceTransformer(PRESETS[kind]))
        # each grace variant is the baseline's 24 blocks re-wired, differing
        # only in queries and norm bookkeeping — sizes essentially identical
        assert abs(grace - base) / base < 0.005, (
            f"{kind} has {grace:,} params vs baseline {base:,}, expected <0.5% match"
        )


@pytest.mark.parametrize("kind", GRACE_KINDS)
def test_flops_are_comparable(kind):
    T = 1024
    bflops = baseline_flops_per_token(PRESETS["baseline"], T)
    gflops = grace_flops_per_token(PRESETS[kind], T)
    ratio = gflops / bflops
    assert 0.95 <= ratio <= 1.05, f"FLOP ratio {kind}/baseline = {ratio:.3f} out of range"
