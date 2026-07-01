"""Sanity: both presets land near 50M params, and their FLOPs are comparable."""

from grace.config import PRESETS
from grace.model_baseline import BaselineTransformer
from grace.model_grace import GraceTransformer
from grace.utils import baseline_flops_per_token, count_params, grace_flops_per_token


def test_both_presets_near_50m():
    base = count_params(BaselineTransformer(PRESETS["baseline_50m"]))
    grace = count_params(GraceTransformer(PRESETS["grace_50m"]))
    for n, p in (("baseline", base), ("grace", grace)):
        assert 45e6 <= p <= 55e6, f"{n} has {p:,} params, expected ~50M"
    # the two models should be matched in size to within ~10%
    assert abs(base - grace) / 50e6 < 0.10


def test_flops_are_comparable():
    bcfg, gcfg = PRESETS["baseline_50m"], PRESETS["grace_50m"]
    T = 1024
    bflops = baseline_flops_per_token(bcfg, T)
    gflops = grace_flops_per_token(gcfg, T)
    ratio = gflops / bflops
    assert 0.7 <= ratio <= 1.4, f"FLOP ratio grace/baseline = {ratio:.2f} out of range"
