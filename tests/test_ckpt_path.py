"""Run-directory resolution: default layout vs. explicit override."""

from grace.train import resolve_run_dir


def test_default_layout_by_model_and_seed():
    assert resolve_run_dir("baseline", 0) == "ckpt/baseline/0"
    assert resolve_run_dir("grace", 7) == "ckpt/grace/7"


def test_explicit_out_overrides():
    assert resolve_run_dir("grace", 3, out="/tmp/myrun") == "/tmp/myrun"
