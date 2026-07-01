"""End-to-end smoke test: the trainer runs, writes safetensors artifacts
(top-3 best + last + best.json + metadata + record), and resumes from last.

Guards integration across data -> model -> loss -> optimizer -> checkpoint IO.
"""

import glob
import json
import os

import numpy as np
import pytest
from safetensors import safe_open

from grace import config
from grace.config import TrainConfig
from grace.data import write_bin
from grace.train import train


def _write_data(tmp_path, vocab=64, n=6000):
    rng = np.random.default_rng(0)
    write_bin(str(tmp_path / "train.bin"), rng.integers(0, vocab, size=n))
    write_bin(str(tmp_path / "val.bin"), rng.integers(0, vocab, size=n // 4))


def _tcfg(tmp_path, epochs):
    return TrainConfig(data_dir=str(tmp_path), epochs=epochs, batch_size=4,
                       overlap=8, device="cpu", seed=0)


@pytest.mark.parametrize("model", ["baseline", "grace"])
def test_train_writes_safetensors_artifacts(tmp_path, monkeypatch, model):
    _write_data(tmp_path)
    monkeypatch.setitem(config.PRESETS, f"{model}_50m", config.PRESETS[f"{model}_tiny"])
    run_dir = str(tmp_path / "run")
    train(model, tcfg=_tcfg(tmp_path, epochs=2), out=run_dir)

    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    assert meta["model"] == model

    # always-saved full-state checkpoint for resume
    assert os.path.exists(os.path.join(run_dir, "last.safetensors"))
    # top-K best (model weights), tracked in best.json, at most 3, and no .pt files
    best = json.load(open(os.path.join(run_dir, "best.json")))
    assert 1 <= len(best) <= 3
    steps = glob.glob(os.path.join(run_dir, "step*.safetensors"))
    assert len(steps) == len(best)
    assert not glob.glob(os.path.join(run_dir, "*.pt"))
    # best.json is sorted ascending by val loss
    assert [r["val_loss"] for r in best] == sorted(r["val_loss"] for r in best)

    # a best checkpoint holds model weights but NOT the tied head or optimizer state
    with safe_open(steps[0], framework="pt") as f:
        keys = set(f.keys())
    assert "embed.weight" in keys
    assert "lm_head.weight" not in keys
    assert not any(k.startswith("opt.") for k in keys)


def test_resume_continues_from_last(tmp_path, monkeypatch):
    _write_data(tmp_path)
    monkeypatch.setitem(config.PRESETS, "grace_50m", config.PRESETS["grace_tiny"])
    run_dir = str(tmp_path / "run")

    train("grace", tcfg=_tcfg(tmp_path, epochs=1), out=run_dir)
    with safe_open(os.path.join(run_dir, "last.safetensors"), framework="pt") as f:
        assert f.metadata()["resume_epoch"] == "1"  # epoch 0 completed

    # resume with a larger epoch budget: should start at epoch 1 and finish it
    train("grace", tcfg=_tcfg(tmp_path, epochs=2), out=run_dir, resume=True)
    with safe_open(os.path.join(run_dir, "last.safetensors"), framework="pt") as f:
        assert f.metadata()["resume_epoch"] == "2"
