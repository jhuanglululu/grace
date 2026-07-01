"""End-to-end smoke test: the trainer runs a full (tiny) 2-epoch run and writes
its artifacts (metadata.json, record.jsonl, one checkpoint per epoch).

Guards against integration bugs across data -> model -> loss -> optimizer -> IO.
"""

import json
import os

import numpy as np
import pytest

from grace import config
from grace.config import TrainConfig
from grace.data import write_bin
from grace.train import train


def _write_data(tmp_path, vocab=64, n=6000):
    rng = np.random.default_rng(0)
    write_bin(str(tmp_path / "train.bin"), rng.integers(0, vocab, size=n))
    write_bin(str(tmp_path / "val.bin"), rng.integers(0, vocab, size=n // 4))


@pytest.mark.parametrize("model", ["baseline", "grace"])
def test_train_writes_run_artifacts(tmp_path, monkeypatch, model):
    _write_data(tmp_path)
    # Point the "<model>_50m" preset at the tiny config so the run is CPU-fast.
    monkeypatch.setitem(config.PRESETS, f"{model}_50m", config.PRESETS[f"{model}_tiny"])
    run_dir = str(tmp_path / "run")
    epochs = 2
    tcfg = TrainConfig(
        data_dir=str(tmp_path),
        epochs=epochs,
        batch_size=4,
        overlap=8,
        grad_accum=2,
        device="cpu",
        seed=0,
    )
    train(model, out=run_dir, tcfg=tcfg)

    # metadata.json carries both configs and is valid JSON
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    assert meta["model"] == model
    assert "model_config" in meta and "train_config" in meta

    # one checkpoint per epoch
    for e in range(1, epochs + 1):
        assert os.path.exists(os.path.join(run_dir, f"epoch_{e}.pt"))

    # record.jsonl parses and contains per-step train losses and per-epoch val losses
    lines = [json.loads(x) for x in open(os.path.join(run_dir, "record.jsonl"))]
    assert lines, "record.jsonl is empty"
    assert all({"step", "epoch", "train_loss", "val_loss", "time"} <= set(r) for r in lines)
    assert any(r["train_loss"] is not None for r in lines)
    assert sum(1 for r in lines if r["val_loss"] is not None) == epochs
