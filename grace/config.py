"""Model configurations for the baseline transformer and GRACE.

Both presets are tuned to ~50M parameters. Because total matmul FLOPs for a
dense LM are ~= 2 * (non-embedding params) * tokens (the "6ND" rule), matching
non-embedding parameter count also matches FLOPs to within GRACE's small
depth-attention overhead (2 * G * pool * d per token per layer, a few percent).
Use ``count_params`` / ``estimate_flops_per_token`` in ``grace.utils`` to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaselineConfig:
    """A modern GPT-style decoder (RMSNorm + RoPE + SwiGLU, pre-norm residual)."""

    vocab_size: int = 8192
    d_model: int = 512
    n_layer: int = 14
    n_head: int = 8
    d_ff: int = 1472  # SwiGLU inner width
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass
class GraceConfig:
    """GRACE: homogeneous alternating layers, ``G`` parallel groups per layer,
    each group with a zero-init query attending over a pool of all prior block
    outputs; outputs append to the pool instead of merging.

    ``layer_types`` lists one entry per layer, each ``"attn"`` or ``"mlp"``.
    Default is strict alternation starting (and, for odd counts, ending) with
    attention, as described in ``claude.md``.
    """

    vocab_size: int = 8192
    d_model: int = 512
    groups: int = 4  # G — parallel groups per layer
    n_head: int = 8  # heads *per group* for attn layers
    d_ff: int = 1024  # SwiGLU inner width for mlp layers
    layer_types: list[str] = field(
        default_factory=lambda: ["attn", "mlp", "attn", "mlp", "attn", "mlp", "attn", "mlp", "attn"]
    )
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert all(t in ("attn", "mlp") for t in self.layer_types), "layer_types must be 'attn'/'mlp'"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_layer(self) -> int:
        return len(self.layer_types)

    @property
    def max_pool(self) -> int:
        """Pool entries: 1 (token embedding) + G per layer."""
        return 1 + self.groups * self.n_layer


@dataclass
class TrainConfig:
    """Training hyperparameters shared by BOTH models.

    These are deliberately not CLI flags: the baseline vs. GRACE comparison is
    only fair if every training knob is identical, so they live here as one
    source of truth. Only the model choice (baseline/grace) and the checkpoint
    path vary between the two runs.
    """

    data_dir: str = "data"
    epochs: int = 3
    batch_size: int = 32
    overlap: int = 256  # sliding-window overlap (window size = model max_seq_len)
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup: int = 0  # 0 => auto (2% of total steps)
    device: Optional[str] = None  # None => cuda if available else cpu
    seed: int = 1337


PRESETS: dict[str, object] = {
    "baseline_50m": BaselineConfig(),
    "grace_50m": GraceConfig(),
    # Tiny configs used by the test-suite for fast CPU verification.
    "baseline_tiny": BaselineConfig(
        vocab_size=64, d_model=32, n_layer=3, n_head=4, d_ff=48, max_seq_len=32
    ),
    "grace_tiny": GraceConfig(
        vocab_size=64,
        d_model=32,
        groups=3,
        n_head=4,
        d_ff=48,
        layer_types=["attn", "mlp", "attn"],
        max_seq_len=32,
    ),
}
