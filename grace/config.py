"""Model configurations for the baseline transformer and GRACE.

The presets are **block-matched**: all three models are built from the same 24
blocks (12 attn + 12 mlp) with identical dimensions (d_model=512, d_ff=1472,
8 heads); they differ only in how those blocks are wired. 24 was chosen
because it divides by 2, 3, and 4, giving a **flattening sweep** at fixed blocks:

- ``baseline`` — the 24 blocks chained sequentially (12 layers x [attn, mlp]).
- ``grace2``   — the same stack flattened 2-wide: 12 alternating layers x 2 groups.
- ``grace3``   — flattened 3-wide: 8 alternating layers x 3 groups.
- ``grace4``   — flattened 4-wide: 6 alternating layers x 4 groups.

Each GRACE variant adds only its zero-init depth-attention queries (<0.05% of
params), so all three land at ~44M params matched to <0.05% with matmul
FLOPs/token within ~1%. Any quality difference across the sweep is purely the
topology — how much sequential depth was traded for parallel width. An earlier
round compared a single 4-wide GRACE against a 14-layer baseline at ~50M
parameter-matched (d_ff 1024 vs 1472); GRACE trailed by ~0.035 val loss, which
this round tests as a d_ff-capacity artifact while measuring how quality
scales with G.
Use ``count_params`` and ``baseline_flops_per_token`` /
``grace_flops_per_token`` in ``grace.utils`` to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Fixed architecture mechanic — identical for both models and never swept, so a
# constant (not a config field) and kept out of metadata.json. Embeddings are
# always tied to the LM head: it's baked into the model code (a necessity at this
# size), not a toggle.
ROPE_THETA = 10000.0


@dataclass
class BaselineConfig:
    """A modern GPT-style decoder (RMSNorm + RoPE + SwiGLU, pre-norm residual)."""

    vocab_size: int = 8192
    d_model: int = 512
    n_layer: int = 12
    n_head: int = 8
    d_ff: int = 1472  # SwiGLU inner width
    max_seq_len: int = 1024

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
    Default is strict alternation starting with attention and ending with mlp,
    as described in ``claude.md``.
    """

    vocab_size: int = 8192
    d_model: int = 512
    groups: int = 2  # G — parallel groups per layer
    n_head: int = 8  # heads *per group* for attn layers
    d_ff: int = 1472  # SwiGLU inner width for mlp layers (== baseline's, see module docstring)
    layer_types: list[str] = field(
        default_factory=lambda: ["attn", "mlp"] * 6
    )
    max_seq_len: int = 1024

    def __post_init__(self) -> None:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert all(t in ("attn", "mlp") for t in self.layer_types), (
            "layer_types must be 'attn'/'mlp'"
        )

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
    """Experiment knobs shared by BOTH models (dumped verbatim to metadata.json).

    These are deliberately not CLI flags: the baseline vs. GRACE comparison is
    only fair if every training knob is identical, so they live here as one
    source of truth. Only the model choice (baseline/grace), the seed, and the
    checkpoint path vary between runs. Fixed training mechanics that are never
    swept (grad clip, warmup fraction, validation cadence) are constants in
    ``train.py`` rather than fields here.
    """

    data_dir: str = "data"
    epochs: int = 3
    batch_size: int = 32
    overlap: int = 256  # sliding-window overlap (window size = model max_seq_len)
    lr: float = 3e-4
    weight_decay: float = 0.1
    device: Optional[str] = None  # None => cuda if available else cpu
    seed: int = 0


PRESETS: dict[str, object] = {
    "baseline": BaselineConfig(),  # 12 sequential layers = 24 sublayers
    "grace2": GraceConfig(),  # 2-wide: 12 layers x 2 groups = 24 blocks
    "grace3": GraceConfig(groups=3, layer_types=["attn", "mlp"] * 4),  # 3-wide: 8 layers x 3 groups
    "grace4": GraceConfig(groups=4, layer_types=["attn", "mlp"] * 3),  # 4-wide: 6 layers x 4 groups
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
