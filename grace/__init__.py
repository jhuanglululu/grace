"""GRACE — Grouped Residual Attention over Composed dEpth, plus a matched baseline.

See ``claude.md`` at the repo root for the architecture rationale.
"""

from .config import BaselineConfig, GraceConfig, PRESETS
from .model_baseline import BaselineTransformer
from .model_grace import GraceTransformer

__all__ = [
    "BaselineConfig",
    "GraceConfig",
    "PRESETS",
    "BaselineTransformer",
    "GraceTransformer",
]
