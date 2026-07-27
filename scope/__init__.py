"""SCOPE: On-policy distillation with OPD+ST.

Training dependencies are imported lazily so lightweight preprocessing helpers can
be used without loading the complete TRL stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SCOPETrainer", "compute_opd_st_loss"]


def __getattr__(name: str) -> Any:
    if name == "SCOPETrainer":
        from scope.trainer import SCOPETrainer

        return SCOPETrainer
    if name == "compute_opd_st_loss":
        from scope.opd_st_loss import compute_opd_st_loss

        return compute_opd_st_loss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
