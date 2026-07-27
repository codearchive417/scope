"""SCOPE: On-policy distillation with OPD+ST."""
from scope.trainer import SCOPETrainer
from scope.opd_st_loss import compute_opd_st_loss

__all__ = ["SCOPETrainer", "compute_opd_st_loss"]
