"""lraudit: interventional audit of lesion reliance in medical image classifiers."""
from .audit import AuditConfig, AuditResult, audit, macro_auroc
from .erasure import background_control, blur_fill, dilate, inpaint, skin_fill
from .saliency import GradCAM, centre_blob, energy_ratio, pointing_game
from .stats import bootstrap_auroc, er_vs_blob, paired_bootstrap_diff, rare_vs_common

__version__ = "0.1.0"
__all__ = [
    "audit", "AuditConfig", "AuditResult", "macro_auroc",
    "GradCAM", "energy_ratio", "pointing_game", "centre_blob",
    "inpaint", "skin_fill", "blur_fill", "dilate", "background_control",
    "bootstrap_auroc", "paired_bootstrap_diff", "er_vs_blob", "rare_vs_common",
]
