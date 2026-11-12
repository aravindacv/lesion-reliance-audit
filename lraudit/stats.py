"""Statistics used in the paper: bootstrap intervals and paired tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from .audit import macro_auroc


def bootstrap_auroc(y_onehot, p, n_boot: int = 1000, seed: int = 0, classes=None):
    """Point estimate and 95% percentile interval of the macro AUROC."""
    rng = np.random.default_rng(seed)
    n = len(y_onehot)
    est = macro_auroc(y_onehot, p, classes)
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            reps.append(macro_auroc(y_onehot[i], p[i], classes))
        except ValueError:
            continue
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return est, lo, hi


def paired_bootstrap_diff(y_onehot, p_a, p_b, n_boot: int = 500, seed: int = 0, classes=None):
    """Macro AUROC of A minus B with a paired 95% percentile interval."""
    rng = np.random.default_rng(seed)
    n = len(y_onehot)
    est = macro_auroc(y_onehot, p_a, classes) - macro_auroc(y_onehot, p_b, classes)
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        reps.append(macro_auroc(y_onehot[i], p_a[i], classes) - macro_auroc(y_onehot[i], p_b[i], classes))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return est, lo, hi


def er_vs_blob(per_image):
    """Wilcoxon signed-rank test of model ER against centre-blob ER (two-sided)."""
    d = per_image[["ER", "ER_blob"]].dropna()
    return wilcoxon(d.ER, d.ER_blob)


def rare_vs_common(per_image, column: str = "ER"):
    """Mann-Whitney U test of a score between rare and common images (two-sided)."""
    a = per_image.loc[per_image.rare, column].dropna()
    b = per_image.loc[~per_image.rare, column].dropna()
    return mannwhitneyu(a, b, alternative="two-sided")
