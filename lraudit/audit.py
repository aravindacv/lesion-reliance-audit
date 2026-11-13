
"""End-to-end lesion-reliance audit.

Given a classifier, images, lesion masks and labels, :func:`audit` runs the
three tests of the paper:

1. Localisation: Grad-CAM energy ratio and pointing game, compared with a
   uniform map (the mask area) and with the image-blind centre prior.
2. Interventional reliance: the Lesion Reliance Index
   LRI = p_c(background-erased) - p_c(lesion-erased).
3. Discrimination without the lesion: macro AUROC on lesion-erased images and
   the share of above-chance discrimination retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .erasure import ERASERS, background_control, dilate, inpaint
from .saliency import GradCAM, centre_blob, energy_ratio, pointing_game


@dataclass
class AuditConfig:
    blob_sigma: float = 40.0          # centre-prior width in pixels (at the network input size)
    lri_inpaint_radius: int = 5       # Telea radius for the LRI
    min_area: float = 0.02            # masks outside [min_area, max_area] are skipped for the LRI
    max_area: float = 0.60
    disc_radius: int = 12             # background-control discs
    disc_margin: int = 8
    erase_dilation: int = 7           # dilation before AUROC erasure
    erase_methods: Sequence[str] = ("inpaint", "skin_fill", "blur")
    activation: str = "sigmoid"       # "sigmoid" (multi-label) or "softmax"
    auroc_on_eligible: bool = True    # Test 3 on the images eligible for the LRI (as in the paper)
    batch_size: int = 32
    seed: int = 0


def _probs(model, preprocess, images, cfg, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(images), cfg.batch_size):
            x = torch.stack([preprocess(im) for im in images[i:i + cfg.batch_size]]).to(device)
            z = model(x)
            p = torch.sigmoid(z) if cfg.activation == "sigmoid" else torch.softmax(z, 1)
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def macro_auroc(y_onehot: np.ndarray, p: np.ndarray, classes: Optional[Sequence[int]] = None) -> float:
    ks = range(y_onehot.shape[1]) if classes is None else classes
    vals = [roc_auc_score(y_onehot[:, k], p[:, k]) for k in ks if 0 < y_onehot[:, k].sum() < len(y_onehot)]
    return float(np.mean(vals))


@dataclass
class AuditResult:
    per_image: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def __repr__(self):
        lines = ["Lesion-reliance audit"] + [f"  {k}: {v}" for k, v in self.summary.items()]
        return "\n".join(lines)


def audit(model: torch.nn.Module,
          target_layer: torch.nn.Module,
          images: Sequence[np.ndarray],
          masks: Sequence[np.ndarray],
          labels: Sequence[int],
          preprocess: Callable[[np.ndarray], torch.Tensor],
          n_classes: int,
          rare_classes: Sequence[int] = (),
          config: AuditConfig = AuditConfig(),
          channels_last: bool = False,
          device: Optional[str] = None) -> AuditResult:
    """Run the audit.

    ``images`` are uint8 RGB arrays already resized to the network input size,
    ``masks`` binary arrays of the same height and width, ``labels`` integer
    class indices and ``preprocess`` maps one image to a normalised CHW tensor
    without resizing, so masks and inputs stay aligned pixel for pixel.
    """
    cfg = config
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    labels = np.asarray(labels, dtype=int)
    Y = np.eye(n_classes)[labels]
    n = len(images)
    h, w = masks[0].shape
    blob = centre_blob((h, w), cfg.blob_sigma)

    # --- Test 1: localisation -------------------------------------------------
    cam = GradCAM(model, target_layer, channels_last=channels_last)
    rows = []
    for i in range(0, n, cfg.batch_size):
        x = torch.stack([preprocess(im) for im in images[i:i + cfg.batch_size]]).to(device)
        S = cam(x, labels[i:i + cfg.batch_size].tolist())
        for j, s in enumerate(S):
            m = masks[i + j]
            er, er_blob = energy_ratio(s, m), energy_ratio(blob, m)
            rows.append(dict(idx=i + j, label=int(labels[i + j]), rare=int(labels[i + j]) in set(rare_classes),
                             area=float((m > 0).mean()), ER=er, PG=pointing_game(s, m),
                             ER_blob=er_blob, beats_blob=bool(er > er_blob)))
    cam.remove()
    df = pd.DataFrame(rows)

    # --- Test 2: Lesion Reliance Index ---------------------------------------
    p0 = _probs(model, preprocess, images, cfg, device)
    df["p"] = p0[np.arange(n), labels]
    eligible, xL, xB = [], [], []
    for i in range(n):
        m = masks[i]
        a = (m > 0).mean()
        if not (cfg.min_area <= a <= cfg.max_area):
            continue
        ctrl = background_control(m, cfg.disc_radius, cfg.disc_margin, seed=cfg.seed + i)
        if ctrl is None:
            continue
        eligible.append(i)
        xL.append(inpaint(images[i], m, cfg.lri_inpaint_radius))
        xB.append(inpaint(images[i], ctrl, cfg.lri_inpaint_radius))
    df["p_lesion_erased"] = np.nan
    df["p_background_erased"] = np.nan
    if eligible:
        pL = _probs(model, preprocess, xL, cfg, device)
        pB = _probs(model, preprocess, xB, cfg, device)
        el = np.array(eligible)
        df.loc[el, "p_lesion_erased"] = pL[np.arange(len(el)), labels[el]]
        df.loc[el, "p_background_erased"] = pB[np.arange(len(el)), labels[el]]
    df["LRI"] = df["p_background_erased"] - df["p_lesion_erased"]

    # --- Test 3: discrimination without the lesion ---------------------------
    sel = np.array(eligible) if (cfg.auroc_on_eligible and eligible) else np.arange(n)
    auc0 = macro_auroc(Y[sel], p0[sel])
    summary = {
        "n_images": n,
        "ER_mean": round(float(df.ER.mean()), 4),
        "uniform_ER_mean (mask area)": round(float(df.area.mean()), 4),
        "centre_blob_ER_mean": round(float(df.ER_blob.mean()), 4),
        "share_beating_centre_blob": round(float(df.beats_blob.mean()), 4),
        "pointing_game": round(float(df.PG.mean()), 4),
        "n_LRI": int(df.LRI.notna().sum()),
        "LRI_mean": round(float(df.LRI.mean()), 4),
        "n_AUROC": int(len(sel)),
        "macro_AUROC": round(auc0, 4),
    }
    for name in cfg.erase_methods:
        erase = ERASERS[name]
        xe = [erase(images[i], dilate(masks[i], cfg.erase_dilation)) for i in sel]
        auc_e = macro_auroc(Y[sel], _probs(model, preprocess, xe, cfg, device))
        summary[f"macro_AUROC_erased[{name}]"] = round(auc_e, 4)
        summary[f"retained[{name}]"] = round((auc_e - 0.5) / (auc0 - 0.5), 4) if auc0 > 0.5 else float("nan")
    if len(rare_classes):
        for grp, sub in df.groupby("rare"):
            summary[f"ER_mean[{'rare' if grp else 'common'}]"] = round(float(sub.ER.mean()), 4)
    return AuditResult(per_image=df, summary=summary)
