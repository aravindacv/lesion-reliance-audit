"""Saliency maps and localisation scores.

Grad-CAM (Selvaraju et al.), the energy ratio (ER), the pointing game (PG)
and the model-free centre-prior baseline used in the audit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM at a chosen layer of a PyTorch classifier.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier returning one logit per class, shape (B, C).
    target_layer : torch.nn.Module
        Layer whose activations are used (normally the last feature stage).
    channels_last : bool
        Set True for layers that output (B, H, W, K) instead of (B, K, H, W),
        as in some Swin / MaxViT implementations.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, channels_last: bool = False):
        self.model = model
        self.channels_last = channels_last
        self._act = None
        self._grad = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inputs, output):
        self._act = output

    def _save_grad(self, module, grad_in, grad_out):
        self._grad = grad_out[0]

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, classes) -> np.ndarray:
        """Return saliency maps of shape (B, H, W), upsampled to the input size.

        ``classes`` is an int or a sequence with one class index per image.
        """
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if isinstance(classes, int):
            classes = [classes] * x.shape[0]
        idx = torch.as_tensor(list(classes), device=logits.device)
        logits.gather(1, idx[:, None]).sum().backward()
        A, G = self._act, self._grad
        if self.channels_last:
            A, G = A.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam[:, 0].detach().cpu().numpy()


def energy_ratio(saliency: np.ndarray, mask: np.ndarray) -> float:
    """Share of saliency energy inside the lesion mask (ER)."""
    s = np.clip(saliency, 0, None).astype(np.float64)
    total = s.sum()
    if total <= 0:
        return float("nan")
    return float(s[mask > 0].sum() / total)


def pointing_game(saliency: np.ndarray, mask: np.ndarray) -> int:
    """1 if the saliency maximum lies inside the mask, else 0 (PG)."""
    i, j = np.unravel_index(np.argmax(saliency), saliency.shape)
    return int(mask[i, j] > 0)


def centre_blob(shape, sigma: float = 40.0) -> np.ndarray:
    """Image-blind isotropic Gaussian centred in the image (the centre prior)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
