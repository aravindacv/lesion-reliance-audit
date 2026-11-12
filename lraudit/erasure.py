"""Lesion erasure and the equal-area background control.

All functions take and return uint8 RGB images of shape (H, W, 3) and binary
masks of shape (H, W).
"""
from __future__ import annotations

import cv2
import numpy as np


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate a binary mask by ``pixels`` with an elliptical kernel."""
    if pixels <= 0:
        return (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate((mask > 0).astype(np.uint8), k)


def inpaint(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Telea inpainting of the masked region."""
    return cv2.inpaint(img, (mask > 0).astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)


def skin_fill(img: np.ndarray, mask: np.ndarray, ring: int = 10, noise_sd: float = 4.0,
              feather: int = 5, seed: int = 0) -> np.ndarray:
    """Fill the mask with the median colour of a surrounding skin ring plus noise, feathered edge."""
    m = (mask > 0).astype(np.uint8)
    ring_mask = dilate(m, ring) & (1 - m)
    if ring_mask.sum() == 0:
        ring_mask = 1 - m
    colour = np.median(img[ring_mask > 0].reshape(-1, 3), axis=0)
    rng = np.random.default_rng(seed)
    fill = np.clip(colour[None, None, :] + rng.normal(0, noise_sd, img.shape), 0, 255)
    alpha = m.astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        alpha = np.maximum(alpha, m.astype(np.float32))
    out = alpha[..., None] * fill + (1 - alpha[..., None]) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_fill(img: np.ndarray, mask: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Replace the masked region with a Gaussian-blurred copy (keeps colour, removes texture)."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    m = (mask > 0)[..., None]
    return np.where(m, blurred, img)


ERASERS = {"inpaint": inpaint, "skin_fill": skin_fill, "blur": blur_fill}


def background_control(mask: np.ndarray, radius: int = 12, margin: int = 8,
                       min_ratio: float = 1.3, seed: int = 0, max_tries: int = 20000):
    """Union of random discs in the background with the same area as the lesion.

    Returns None when the background (outside ``margin`` pixels of the lesion)
    is smaller than ``min_ratio`` times the lesion area.
    """
    m = (mask > 0).astype(np.uint8)
    target = int(m.sum())
    allowed = 1 - dilate(m, margin)
    if target == 0 or allowed.sum() < min_ratio * target:
        return None
    h, w = m.shape
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(allowed)
    ctrl = np.zeros_like(m)
    disc = np.zeros_like(m)
    for _ in range(max_tries):
        if ctrl.sum() >= target:
            break
        k = rng.integers(len(ys))
        disc[:] = 0
        cv2.circle(disc, (int(xs[k]), int(ys[k])), radius, 1, -1)
        ctrl |= disc & allowed
    return ctrl if ctrl.sum() >= 0.9 * target else None
