import numpy as np
import torch
import torch.nn as nn

from lraudit import (AuditConfig, GradCAM, audit, background_control, blur_fill, centre_blob,
                     energy_ratio, inpaint, pointing_game, skin_fill)


def disc_mask(h=64, w=64, r=12):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < r ** 2).astype(np.uint8)


def test_energy_ratio_uniform_equals_area():
    m = disc_mask()
    assert abs(energy_ratio(np.ones(m.shape), m) - m.mean()) < 1e-9


def test_pointing_game_and_blob():
    m = disc_mask()
    b = centre_blob(m.shape, 8)
    assert pointing_game(b, m) == 1
    assert energy_ratio(b, m) > m.mean()


def test_background_control_area_and_disjoint():
    m = disc_mask()
    c = background_control(m, radius=4, margin=3, seed=1)
    assert c is not None
    assert (c & m).sum() == 0
    assert c.sum() >= 0.9 * m.sum()


def test_control_none_when_background_too_small():
    m = np.ones((32, 32), np.uint8); m[:2] = 0
    assert background_control(m) is None


def test_erasers_change_only_mask_region_roughly():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    m = disc_mask()
    for f in (inpaint, skin_fill, blur_fill):
        out = f(img, m)
        assert out.shape == img.shape and out.dtype == np.uint8
        far = np.zeros_like(m); far[:5, :5] = 1
        assert np.abs(out[far > 0].astype(int) - img[far > 0].astype(int)).max() <= 1


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


def test_gradcam_shape():
    m = Tiny(); cam = GradCAM(m, m.features[0])
    s = cam(torch.rand(2, 3, 32, 32), [0, 1]); cam.remove()
    assert s.shape == (2, 32, 32) and (s >= 0).all()


def test_audit_runs_end_to_end():
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(12)]
    masks = [disc_mask() for _ in range(12)]
    labels = np.arange(12) % 3
    pre = lambda im: torch.tensor(im, dtype=torch.float32).permute(2, 0, 1) / 255
    m = Tiny()
    res = audit(m, m.features[0], imgs, masks, labels, pre, n_classes=3, rare_classes=[2],
                config=AuditConfig(blob_sigma=10, disc_radius=4, disc_margin=3, erase_dilation=2, batch_size=4))
    assert len(res.per_image) == 12
    assert "retained[inpaint]" in res.summary
    assert res.per_image.LRI.notna().sum() > 0
