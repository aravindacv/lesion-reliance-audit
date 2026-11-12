"""Runnable demo on synthetic 'lesion' images (no download needed).

Two toy classifiers are trained on images where the class is visible both in a
central 'lesion' and in a background tint. The audit then shows that one can
look lesion-centred while still discriminating after the lesion is erased.

    python examples/demo_synthetic.py
"""
import numpy as np
import torch
import torch.nn as nn

from lraudit import AuditConfig, audit
from lraudit.stats import er_vs_blob

rng = np.random.default_rng(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
H = W = 96
N_CLASSES = 3


def make_image(label):
    img = np.full((H, W, 3), 200, np.uint8)
    img[..., label] = 215                                   # class-dependent background tint (a shortcut)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2 + rng.normal(0, 6), W / 2 + rng.normal(0, 6)
    r = rng.uniform(14, 22)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2).astype(np.uint8)
    colour = np.array([90, 60, 40]); colour[label] += 80     # class-dependent lesion colour
    img[mask > 0] = colour
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img, mask


def preprocess(im):
    return torch.tensor(im, dtype=torch.float32).permute(2, 0, 1) / 255.0


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(32, N_CLASSES)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


def main():
    data = [make_image(int(l)) for l in rng.integers(0, N_CLASSES, 400)]
    labels = np.array([np.argmax([d[0][..., k][d[1] == 0].mean() for k in range(3)]) for d in data])
    imgs, masks = [d[0] for d in data], [d[1] for d in data]
    X = torch.stack([preprocess(i) for i in imgs]); Y = torch.tensor(labels)
    model = SmallCNN(); opt = torch.optim.Adam(model.parameters(), 3e-3)
    for _ in range(15):
        for i in range(0, 300, 32):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model(X[i:i + 32]), nn.functional.one_hot(Y[i:i + 32], N_CLASSES).float()).backward()
            opt.step()
    res = audit(model, model.features[-2], imgs[300:], masks[300:], labels[300:], preprocess,
                n_classes=N_CLASSES, config=AuditConfig(blob_sigma=15, disc_radius=6, disc_margin=4,
                                                        erase_dilation=3, batch_size=50))
    print(res)
    print("ER vs centre blob (Wilcoxon):", er_vs_blob(res.per_image))


if __name__ == "__main__":
    main()
