# lesion-reliance-audit (`lraudit`)

**An interventional audit that tests whether a medical image classifier actually relies on the lesion, and whether its saliency map tells you so.**

A heatmap that lights up the lesion is often taken as evidence that a model "looks at the right thing". `lraudit` tests that claim directly. Give it a trained classifier, images, lesion masks and labels, and it returns three tests:

| Test | Question | Output |
|---|---|---|
| 1. Localisation against two baselines | Does Grad-CAM fall on the lesion more than a uniform map **and** more than an image-blind Gaussian at the image centre? | Energy ratio (ER), pointing game, share of images beating the centre prior |
| 2. Interventional reliance | Does erasing the lesion lower the model's confidence more than erasing an equal area of skin? | Lesion Reliance Index, LRI = p(background erased) − p(lesion erased) |
| 3. Discrimination without the lesion | How much of the model's discrimination survives when every lesion is erased? | Macro AUROC on lesion-erased images (inpainting, skin fill, blur) and the share of above-chance AUROC retained |

The protocol is described in:

> Aravinda C V, Raja K B, Reddy K R U K, Arjun B C, Rajendran A. *Interventional auditing reveals that saliency maps overstate lesion reliance in dermoscopy classifiers* (under review).

## Install

```bash
git clone https://github.com/<your-account>/lesion-reliance-audit.git
cd lesion-reliance-audit
pip install -e .
```

Requirements: Python ≥ 3.9, PyTorch, OpenCV, NumPy, pandas, scikit-learn, SciPy.

## Quick start

```python
import torch, timm
from lraudit import audit, AuditConfig

model = timm.create_model("efficientnet_b3", num_classes=11)
model.load_state_dict(torch.load("my_checkpoint.pt"))

# images: list of uint8 RGB arrays already resized to the network input (e.g. 224x224)
# masks:  list of binary arrays of the same size
# labels: list of integer class indices
mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
preprocess = lambda im: (torch.tensor(im).permute(2, 0, 1).float() / 255 - mean) / std

res = audit(model, model.conv_head, images, masks, labels, preprocess,
            n_classes=11, rare_classes=[2, 4, 5, 6, 10])
print(res)               # summary
res.per_image.to_csv("audit_per_image.csv")
```

`preprocess` must **not** resize or crop, so that masks and network inputs stay aligned pixel for pixel. For transformer models whose last stage outputs `(B, H, W, C)` (some Swin and MaxViT implementations), pass `channels_last=True`.

### How to read the summary

* `ER_mean` above `uniform_ER_mean` only means saliency is better than a flat map. The informative comparison is with `centre_blob_ER_mean` and `share_beating_centre_blob`: dermoscopy photographs are framed with the lesion in the centre, so an image-blind blob already scores well.
* `LRI_mean` near 0 means that removing the lesion costs the model no more confidence than removing an equal patch of skin.
* `retained[method]` near 1 means the model still discriminates almost as well with every lesion erased, i.e. it uses context outside the lesion.

## Demo without any data

```bash
python examples/demo_synthetic.py
```

The demo trains a small CNN on synthetic images in which the class is visible both in a central "lesion" and in a background tint. The audit shows a saliency map that hits the lesion (pointing game 1.0, ER three times the mask area) yet never beats the centre prior, while discrimination is fully retained after the lesion is erased: the model uses the background shortcut.

## Defaults

All defaults follow the paper (224 × 224 inputs): centre-prior σ = 40 px; LRI by Telea inpainting (radius 5) with a background control made of 12-px discs outside an 8-px margin, matched in area; masks covering 2–60 % of the image; erasure for AUROC after 7-px dilation. Change them with `AuditConfig`.

## Statistics

`lraudit.stats` provides percentile-bootstrap AUROC intervals, paired bootstrap differences, the Wilcoxon test of ER against the centre prior and the Mann–Whitney test between rare and common classes.

## Tests

```bash
pip install pytest && pytest -q
```

## Licence and citation

MIT licence. If you use this package, please cite the paper above (see `CITATION.cff`).
