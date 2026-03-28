# Y Channel Extension Plan

## Why Add Y Channel

The current Cb stream exploits chrominance-blue inconsistency — well-suited for the MyKad which is predominantly blue. The Y (luminance) channel is complementary:

- **Cb** catches: color temperature differences, spliced face with different blue-channel profile than card background
- **Y** catches: brightness inconsistencies, printing artifacts, contrast differences between genuine printed text and digitally overlaid content

A tampered region typically differs from its surroundings in **both** color and brightness. Using Y+Cb together gives the model two independent forensic signals instead of one.

**Why not Cr?** The MyKad has minimal red content — Cr is nearly uniform across the card for both genuine and tampered samples. Adding Cr would introduce noise without signal. Y is the right complement to Cb for this specific document.

---

## What Changes and Where

### 1. `src/models/cb_utils.py`

**Add a new extraction function** alongside `extract_cb_channel`:

```python
def extract_y_channel(rgb: torch.Tensor) -> torch.Tensor:
    """Extract Y (luminance) channel from RGB images."""
    ycbcr = rgb_to_ycbcr(rgb)
    return ycbcr[:, 0:1, :, :]  # Y is index 0


def extract_ycb_channels(rgb: torch.Tensor) -> torch.Tensor:
    """Extract Y and Cb channels stacked as 2-channel tensor."""
    ycbcr = rgb_to_ycbcr(rgb)
    return ycbcr[:, 0:2, :, :]  # (B, 2, H, W) — Y and Cb
```

**Update `BackgroundRegionExtractor.forward`** to accept a 2-channel input:

The method signature stays the same — just pass a 2-channel tensor instead of 1-channel. The `forward` method already handles arbitrary `C` via `B, C, H, W = cb_channel.shape`. The patches output becomes `(B, N, 2, P_h, P_w)` instead of `(B, N, 1, P_h, P_w)`.

No structural changes needed to `BackgroundRegionExtractor` — it is already channel-agnostic.

---

### 2. `src/models/model.py`

**Update `CbEncoder.__init__`** — change input channels from 1 to 2:

```python
# Before:
nn.Conv2d(1, 32, kernel_size=3, padding=1),

# After (rename class to YCbEncoder or add in_channels param):
def __init__(self, in_channels: int = 1, output_dim: int = 128, dropout: float = 0.3):
    ...
    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
```

**Update instantiation in `RGBCbTamperDetector.__init__`:**

```python
# Before:
self.cb_encoder = CbEncoder(output_dim=cb_encoder_dim, dropout=dropout)

# After:
self.cb_encoder = CbEncoder(in_channels=2, output_dim=cb_encoder_dim, dropout=dropout)
```

**Update `forward()` to extract both channels:**

```python
# Before:
cb_channel = extract_cb_channel(images)
cb_patches = self.bg_extractor(cb_channel)

# After:
ycb_channel = extract_ycb_channels(images)   # (B, 2, H, W)
cb_patches = self.bg_extractor(ycb_channel)  # (B, 80, 2, P_h, P_w)
```

**Update import:**
```python
# Before:
from .cb_utils import extract_cb_channel, BackgroundRegionExtractor

# After:
from .cb_utils import extract_ycb_channels, BackgroundRegionExtractor
```

---

### 3. `src/models/losses.py`

**No changes needed.** `CbConsistencyLoss` operates on encoded patch features `(B, N, D)` — it never sees raw channels. The loss is channel-agnostic.

---

### 4. `configs/config.yaml`

Add a flag to make this togglable without code changes:

```yaml
model:
  use_y_channel: true    # false = Cb only (original), true = Y+Cb
```

Then in `model.py`, read this flag and pass `in_channels=2` or `in_channels=1` accordingly.

---

## Tensor Shape Trace (with Y+Cb)

```
RGB input (B, 3, 299, 299)
    ↓
extract_ycb_channels()
    ↓
(B, 2, 299, 299)   ← Y and Cb stacked
    ↓
BackgroundRegionExtractor  [no change needed]
    ↓
(B, 80, 2, 14, 14)   ← 80 patches, 2 channels each
    ↓
CbEncoder  [Conv2d input: 1→2]
    ↓
(B, 80, 128)   ← same output shape, more informative input
    ↓
CrossModalFusion  [no change]
    ↓
...
```

The output shape `(B, 80, 128)` is **identical** to the current Cb-only stream. Everything downstream (fusion, classifier, loss) requires zero changes.

---

## Parameter Impact

| Component | Cb only | Y+Cb |
|---|---|---|
| `CbEncoder` first Conv2d | `Conv2d(1, 32)` = 288 params | `Conv2d(2, 32)` = 576 params |
| All other layers | unchanged | unchanged |
| Total delta | — | +288 params |

Negligible increase (~288 extra weights). The encoder is already tiny — the additional channel costs almost nothing.

---

## Expected Benefit

Luminance captures:
- **Printed vs digital content**: Genuine ID card text/photo is printed with consistent luminance across the card. A digitally overlaid face photo has different brightness characteristics (screen-captured photos, different gamma, JPEG compression artifacts).
- **Shadow and lighting consistency**: Tampered regions often have inconsistent shadows or exposure compared to the surrounding card.
- **Edge artifacts**: Splice boundaries show luminance discontinuities that Cb alone may miss if the color is similar.

---

## Implementation Order

1. `cb_utils.py` — add `extract_ycb_channels()` (5 min)
2. `model.py` — add `in_channels` param to `CbEncoder`, update `forward()` (10 min)
3. `config.yaml` — add `use_y_channel` flag (2 min)
4. Verify shape with a quick forward pass test (5 min)

Total: ~20 min of changes. All downstream code (loss, fusion, classifier, training loop) requires zero modification.

---

## Verification

```python
import torch, yaml
from src.models.model import RGBCbTamperDetector

cfg = yaml.safe_load(open('configs/config.yaml'))
model = RGBCbTamperDetector(**cfg['model']).cuda()
x = torch.randn(2, 3, 299, 299).cuda()
out = model(x, return_features=True)

print(out['logits'].shape)       # (2, 2)
print(out['cb_features'].shape)  # (2, 80, 128)  — same as before
```
