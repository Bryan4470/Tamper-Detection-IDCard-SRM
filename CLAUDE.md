# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Train the model
python train.py

# Predict a single image
python predict.py --image path/to/card.jpg

# Predict all images in a folder
python predict.py --folder path/to/folder

# Evaluate on test set (FAR, FRR, classification report)
python predict.py --eval

# Tune classification threshold across a range
python tune_threshold.py
```

All scripts must be run from the `SRM/` directory. They internally `os.chdir` into `src/` so that Xception weight loading resolves correctly — keep this in mind when editing path logic.

## Data

- **Train / Val source:** `C:\Users\bryancfk\extracted_images\genuine` and `\tamper`
- **Test source:** `C:\Users\bryancfk\extracted_images_test\genuine` and `\tamper`
- No data copying or `prepare_data.py` step is needed. `train.py` performs an in-memory stratified 85/15 train/val split at runtime using `make_splits()`.
- Dataset: ~6875 genuine, ~5500 tamper images (mild class imbalance).

## Architecture

The model (`src/model_core.py: Two_Stream_Net`) is a **two-stream network** adapted from the CVPR 2021 paper *"Generalizing Face Forgery Detection with High-Frequency Features"*, repurposed for IC card tamper detection.

**Two streams running in parallel:**
- **RGB stream** — standard Xception backbone processing the raw image
- **SRM stream** — Xception backbone processing SRM (Steganalysis Rich Model) noise residuals

**Key modules in `src/components/`:**
- `srm_conv.py` — fixed (non-learnable) high-frequency filter kernels (KB, KV, horizontal 2nd-order). Applied at input and at early feature layers to inject noise residuals into the SRM stream
- `attention.py` — three attention mechanisms:
  - `SpatialAttention` / `ChannelAttention` (CBAM-style) — used in residual-guided spatial attention
  - `DualCrossModalAttention` — bidirectional cross-modal attention applied twice (at middle and late feature stages) to let RGB and SRM streams inform each other
- `model_core.py: FeatureFusionModule` — concatenates final RGB and SRM features (2048+2048 → 2048) with channel attention before classification

**Forward pass flow:**
1. SRM noise extracted from input → fed into SRM stream
2. Early SRM features injected into RGB stream via `SRMConv2d_Separate` residual connections
3. SRM spatial attention map modulates RGB features (residual-guided attention)
4. `DualCrossModalAttention` applied at two depths for cross-stream feature exchange
5. Streams fused → Xception classifier head → logits for 2 classes (genuine=0, tamper=1)

**Pretrained weights:** Xception backbone initialized from `src/networks/xception-b5690688.pth` (ImageNet pretrained).

## Key Config Locations

| Setting | File | Variable |
|---|---|---|
| Source data paths | `train.py` | `SRC_DIR` |
| Test data path | `predict.py` | `TEST_DIR` |
| Train/val split ratio | `train.py` | `VAL_SPLIT` |
| Tamper decision threshold | `predict.py` | `TAMPER_THRESHOLD` |
| Image size | `train.py` / `predict.py` | `IMAGE_SIZE` |
| Dropout | `src/model_core.py` | `dropout=0.3` in both `TransferModel` calls |
| Tamper class loss weight | `config.yaml` | `training.loss_weight_tamper` |
| Attention aux loss weight | `config.yaml` | `training.att_aux_weight` |
| Focal loss gamma | `config.yaml` | `training.focal_gamma` |
| Early stopping monitor | `config.yaml` | `early_stopping.monitor` |

## Classification Threshold

`TAMPER_THRESHOLD` in `predict.py` controls the tamper/genuine boundary on the softmax probability of the tamper class. Default is `0.25` (biased toward catching fraud). Lower = more sensitive to tampering (lower FAR, higher FRR). Tune using `tune_threshold.py` after training — no retraining needed.

## Loss & Class Imbalance

Current loss is a three-component combination:

1. **Focal loss** (`src/loss/am_softmax.py: focal_loss`, gamma=2.0) applied on per-sample cross-entropy — down-weights easy correct predictions and focuses gradient on hard borderline tamper cases (the primary FAR failure mode).
2. **Class-weighted CE** — tamper class upweighted (`loss_weight_tamper: 2.0` in `config.yaml`) to penalise false negatives on tampered cards more heavily.
3. **Auxiliary attention map loss** — BCE loss between `att_map` output and the tamper label (weight=0.1, `att_aux_weight` in `config.yaml`). Forces the `SRMPixelAttention` module to spatially localize tamper regions rather than treating every image globally.

```python
per_sample_ce = F.cross_entropy(logits, labels, weight=criterion.weight, reduction='none')
loss = focal_loss(per_sample_ce, gamma=FOCAL_GAMMA)
loss += ATT_AUX_WEIGHT * F.binary_cross_entropy(att_map_upsampled, tamper_target)
```

All three weights are configurable in `config.yaml` under `training`. `src/loss/am_softmax.py` also contains `AMSoftmaxLoss` (not currently used).
