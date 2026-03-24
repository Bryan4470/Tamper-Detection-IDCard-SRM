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
| Dropout | `src/model_core.py` | `dropout=0.5` in both `TransferModel` calls |

## Classification Threshold

`TAMPER_THRESHOLD` in `predict.py` controls the tamper/genuine boundary on the softmax probability of the tamper class. Default is `0.25` (biased toward catching fraud). Lower = more sensitive to tampering (lower FAR, higher FRR). Tune using `tune_threshold.py` after training — no retraining needed.

## Loss & Class Imbalance

Current loss: `nn.CrossEntropyLoss()`. To address the ~55/45 genuine/tamper imbalance and reduce FAR, apply class weighting:
```python
nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.25]).to(DEVICE))
```
`src/loss/am_softmax.py` contains `AMSoftmaxLoss` and `focal_loss` — these were part of the original face forgery codebase and are not used in the current IC card training pipeline.
