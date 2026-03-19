# Face Tamper Detection — Model Documentation & Enhancement Guide

> Last updated: 2026-03-18
> Branch: `dual_stream`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Deep Dive](#2-component-deep-dive)
   - 2.1 [RGB Stream](#21-rgb-stream)
   - 2.2 [Cb Stream](#22-cb-stream)
   - 2.3 [Cross-Modal Fusion](#23-cross-modal-fusion)
   - 2.4 [Classification Head](#24-classification-head)
3. [Loss Functions](#3-loss-functions)
4. [Training Pipeline](#4-training-pipeline)
5. [Inference Pipeline](#5-inference-pipeline)
6. [Core Forensic Hypothesis](#6-core-forensic-hypothesis)
7. [Real-World Limitations](#7-real-world-limitations)
8. [Enhancement Suggestions](#8-enhancement-suggestions)
   - 8.1 [SRM High-Frequency Stream](#81-srm-high-frequency-stream)
   - 8.2 [Supervised Contrastive Loss](#82-supervised-contrastive-loss)
   - 8.3 [Add Cr Channel (CbCr)](#83-add-cr-channel-cbcr)
   - 8.4 [ELA Auxiliary Branch](#84-ela-auxiliary-branch)
   - 8.5 [DCT Coefficient Stream](#85-dct-coefficient-stream)
   - 8.6 [Boundary Supervision Head](#86-boundary-supervision-head)
   - 8.7 [Noiseprint++ Integration](#87-noiseprint-integration)
   - 8.8 [Realistic Tamper Augmentation](#88-realistic-tamper-augmentation)
   - 8.9 [Deformable Patch Selection](#89-deformable-patch-selection)
   - 8.10 [ViT Backbone Upgrade](#810-vit-backbone-upgrade)
9. [Proposed Redesigned Architecture](#9-proposed-redesigned-architecture)
10. [Enhancement Priority Table](#10-enhancement-priority-table)
11. [Key Configuration Files](#11-key-configuration-files)
12. [References](#12-references)

---

## 1. Architecture Overview

The model is a **Dual-Stream Deep Learning** binary classifier that detects whether an ID card has been physically tampered with. It targets Malaysian IC (MyKad) cards but is generalisable to other ID card formats.

```
Input Image (224×224 RGB)
            │
     ┌──────┴──────┐
     │             │
[RGB Stream]   [Cb Stream]
     │               │
EfficientNet-B3   Extract Cb channel (YCbCr)
(ImageNet pre-    → 5 predefined card regions
 trained)         → each region → 4×4 grid = 16 patches
     │            → 3-layer CNN encoder per patch
1536D global      → 80 × 128D patch features
 features          │
     └──────┬──────┘
            │
    CrossModalFusion
    (Multi-head Attention:
     RGB as query, Cb patches as key/value)
            │
         256D fused features
            │
    Classifier (256 → 128 → 2)
            │
    Genuine / Tampered
```

**Key files:**

| File | Purpose |
|------|---------|
| `src/models/model.py` | `RGBCbTamperDetector`, `CbEncoder`, `CrossModalFusion` |
| `src/models/cb_utils.py` | RGB→YCbCr conversion, `BackgroundRegionExtractor` |
| `src/models/losses.py` | `CombinedLoss`, `CbConsistencyLoss`, `FocalLoss` |
| `src/data/dataloader.py` | `EKYCDataLoader`, `EKYCDataset`, stratified splits |
| `src/data/augmentation.py` | Transform pipeline from config |
| `scripts/train.py` | Training loop with early stopping |
| `scripts/inference.py` | Inference wrapper with evaluation metrics |
| `scripts/api.py` | FastAPI server (port 7500) |
| `configs/config.yaml` | All hyperparameters |
| `configs/regions.yaml` | ID card region coordinate definitions |

---

## 2. Component Deep Dive

### 2.1 RGB Stream

**Class:** `RGBCbTamperDetector.rgb_backbone`
**File:** `src/models/model.py:134–158`

A pretrained CNN backbone with the final classification layer removed, followed by global average pooling. Acts as the semantic understanding stream — it learns what objects are present, their textures, and general visual features.

**Supported backbones:**

| Backbone | Output Dim | Parameters |
|----------|-----------|-----------|
| `resnet18` | 512 | ~11M |
| `resnet50` | 2048 | ~23M |
| `efficientnet_b0` | 1280 | ~5M |
| `efficientnet_b3` | 1536 | ~12M (default) |

Weights are loaded from PyTorch's model hub (ImageNet pretrained). The backbone is fine-tuned end-to-end during training.

---

### 2.2 Cb Stream

**Classes:** `BackgroundRegionExtractor`, `CbEncoder`
**Files:** `src/models/cb_utils.py`, `src/models/model.py:21–55`

#### Color Space Conversion (ITU-R BT.601)

```
Y  =  0.299R + 0.587G + 0.114B
Cb = -0.169R - 0.331G + 0.500B   ← only this channel is used
Cr =  0.500R - 0.419G - 0.081B
```

The **Cb channel** (chrominance-blue) is extracted and processed independently. It captures blue-channel color difference information that varies significantly between images captured by different cameras, under different lighting, or with different color pipelines.

#### Region Extraction (`configs/regions.yaml`)

Five hardcoded card regions (normalised 0–1 coordinates) are extracted:

| Region | Description | Why Forensically Relevant |
|--------|-------------|--------------------------|
| Face zone | Main photo area | Primary tampering target |
| ID number | Numeric identifier field | Can be altered digitally |
| Name/address | Text identity fields | Text replacement target |
| Ghost face | Watermark/holographic layer | Watermark consistency |
| Religion/gender | Additional identity fields | Secondary text targets |

#### Patch Division

Each region is:
1. Resized to 56×56 pixels
2. Divided into a 4×4 grid → 16 non-overlapping patches of 14×14 each
3. Total: 5 regions × 16 patches = **80 patches per image**

#### CbEncoder

A lightweight 3-layer CNN processes each 14×14 patch independently:

```
Input (1, 14, 14)
  → Conv2d(1→32, 3×3) + BN + ReLU + MaxPool
  → Conv2d(32→64, 3×3) + BN + ReLU + MaxPool
  → Conv2d(64→128, 3×3) + BN + ReLU + AdaptiveAvgPool
  → Linear(128 → 128) + ReLU + Dropout(0.3)
Output: 128D feature vector per patch
```

All 80 patches are processed in a single batched forward pass:
`(B, 80, 1, 14, 14)` → `(B×80, 1, 14, 14)` → `(B, 80, 128)`

---

### 2.3 Cross-Modal Fusion

**Class:** `CrossModalFusion`
**File:** `src/models/model.py:58–93`

Combines the global RGB semantic features with the local Cb patch features using cross-attention:

```
RGB global (B, 1536) → Linear → (B, 1, 256)   [query]
Cb patches (B, 80, 128) → Linear → (B, 80, 256) [key, value]

MultiheadAttention(4 heads):
  query = RGB embedding
  key = value = Cb patch embeddings
  → RGB attends to the most forensically relevant Cb patches
  → Output: (B, 1, 256) attended features

Fusion: concat(RGB_embed, attended_cb) → MLP(512→256→256)
Output: (B, 256) fused feature vector
```

The attention mechanism allows the model to learn which Cb patches are most discriminative. In practice, the face region patches get the highest attention weights when a face swap has occurred.

---

### 2.4 Classification Head

**File:** `src/models/model.py:170–175`

```
Input (B, 256)
→ Linear(256→128) + ReLU + Dropout(0.5)
→ Linear(128→2)
→ Softmax → [P(genuine), P(tampered)]
```

Output is the raw logit pair; softmax is applied for probability output at inference time.

---

## 3. Loss Functions

**File:** `src/models/losses.py`

### Combined Loss

```
L_total = 1.0 × L_classification + 0.3 × L_cb_consistency
```

### Classification Loss

Default: `CrossEntropyLoss`
Optional: `FocalLoss(α=0.25, γ=2.0)` — use when class imbalance is significant.

### Cb Consistency Loss (the novel component)

This is a **region-level metric learning** objective that shapes the Cb feature space:

```
For each region r (5 regions):
  region_score[r] = mean pairwise L2 distance between 16 patches

For GENUINE images:
  L_genuine = max(region_scores)  ← penalise any region with inconsistency
  (all regions should have low intra-region distance)

For TAMPERED images:
  L_tampered = ReLU(margin - max(region_scores))
  (at least one region must exceed distance margin = 0.5)
```

**Intuition:**
- A genuine card was printed as a single physical object under consistent conditions — all Cb patches should be similar across the card.
- A tampered card (face replaced) has one region from a different source — at least one region will show high intra-region Cb variance.

---

## 4. Training Pipeline

**File:** `scripts/train.py`
**Config:** `configs/config.yaml`

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNet-B3 |
| Optimizer | AdamW (lr=0.0005, weight_decay=0.01) |
| Scheduler | Cosine annealing with 5-epoch warmup |
| Batch size | 64 |
| Max epochs | 60 |
| Early stopping patience | 10 |
| Gradient clip norm | 1.0 |
| Mixed precision | Optional |
| Train/val split | 80/20 stratified |

### Data Augmentation

Deliberately minimal to preserve Cb signal integrity:

| Augmentation | Applied | Reason |
|---|---|---|
| Rotation ±5° | Yes | Simulate card angle variation |
| Gaussian blur (10% prob) | Yes | Simulate scan quality variation |
| Color jitter | **No** | Would corrupt Cb consistency signal |
| Color augmentation | **No** | Would corrupt Cb consistency signal |
| Horizontal flip | No | IDs are orientation-specific |

### Checkpoint Strategy

| File | Saved When |
|------|-----------|
| `best_f1.pth` | Best F1 score |
| `best_auc.pth` | Best AUC-ROC |
| `best_frr_under_far.pth` | Best FRR when FAR ≤ 1% |
| `last_epoch.pth` | Every epoch (for resuming) |

### Evaluation Metrics

- Accuracy, Precision, Recall, F1
- AUC-ROC (threshold-independent)
- FAR (False Accept Rate) — tampered accepted as genuine
- FRR (False Reject Rate) — genuine rejected as tampered
- Confusion matrix (TP/TN/FP/FN)

---

## 5. Inference Pipeline

**File:** `scripts/inference.py`
**API:** `scripts/api.py` (FastAPI, port 7500)

### Output Format

```csv
image_path,prediction,confidence,prob_genuine,prob_tampered
/path/to/img.jpg,genuine,0.92,0.92,0.08
/path/to/img2.jpg,tampered,0.85,0.15,0.85
```

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Single image prediction |
| `POST /predict_batch` | Batch of images |

### Error Analysis

False positives (genuine → tampered) and false negatives (tampered → genuine) are saved to `error_cases/` with per-sample confidence scores.

---

## 6. Core Forensic Hypothesis

The model's effectiveness rests on the following chain of reasoning:

1. **Physical ID card printing** produces a card with uniform optical properties — the entire card was printed by the same device, at the same time, under the same conditions.

2. **Face photo replacement** means the new photo comes from a different source (a phone camera, a different scanner, a different printing process).

3. **Different imaging pipelines** produce different Cb (chrominance-blue) characteristics due to:
   - Different camera sensors and colour filters
   - Different white balance settings
   - Different JPEG compression quality and quantisation tables
   - Different printer/scanner colour profiles

4. Therefore, a **tampered face region** will have Cb statistics inconsistent with the rest of the card, even if RGB appearance looks plausible.

5. The **Cb Consistency Loss** trains the model to exploit this inconsistency as the primary tampering signal.

---

## 7. Real-World Limitations

The model assumes a specific tamper signature (cross-source Cb inconsistency). Sophisticated real-world attacks can break these assumptions:

| Attack Type | How It Fools the Model | Severity |
|---|---|---|
| **Colour-matched face swap** | Adversary histogram-matches Cb channel of inserted photo to the background — directly eliminates the Cb inconsistency signal | High |
| **GAN / inpainting synthesis** | Synthesised faces have smooth spectral characteristics with no physical printing boundaries | High |
| **Screen recapture** | Re-photographing a tampered card collapses all JPEG histories into one capture, erasing double-compression traces | High |
| **High-quality scan + reprint** | Printing on the same printer/paper as the original card makes Cb look identical | Medium-High |
| **Cropped genuine photo** | Using a genuine photo that happens to share similar Cb statistics | Medium |
| **Text field tampering** | Altering name/ID number digits — the model primarily uses face region patches | Medium |
| **Digital-only edits** | Editing the scanned digital image directly with consistent JPEG quality settings | Medium |
| **Partial face cover** | Obscuring part of the face with a sticker or object | Low-Medium |
| **Cross-card composite** | Reassembling valid card components from multiple genuine cards | Medium |

---

## 8. Enhancement Suggestions

---

### 8.1 SRM High-Frequency Stream

**Priority: HIGH**
**Complexity: LOW**
**Basis:** Luo et al., *"Generalizing Face Forgery Detection with High-Frequency Features"*, CVPR 2021. CFL-Net, WACV 2023.

#### What it is

The **Spatial Rich Model (SRM)** uses 30 fixed high-pass convolutional filter kernels tuned for image forensics. They extract manipulation-agnostic noise residuals — the subtle high-frequency artifacts that all editing operations leave behind, regardless of how well the colour space was matched.

Unlike the Cb stream (which can be defeated by colour matching), noise residuals are intrinsic to the imaging pipeline and extremely difficult to fake.

SRM filters capture:
- Resampling artifacts (scaling/rotation to fit a spliced face)
- JPEG double-compression boundary artifacts
- Sharpening/blurring traces left by photo editing tools
- Demosaicing pattern inconsistencies between different cameras

#### Architecture Change

Add a third parallel stream:

```
Input
  ├── RGB stream (existing)
  ├── Cb stream (existing)
  └── SRM noise stream (NEW)
       → 30 fixed SRM filter kernels (non-trainable Conv2d)
       → Lightweight CNN encoder (3 layers, similar to CbEncoder)
       → 128D noise feature vector
       → Fused into CrossModalFusion
```

The 30 SRM kernels are **fixed** (not trained) — they are copied directly from the SRM steganalysis paper. This adds negligible parameters while providing a qualitatively different forensic signal.

#### Why It Works

Luo et al. CVPR 2021 showed that SRM-based dual-stream models generalise to **unseen forgery methods** at test time, because manipulations always leave noise traces even when colour is carefully matched. This is the single most effective intervention for improving generalisation to novel real-world attacks.

---

### 8.2 Supervised Contrastive Loss

**Priority: HIGH**
**Complexity: LOW**
**Basis:** CFL-Net, WACV 2023. SeeABLE, ICCV 2023.

#### What it is

The current classification head is trained with cross-entropy, which does not explicitly separate the feature space. Adding **supervised contrastive loss** on the fused features forces genuine IDs to cluster together and tampered IDs to cluster together, regardless of attack style.

```
L_total = L_cls + 0.3 × L_cb_consistency + λ × L_contrastive

L_contrastive = InfoNCE(genuine_fused_features, tampered_fused_features)
```

Within a batch, all genuine samples form positive pairs with each other, and tampered samples form positive pairs with each other. Genuine and tampered samples are negative pairs.

#### Why It Works

CFL-Net (WACV 2023) showed that adding supervised contrastive loss on the fused feature space improves cross-dataset AUC by **5–15%** without any architectural change. The intuition: cross-entropy only learns decision boundaries for the training distribution. Contrastive loss learns a metric space that generalises to novel attack types encountered at deployment.

SeeABLE (ICCV 2023) extends this to a one-class formulation — training only on genuine images and treating all deviations as tampered — which is worth exploring if the tampered training set is small.

---

### 8.3 Add Cr Channel (CbCr)

**Priority: MEDIUM**
**Complexity: MINIMAL**
**Basis:** Chrominance forensics literature. Springer 2025 deepfake chrominance study.

#### What it is

The current model uses only the **Cb** channel (chrominance-blue). The **Cr** channel (chrominance-red) carries complementary colour difference information, and is particularly discriminative for **skin tone inconsistencies** — critical for face photo replacement on ID cards.

#### Architecture Change

Modify `cb_utils.py` to extract both channels:

```python
def extract_cbcr_channels(rgb: torch.Tensor) -> torch.Tensor:
    ycbcr = rgb_to_ycbcr(rgb)
    return ycbcr[:, 1:3, :, :]  # (B, 2, H, W) — Cb and Cr
```

Modify `CbEncoder` to accept 2-channel input (`nn.Conv2d(2, 32, ...)` instead of `nn.Conv2d(1, 32, ...)`).

This is a minimal change that nearly doubles the chrominance forensic signal with no additional model complexity.

---

### 8.4 ELA Auxiliary Branch

**Priority: MEDIUM**
**Complexity: LOW**
**Basis:** ELA-Enhanced Dual-Branch model (2024 preprint). Scientific Reports (Nature) 2023.

#### What it is

**Error Level Analysis (ELA)** re-saves the image at a known JPEG quality level and computes the absolute difference from the original. Regions with a different JPEG compression history — spliced regions saved at a different quality — show distinctly different error levels (appear brighter in the ELA map).

This is especially powerful for ID card scans which have a predictable and uniform compression history. A spliced face photo was JPEG-compressed before insertion, then the whole card was JPEG-compressed again — the double-compression leaves detectable traces.

```python
def compute_ela(image_tensor, quality=95):
    # Re-compress at known quality
    buffer = io.BytesIO()
    pil_img = to_pil_image(image_tensor)
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = to_tensor(Image.open(buffer))
    ela_map = (image_tensor.float() - recompressed.float()).abs()
    return ela_map  # (3, H, W)
```

Add as a third CNN branch. The ELA map can be processed by a lightweight CNN (similar to `CbEncoder`) and fused into the CrossModalFusion module.

Scientific Reports 2023 showed ELA adds ~2.7% accuracy at only ~5.6% additional processing overhead.

---

### 8.5 DCT Coefficient Stream

**Priority: MEDIUM-HIGH (for digital tampering)**
**Complexity: MEDIUM**
**Basis:** CAT-Net, WACV 2021. ADCD-Net, ICCV 2025 (arXiv 2507.16397).

#### What it is

JPEG compresses images by storing 8×8 block DCT coefficients at a given quantisation level. When a face is digitally inserted and the image is re-saved, the spliced region has a **different quantisation table and compression history** than the original regions. This creates detectable double-compression artifacts in the DCT coefficient distributions.

**CAT-Net** (WACV 2021) encodes DCT coefficients as a binary volume (presence/absence of each coefficient per frequency band) and processes it with a dedicated CNN stream alongside RGB.

**ADCD-Net** (ICCV 2025) improves on this with an **adaptive alignment score** that compensates for 8×8 block misalignment caused by image resizing — a common issue when card images are cropped or resized before the DCT analysis.

This stream is most valuable when attacks involve:
1. Scanning a genuine card
2. Editing the digital scan (pasting a new face)
3. Printing and re-scanning, OR submitting the digital image directly

---

### 8.6 Boundary Supervision Head

**Priority: MEDIUM**
**Complexity: MEDIUM**
**Basis:** MVSS-Net, ICCV 2021 / IEEE T-PAMI 2022. CECL-Net, Electronics 2024.

#### What it is

Regardless of how well colour or noise is matched, **physical tamper boundaries** leave edge artifacts at the splice boundary — brightness discontinuities, delamination shadows, misaligned printing edges. These are semantically agnostic and generalise across attack types.

Add an auxiliary decoder head that predicts a **tamper boundary mask** during training:

```
Fused Features (or intermediate backbone features)
  ├── Classification Head (existing — primary output)
  └── Boundary Segmentation Head (NEW — auxiliary, training only)
       → FPN-style upsampling decoder
       → Binary mask: tampered region boundary pixels
       → Trained with Dice Loss + BCE Loss
       → Ground truth: dilated edge of tamper mask annotation
```

This auxiliary task forces the backbone to learn spatially precise features. At inference, only the classification head is needed — the boundary head can be dropped after training.

Requires tamper boundary annotations in the training data. If only binary image-level labels exist, pseudo-boundary masks can be generated from the ground-truth tamper region masks.

---

### 8.7 Noiseprint++ Integration

**Priority: HIGH**
**Complexity: LOW (plug-and-play)**
**Basis:** TruFor, CVPR 2023. EdgeDoc, ICCV 2025 DeepID Challenge (3rd place).

#### What it is

**Noiseprint** (Cozzolino & Verdoliva, IEEE T-IFS 2020) is a Siamese CNN pretrained on camera model pairs. It extracts a **camera-model fingerprint** — the intrinsic noise pattern produced by a specific camera model — that is suppressed of scene content but preserves model-related imaging artifacts.

When an image is spliced, the noiseprint is **inconsistent** across the splice boundary because the spliced region came from a different camera model.

**Advantages over PRNU:**
- Does not require pristine reference images from the same device (fully blind forensics)
- Works at camera-model level (not device level), so it detects cross-camera splicing even with the same phone model

**Noiseprint++** (used in TruFor) is the self-supervised improved version.

```python
# From TruFor open-source (github.com/grip-unina/TruFor)
# Frozen pretrained weights — no training required
noiseprint_encoder = load_noiseprint_model(weights_path)
noiseprint_encoder.eval()
for param in noiseprint_encoder.parameters():
    param.requires_grad = False

# In forward pass:
noise_map = noiseprint_encoder(images)  # (B, 1, H, W)
# Feed noise_map as additional stream into fusion
```

This is pure inference — no additional training of the Noiseprint model. EdgeDoc at ICCV 2025 placed 3rd in the DeepID Challenge by fusing Noiseprint++ with RGB through a convolutional-transformer architecture.

**Open-source:** `github.com/grip-unina/TruFor`

---

### 8.8 Realistic Tamper Augmentation

**Priority: HIGH (addresses the data gap directly)**
**Complexity: MEDIUM**
**Basis:** General adversarial robustness practice. DF40 NeurIPS 2024 benchmark.

#### The Problem

The model achieves good performance on its current test set because the training distribution closely matches it. If tampered training images use crude face swaps (different lighting, obvious Cb mismatch), the model learns to exploit that crude signal — and fails against sophisticated attacks that normalise Cb.

#### Synthetic Tamper Generation Pipeline

Generate adversarially-augmented tampered samples on-the-fly during training:

```python
def generate_adversarial_tamper(genuine_card, face_pool, augment_config):
    """
    Take a genuine card, paste a face from the pool,
    apply adversarial augmentations that simulate the adversary
    trying to hide the tampering.
    """
    # 1. Randomly select a face from another genuine card
    donor_face = sample_face(face_pool)

    # 2. Paste onto the genuine card's face region
    tampered = paste_face(genuine_card, donor_face)

    # 3. Apply adversarial augmentations to reduce forensic traces
    if random() < 0.5:
        tampered = histogram_match_face(tampered)  # Cb matching attack
    if random() < 0.3:
        tampered = add_gaussian_noise_to_face(tampered)  # Noise normalisation
    if random() < 0.4:
        tampered = re_jpeg_compress(tampered, quality=random_int(75, 95))  # Re-save
    if random() < 0.3:
        tampered = blur_face_edges(tampered, sigma=random_float(0.5, 1.5))  # Edge smoothing

    return tampered
```

| Augmentation | Attack It Simulates |
|---|---|
| Histogram matching of Cb channel | Colour-matched splice attack |
| Gaussian noise addition to spliced region | Noise-level normalisation |
| Re-JPEG compression after splicing | Screen recapture / digital resave |
| Slight blur on inserted photo | Different sharpness from different camera |
| Brightness/contrast jitter on face only | Different exposure from different camera |
| Perspective warp on inserted face | Non-frontal or tilted photo pasted in |
| Random JPEG quality on donor face | Varied compression history |

Training on these harder samples forces the model to use the most robust forensic signals rather than relying on crude artefacts.

---

### 8.9 Deformable Patch Selection

**Priority: MEDIUM**
**Complexity: HIGH**
**Basis:** TruFor (CVPR 2023), TransForensics (ICCV 2021).

#### The Problem

`BackgroundRegionExtractor` uses **5 hardcoded region coordinates** from `configs/regions.yaml`. This has two failure modes:
1. **Misaligned cards** — rotation, perspective distortion, partial cropping shift the face region outside the expected coordinates
2. **Unexpected tampering locations** — the adversary may alter a region not covered by the hardcoded zones

#### Proposed Change

Replace the fixed region extractor with an **attention-based adaptive patch selector**:

```
Input Image
  → ViT patch tokenizer (non-overlapping 16×16 patches → 196 tokens)
  → Self-attention computes per-patch entropy / forensic saliency
  → Top-K most informative patches forwarded to CbEncoder
  → No dependency on precise card coordinates
```

Alternatively, use **Deformable Convolution** to allow the patch grid to shift adaptively based on learned offsets, making it robust to minor card misalignments without a full ViT replacement.

---

### 8.10 ViT Backbone Upgrade

**Priority: MEDIUM**
**Complexity: HIGH**
**Basis:** IML-ViT (arXiv 2307.14863, 2023). HiFi-IFDL, CVPR 2023. TruFor, CVPR 2023.

#### What it is

Replace or augment the EfficientNet-B3 RGB backbone with a **Vision Transformer (ViT)**. ViT's global self-attention captures long-range dependencies that CNNs miss — particularly useful for:
- Detecting inconsistency between a face photo and the surrounding card background
- Capturing lighting and shadow inconsistencies across distant regions
- Detecting semantic-level identity mismatch (face doesn't match ghost watermark)

#### Recommended Approach

Rather than a full ViT replacement (expensive, needs more data), use a **hybrid CNN-Transformer**:

```
EfficientNet-B3 (frozen/lightly tuned) → local feature maps
          +
ViT Encoder (pretrained DINO/DINOv2) → global context tokens
          ↓
Cross-attention fusion of CNN local features and ViT global tokens
          ↓
Combined feature for classification
```

**IML-ViT** (2023) showed that ViT with high-resolution capacity and multi-scale feature extraction outperforms CNN-based SOTA on image manipulation localisation benchmarks. The key modification: high-resolution ViT (patch size 8 instead of 16) to preserve fine-grained forensic detail.

---

## 9. Proposed Redesigned Architecture

Combining the highest-impact enhancements into a unified **Tri-Stream + Multi-Loss** architecture:

```
Input Image (224×224)
        │
  ┌─────┼──────────────────┐
  │     │                  │
[RGB]  [Cb+Cr]           [SRM / ELA]
  │       │                   │
EfficientNet-B3  CbCr          Noise/ELA
(1536D)      Encoder          Encoder
  │          (256D)           (128D)
  │             │                │
  └──────┬──────┘                │
         │                       │
  Cross-Modal Attn               │
  (RGB ↔ CbCr patches)           │
         │                       │
         └────────┬──────────────┘
                  │
          Concat + Projection (512D → 256D)
                  │
         ┌────────┴────────┐
         │                 │
  Classification     Boundary Mask
    Head (2-cls)     Head (auxiliary,
         │           training only)
  Genuine/Tampered
```

### Multi-Component Loss

```
L_total = 1.0 × L_cls
        + 0.3 × L_cbcr_consistency
        + 0.2 × L_srm_consistency
        + 0.3 × L_supervised_contrastive
        + 0.1 × L_boundary_dice_bce
```

### Incremental Adoption Path

Enhancements can be added progressively rather than all at once:

**Phase 1 (Low risk, high return):**
- Add Cr channel alongside Cb (minimal change)
- Add supervised contrastive loss to existing training
- Implement adversarial tamper augmentation

**Phase 2 (Moderate change):**
- Add SRM noise stream as a third encoder
- Add ELA computation as preprocessing + auxiliary branch
- Retrain end-to-end

**Phase 3 (Architecture redesign):**
- Integrate Noiseprint++ as frozen pretrained encoder
- Add boundary supervision head with annotated masks
- Consider ViT backbone for RGB stream

---

## 10. Enhancement Priority Table

| Enhancement | Impact | Complexity | Research Basis |
|---|---|---|---|
| **SRM high-frequency stream** | High — generalises to unseen attacks | Low | Luo CVPR'21, CFL-Net WACV'23 |
| **Supervised contrastive loss** | High — cross-distribution generalisation | Low | CFL-Net WACV'23, SeeABLE ICCV'23 |
| **Adversarial tamper augmentation** | High — directly closes the data gap | Medium | DF40 NeurIPS'24, general practice |
| **Noiseprint++ integration** | High — pretrained forensic features | Low (plug-and-play) | TruFor CVPR'23, EdgeDoc ICCV'25 |
| **Cb → CbCr (add Cr channel)** | Medium — complementary chrominance | Minimal | Chrominance forensics literature |
| **ELA auxiliary branch** | Medium — JPEG history forensics | Low | ELA-Enhanced 2024, Sci. Reports 2023 |
| **DCT compression stream** | Medium-High for digital tampering | Medium | CAT-Net WACV'21, ADCD-Net ICCV'25 |
| **Boundary supervision head** | Medium — forces spatial precision | Medium | MVSS-Net ICCV'21, CECL-Net 2024 |
| **Deformable patch selection** | Medium — layout robustness | High | TruFor CVPR'23, TransForensics ICCV'21 |
| **ViT backbone upgrade** | Medium — global attention | High | IML-ViT 2023, HiFi-IFDL CVPR'23 |

---

## 11. Key Configuration Files

### `configs/config.yaml` (key fields)

```yaml
model:
  backbone: efficientnet_b3
  pretrained: true
  num_classes: 2
  cb_encoder_dim: 128
  fusion_dim: 256
  patch_grid_size: 4
  dropout: 0.5

training:
  epochs: 60
  batch_size: 64
  learning_rate: 0.0005
  weight_decay: 0.01
  early_stopping_patience: 10
  grad_clip_norm: 1.0

loss:
  cls_weight: 1.0
  cb_weight: 0.3
  cb_margin: 0.5
  use_focal_loss: false

evaluation:
  threshold: 0.5
  far_constraint: 0.01   # FAR must be ≤ 1%
```

### `configs/regions.yaml` (structure)

```yaml
face_region:
  x1: 0.62
  y1: 0.21
  x2: 0.98
  y2: 0.88
# ... 4 more regions
```

All coordinates are normalised (0–1) relative to the image dimensions. To adapt to a different card layout, only this file needs modification.

---

## 12. References

### Architecture & Dual-Stream

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Luo et al., *"Generalizing Face Forgery Detection with High-Frequency Features"* | CVPR 2021 | SRM-based dual-stream; first to show HF features generalise cross-dataset |
| Niloy et al., *"CFL-Net: Image Forgery Localization Using Contrastive Learning"* | WACV 2023 | Contrastive loss + SRM for generalisation; ASPP multi-scale aggregation |
| Munawar et al., *"Explainable Dual-Stream Attention Network with Contrastive Learning"* | IET 2025 | DSCL-Net: RGB + SRM contrastive dual-stream |
| Liu et al., *"PSCC-Net: Progressive Spatio-Channel Correlation Network"* | IEEE T-CSVT 2022 | 50+ FPS progressive localization at 1080p |

### Document Forgery Detection

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Chen et al., *"Image Manipulation Detection by Multi-View Multi-Scale Supervision (MVSS-Net)"* | ICCV 2021 | Edge-supervised + noise-sensitive dual branch; auxiliary boundary supervision |
| Wong et al., *"ADCD-Net"* | ICCV 2025 (arXiv 2507.16397) | Adaptive DCT alignment score for documents; handles resize/crop misalignment |
| George et al., *"EdgeDoc"* | ICCV 2025 DeepID Challenge | Noiseprint++ + RGB hybrid CNN-Transformer for ID document forgery |
| *DocForgeNet* | Springer 2024 | Parallel CNN+Transformer on RGB+DCT for scanned document forgery |

### Chrominance & Color Forensics

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| *"Effective Image Splicing Detection Based on Image Chroma"* | ResearchGate (classic) | Cb-channel Markov transitions; 95.5% on CASIA v2.0 |
| *"Detecting Digital Image Splicing in Chroma Spaces"* | EURASIP / Springer | Optimal chroma channel design for passive splicing detection |
| *"Chrominance and Luminance Study for Deepfake Detection"* | Multimedia Tools 2025 | CbCr more sensitive than Y for deepfake artifacts |

### Noise & JPEG Forensics

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Cozzolino & Verdoliva, *"Noiseprint: A CNN-Based Camera Model Fingerprint"* | IEEE T-IFS 2020 | Camera-model noise fingerprint; blind forensics, no reference needed |
| Guillaro et al., *"TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection"* | CVPR 2023 | Noiseprint++ + DINO ViT fusion; pixel-level localization + image-level score |
| Kwon et al., *"CAT-Net: Compression Artifact Tracing Network"* | WACV 2021 | DCT binary volume stream; detects double-JPEG compression traces |
| *"Learning JPEG Compression Artifacts for Image Manipulation Detection"* | IJCV 2022 | Deep learning on JPEG artifact patterns |

### ELA

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| *"ELA-Enhanced Dual-Branch Deep Learning Model"* | Preprint 2024 | ResNet50 RGB + ELA CNN dual-branch; 3-class forgery + localization |
| *"Deep Fake Detection with ELA"* | Scientific Reports 2023 | +2.7% accuracy from ELA at 5.6% overhead; ResNet18+KNN best |

### Vision Transformers

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Ma et al., *"IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer"* | arXiv 2307.14863, 2023 | First pure ViT benchmark for manipulation localization; high-res ViT with multi-scale |
| Guo et al., *"HiFi-IFDL: Hierarchical Fine-Grained Image Forgery Detection and Localization"* | CVPR 2023 | Multi-branch hierarchical; 96.8% AUC covering 13 forgery methods |

### Contrastive & Face Forgery

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Larue et al., *"SeeABLE: Soft Discrepancies and Bounded Contrastive Learning for Exposing Deepfakes"* | ICCV 2023 | One-class contrastive; trained only on real faces, SOTA on FF++ and DFDC |
| Huang et al., *"Implicit Identity Driven Deepfake Face Swapping Detection"* | CVPR 2023 | Identity semantic encoder for face-background consistency |

### Datasets & Benchmarks

| Resource | Purpose |
|----------|---------|
| CASIA v2.0 | Standard splicing + copy-move forgery benchmark |
| DocTamper (170k images) | Large-scale document tamper detection |
| CMID (893 images) | Copy-move forgery in ID documents specifically |
| SIDTD (Nature Scientific Data 2024) | Synthetic ID/travel document forgery dataset |
| DF40 (NeurIPS 2024) | 40 forgery methods including identity swapping |

---

*Generated from architectural analysis of the `dual_stream` branch and survey of CVPR/ICCV/WACV 2021–2025 literature on image forgery detection.*
