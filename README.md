# RGB+Cb Tamper Detection

Dual-stream deep learning model for detecting tampered ID cards using RGB semantic features and Cb chrominance color consistency analysis.

## Overview

**Why Cb Channel?**

Tampered ID cards often look visually convincing in RGB, but exhibit color inconsistencies in the Cb (chrominance-blue) channel:

| Scenario | RGB Appearance | Cb Channel |
|----------|----------------|------------|
| Genuine ID | Normal | Consistent color values within regions |
| Tampered ID | Looks normal | Inconsistent - different image sources have different Cb characteristics |

**Example:** When someone replaces a face photo, the new face comes from a different camera/lighting. Even if RGB looks seamless, Cb values differ from the original card's color profile.

**How the Model Works:**
1. **RGB Stream**: EfficientNet extracts semantic features (what's in the image)
2. **Cb Stream**: Analyzes color consistency across predefined regions
3. **Fusion**: Combines both streams to detect tampering

---

## Table of Contents

- [Quick Start](#quick-start)
- [Inference](#inference)
- [Training](#training)
- [Configuration Reference](#configuration-reference)
- [Customization](#customization)
- [Architecture](#architecture)
- [Project Structure](#project-structure)

---

## Quick Start

### Docker (Recommended)

```bash
# 1. Start and enter container
docker compose up -d
docker compose exec -it tamper-detection bash

# 2. Train model (inside container)
python scripts/train.py --config configs/config.yaml

# 3. Run inference (inside container)
python scripts/inference.py \
  --model checkpoints/20250209_120000/best_f1.pth \
  --image data/test.jpg
```

Other useful commands:
```bash
docker compose down          # Stop container
docker compose up -d --build # Rebuild after code changes
```

### Local (Without Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train
python scripts/train.py --config configs/config.yaml

# 3. Inference
python scripts/inference.py \
  --model checkpoints/20250209_120000/best_f1.pth \
  --image data/test.jpg
```

---

## Inference

API server starts automatically with `docker compose up -d` on port 7500.

**`crop` parameter:** Add `crop=true` if input images are **original/uncropped** photos. This auto-crops via AI engine API before prediction.

| crop | Input Image |
|------|-------------|
| `false` (default) | Cropped ID card |
| `true` | Original photo (auto-crop first) |

### Single Image

**API:**
```bash
curl -X POST "http://localhost:7500/predict" -F "file=@test.jpg"
curl -X POST "http://localhost:7500/predict?crop=true" -F "file=@original.jpg"
```

**CLI** (inside container):
```bash
python scripts/inference.py --model checkpoints/best_f1.pth --image test.jpg
python scripts/inference.py --model checkpoints/best_f1.pth --image original.jpg --crop
```

### Batch (Directory)

**API:**
```bash
curl -X POST "http://localhost:7500/predict_batch?directory=/app/data/test_images&output=/app/output/results.csv"
curl -X POST "http://localhost:7500/predict_batch?directory=/app/data/test_images&output=/app/output/results.csv&crop=true"
```

**CLI** (inside container):
```bash
python scripts/inference.py --model checkpoints/best_f1.pth --directory data/test_images/ --output results.csv
python scripts/inference.py --model checkpoints/best_f1.pth --directory data/test_images/ --output results.csv --crop
```

### Batch (CSV)

**API:**
```bash
curl -X POST "http://localhost:7500/predict_batch?csv=/app/data/test_list.csv&output=/app/output/results.csv"
```

**CLI** (inside container):
```bash
python scripts/inference.py --model checkpoints/best_f1.pth --csv test_list.csv --output results.csv
python scripts/inference.py --model checkpoints/best_f1.pth --csv test_list.csv --output results.csv --crop
```

### Output Format

```csv
image_path,prediction,confidence,prob_genuine,prob_tampered
/app/data/img1.jpg,genuine,0.92,0.92,0.08
/app/data/img2.jpg,tampered,0.85,0.15,0.85
```

### Configuration

Model path is configured in `docker-compose.yml`:

```yaml
environment:
  - MODEL_PATH=/mnt2/auto-ekyc/id_physical_tamper_new/_dual_stream/trained_model/tamper_with_color_space.pth  # Default
  - CONFIG_PATH=/app/configs/config.yaml
  - THRESHOLD=0.5
```

After changing, restart the container:
```bash
docker compose down && docker compose up -d
```

---

## Training

### 1. Prepare Dataset

```
data/
├── genuine/              # Genuine (non-tampered) images
│   ├── dataset1.csv
│   └── dataset2.csv
└── tamper/               # Tampered images
    ├── dataset1.csv
    └── dataset2.csv
```

Each CSV file:
```csv
image_path
/full/path/to/image1.jpg
/full/path/to/image2.jpg
```

**Note:** Use absolute paths or paths relative to container's `/app/data` directory.

**Optional:** External test dataset (can be anywhere):
```
testing_dataset/
├── color_ghost.csv
├── cover_face.csv
└── digital_tamper_face.csv
```

### 2. Loss Functions & Key Innovation

The model minimizes: **`L_total = L_classification + 0.3 × L_cb_consistency`**

| Loss | Description |
|------|-------------|
| **Classification** | CrossEntropyLoss (default); Focal Loss available for class imbalance |
| **Cb Consistency** | Exploits tampered regions showing inconsistent chroma (Cb channel) patterns |

**How Cb Consistency Loss works:**
- **Genuine images:** Minimize pairwise distances within regions (patches should be consistent)
- **Tampered images:** Margin-based loss — at least one region must show inconsistency (distance > margin)

### 3. Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 60 | Maximum training epochs |
| `learning_rate` | 0.0005 | Initial learning rate |
| `batch_size` | 64 | Images per batch |
| `weight_decay` | 0.01 | AdamW regularization |
| `cb_consistency_weight` | 0.3 | Weight for Cb consistency loss |
| `cb_margin` | 0.5 | Margin threshold for tampered images |
| `val_split` | 0.2 | Train/validation split (stratified) |

**Augmentation:** Rotation ±5°, Gaussian blur 10%. Color augmentation intentionally excluded to preserve Cb channel consistency signal.

Edit `configs/config.yaml` to customize. See [Configuration Reference](#configuration-reference) for all options.

### 4. Start Training

All commands below are run **inside the container**.

```bash
python scripts/train.py --config configs/config.yaml
```

### 5. Monitor Progress

```bash
# View logs (run from host, outside container)
docker compose logs -f tamper-detection

# TensorBoard (inside container)
tensorboard --logdir logs --host 0.0.0.0
# Open http://localhost:6006
```

**Metrics logged:** Train loss (total, classification, Cb consistency), Validation F1, AUC, FAR/FRR curves.

### 6. Early Stopping

| Setting | Value |
|---------|-------|
| Monitor metric | Validation F1 score |
| Patience | 10 epochs without improvement |
| Behavior | Training stops automatically when plateau detected |

### 7. Resume from Checkpoint

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --resume checkpoints/20250209_120000/last_epoch.pth
```

### 8. Training Outputs

```
checkpoints/20250209_120000/
├── best_f1.pth              # Best F1 score
├── best_auc.pth             # Best AUC-ROC
├── best_frr_under_far.pth   # Best FRR under FAR constraint
├── last_epoch.pth           # Latest (for resuming)
└── eval_results/
    ├── best_f1_results.txt
    └── best_auc_results.txt
```

**Checkpoint Saving Strategy:**

| Checkpoint | Selection Criteria | Recommended Use Case |
|------------|-------------------|----------------------|
| `best_f1.pth` | Highest validation F1 | Balanced precision/recall |
| `best_auc.pth` | Highest validation AUC | Overall ranking quality |
| `best_frr_under_far.pth` | Lowest FRR when FAR ≤ 1% | Production (security-focused) |
| `last_epoch.pth` | Final training epoch | Resume interrupted training |

---

## Configuration Reference

### configs/config.yaml

```yaml
# Data
data:
  root_dir: "./data"              # Contains genuine/ and tamper/ subdirs
  image_size: 224
  batch_size: 64
  num_workers: 8
  val_split: 0.2                  # 20% for validation
  test_split: 0.0

  test_dataset:                   # Optional external test set
    enabled: true
    csv_dir: "./data/testing_dataset"
    csv_files: []                 # Empty = load all CSVs in csv_dir

# Model
model:
  backbone: "efficientnet_b3"     # resnet18, resnet50, efficientnet_b0, efficientnet_b3
  pretrained: true
  dropout: 0.5
  cb_encoder_dim: 128
  fusion_dim: 256
  patch_grid_size: 4              # 4x4 = 16 patches per region

# Training
training:
  epochs: 60
  learning_rate: 0.0005
  weight_decay: 0.01
  optimizer: "adamw"

  loss:
    classification_weight: 1.0
    cb_consistency_weight: 0.3    # Cb consistency strength
    cb_margin: 0.5

  scheduler:
    type: "cosine"
    warmup_epochs: 5
    min_lr: 0.00005

  early_stopping:
    patience: 10
    monitor_metric: "f1_score"

# Evaluation
evaluation:
  classification_threshold: 0.5
  far_constraint_threshold: 0.01  # For best_frr_under_far.pth

# Device
device:
  use_cuda: true
  gpu_id: 0
  mixed_precision: true
```

### configs/regions.yaml

Defines regions for Cb consistency analysis. Coordinates are normalized (0.0-1.0):

```yaml
face:
  x1: 0.62
  y1: 0.21
  x2: 0.98
  y2: 0.88
  description: "Face photo region"

id_number:
  x1: 0.02
  y1: 0.21
  x2: 0.38
  y2: 0.32
  description: "ID number text region"

name_address:
  x1: 0
  y1: 0.53
  x2: 0.65
  y2: 1
  description: "Name and address region"

ghost_face:
  x1: 0.45
  y1: 0.25
  x2: 0.7
  y2: 0.58
  description: "Ghost face region"

islam_gender:
  x1: 0.55
  y1: 0.8
  x2: 1
  y2: 1
  description: "Religion and gender region"
```

**Note:** These regions are for Malaysian IC cards. Modify if using different card layouts.

---

## Customization

### FAR Threshold

Controls when `best_frr_under_far.pth` is saved:

```yaml
evaluation:
  far_constraint_threshold: 0.01   # Default: 1%
```

| Scenario | Threshold | Effect |
|----------|-----------|--------|
| High security | `0.001` | Stricter, higher FRR |
| Balanced | `0.01` | Default |
| User-friendly | `0.05` | More permissive |

### Backbone

```yaml
model:
  backbone: "efficientnet_b3"
```

| Backbone | Params | Speed | Accuracy | Use Case |
|----------|--------|-------|----------|----------|
| `resnet18` | 11M | Fast | Lower | Quick experiments |
| `resnet50` | 25M | Medium | Medium | Balanced |
| `efficientnet_b0` | 5M | Fast | Medium | Edge deployment |
| `efficientnet_b3` | 12M | Medium | Higher | Production |

### Adding Regions

Edit `configs/regions.yaml`:

```yaml
new_region:
  x1: 0.05      # Left edge (5% from left)
  y1: 0.35      # Top edge (35% from top)
  x2: 0.40      # Right edge
  y2: 0.45      # Bottom edge
  description: "New region description"
```

**Tips:**
1. Use image viewer to get pixel coordinates
2. Divide by image dimensions → normalized values
3. Choose regions with expected color consistency
4. Avoid overlapping regions

### Cb Consistency Loss

```yaml
training:
  loss:
    cb_consistency_weight: 0.3   # Higher = stronger enforcement
    cb_margin: 0.5
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│          Input Image (RGB)                   │
└─────────────────┬───────────────────────────┘
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
┌─────────────┐         ┌─────────────┐
│ RGB Stream  │         │  Cb Stream  │
│ EfficientNet│         │ Lightweight │
│    -B3      │         │ CNN Encoder │
└──────┬──────┘         └──────┬──────┘
       │                       │
       │ Semantic Features     │ Color Consistency
       │                       │ Features
       └───────────┬───────────┘
                   ▼
         ┌─────────────────┐
         │ Cross-Modal     │
         │ Fusion          │
         │ (Multi-head     │
         │  Attention)     │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ Classification  │
         │ Genuine/Tampered│
         └─────────────────┘
```

**Components:**
- **RGB Stream**: Semantic features via pretrained EfficientNet
- **Cb Stream**: 5 regions × 16 patches each → color consistency features
- **Cross-Modal Fusion**: Multi-head attention combines both streams
- **Output**: Binary classification (genuine vs tampered)

### Performance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **F1 Score** | Precision-recall balance | Overall performance |
| **AUC-ROC** | Ranking quality | Threshold-independent |
| **FAR** | Tampered → Genuine errors | Security critical |
| **FRR** | Genuine → Tampered errors | User experience |

---

## Project Structure

```
rgb_cb_tamper_detection/
├── src/
│   ├── models/
│   │   ├── model.py        # RGBCbTamperDetector
│   │   ├── cb_utils.py     # Cb channel extraction
│   │   └── losses.py       # CombinedLoss
│   └── data/
│       ├── dataloader.py   # Data loading
│       └── augmentation.py # Augmentation
├── scripts/
│   ├── train.py            # Training
│   └── inference.py        # Inference
├── configs/
│   ├── config.yaml         # Training config
│   └── regions.yaml        # Region definitions
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
