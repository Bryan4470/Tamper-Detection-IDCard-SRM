"""
Training script for IC Card Tamper Detection using Two-Stream SRM Network.
Run from the SRM/ directory:
    python train.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# Change to src dir so xception weight path resolves correctly
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model_core import Two_Stream_Net

# ── Config ──────────────────────────────────────────────────────────────────
SRC_DIR     = '/mnt3/auto-ekyc/id_physical_tamper_new/data'   # genuine/ and tamper/ inside
CKPT_DIR    = '../checkpoints'
BATCH_SIZE  = 4
NUM_EPOCHS  = 30
LR          = 1e-4
IMAGE_SIZE  = 256
NUM_CLASSES = 2                  # genuine=0, tampered=1
VAL_SPLIT   = 0.15               # fraction of SRC_DIR images used for validation
SEED        = 42
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(CKPT_DIR, exist_ok=True)

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def make_splits(src_dir, val_fraction, seed):
    """Load image paths from CSVs under src_dir/genuine and src_dir/tamper.
    Each CSV must have an 'image_path' column.
    Returns (train_samples, val_samples) as [(abs_path, label), ...].
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    train_samples, val_samples = [], []

    for cls_name, label in [('genuine', 0), ('tamper', 1)]:
        cls_dir = os.path.join(src_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing folder: {cls_dir}")
            continue
        csv_files = sorted(f for f in os.listdir(cls_dir) if f.endswith('.csv'))
        files = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(os.path.join(cls_dir, csv_file), dtype=str, keep_default_na=False)
                if 'image_path' not in df.columns:
                    print(f"[SKIP] {csv_file}: no 'image_path' column")
                    continue
                files.extend(p for p in df['image_path'] if os.path.exists(p))
            except Exception as e:
                print(f"[ERROR] {csv_file}: {e}")
        print(f"  {cls_name}: {len(files)} images from {len(csv_files)} CSV(s)")

        idx = rng.permutation(len(files))
        n_val = max(1, int(len(files) * val_fraction))
        train_samples.extend((files[i], label) for i in idx[n_val:])
        val_samples.extend((files[i],   label) for i in idx[:n_val])

    return train_samples, val_samples


class ICCardDataset(Dataset):
    """IC card dataset backed by a pre-built sample list [(path, label), ...]."""

    def __init__(self, samples, split_name='split', transform=None):
        self.samples   = samples
        self.transform = transform
        print(f"[{split_name}] {len(self.samples)} images loaded "
              f"({sum(1 for _,l in self.samples if l==0)} genuine, "
              f"{sum(1 for _,l in self.samples if l==1)} tamper)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# Transforms: resize, random flips for training, normalize to [-1,1]
train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),                          # [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],          # → [-1,1]
                         [0.5, 0.5, 0.5]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def train():
    # Build stratified in-memory split from source directory
    train_samples, val_samples = make_splits(SRC_DIR, VAL_SPLIT, SEED)

    # Datasets & loaders
    train_ds = ICCardDataset(train_samples, split_name='train', transform=train_tf)
    val_ds   = ICCardDataset(val_samples,   split_name='val',   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model
    model = Two_Stream_Net().to(DEVICE)
    print(f"Model loaded. Running on: {DEVICE}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, feats, att = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ───────────────────────────────────────────
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]  "):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(imgs)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)

        val_acc = v_correct / v_total if v_total > 0 else 0.0

        print(f"Epoch {epoch:3d} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        scheduler.step()

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CKPT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, ckpt_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f}) → {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    train()
