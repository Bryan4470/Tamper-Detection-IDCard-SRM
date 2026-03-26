"""
Training script for IC Card Tamper Detection using Two-Stream SRM Network.
Run from the SRM/ directory:
    python train.py
    python train.py --resume checkpoints/best_acc.pth
"""

import os
import sys
import argparse
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
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# Add src to path so imports work
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'src'))
# Change to src dir so xception weight path resolves correctly
os.chdir(os.path.join(_THIS_DIR, 'src'))

from model_core import Two_Stream_Net

# ── Config ──────────────────────────────────────────────────────────────────
SRC_DIR     = '/mnt3/auto-ekyc/id_physical_tamper_new/data'   # genuine/ and tamper/ inside
CKPT_DIR    = '../checkpoints'
METRICS_CSV = '../checkpoints/training_metrics.csv'
LOG_INTERVAL = 3            # log metrics every N epochs
BATCH_SIZE  = 18
NUM_WORKERS = 6             # dataloader workers (0 = main process only)
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
    transforms.Normalize([0.5, 0.5, 0.5],          # -> [-1,1]
                         [0.5, 0.5, 0.5]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, ckpt_path):
    """Save checkpoint with all training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': metrics['best_acc'],
        'best_auc': metrics['best_auc'],
        'best_f1': metrics['best_f1'],
        'val_acc': metrics.get('val_acc', 0.0),
        'val_auc': metrics.get('val_auc', 0.0),
        'val_f1': metrics.get('val_f1', 0.0),
    }, ckpt_path)


def load_checkpoint(ckpt_path, model, optimizer, scheduler):
    """Load checkpoint and return start epoch and best metrics."""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    start_epoch = ckpt['epoch'] + 1
    best_metrics = {
        'best_acc': ckpt.get('best_acc', ckpt.get('val_acc', 0.0)),
        'best_auc': ckpt.get('best_auc', 0.0),
        'best_f1': ckpt.get('best_f1', 0.0),
    }
    print(f"Resumed from epoch {ckpt['epoch']} | "
          f"best_acc={best_metrics['best_acc']:.4f} | "
          f"best_auc={best_metrics['best_auc']:.4f} | "
          f"best_f1={best_metrics['best_f1']:.4f}")
    return start_epoch, best_metrics


def train(resume_path=None):
    # Build stratified in-memory split from source directory
    train_samples, val_samples = make_splits(SRC_DIR, VAL_SPLIT, SEED)

    # Datasets & loaders
    train_ds = ICCardDataset(train_samples, split_name='train', transform=train_tf)
    val_ds   = ICCardDataset(val_samples,   split_name='val',   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = Two_Stream_Net().to(DEVICE)
    print(f"Model loaded. Running on: {DEVICE}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Initialize best metrics
    start_epoch = 1
    best_metrics = {'best_acc': 0.0, 'best_auc': 0.0, 'best_f1': 0.0}

    # Resume from checkpoint if specified
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_metrics = load_checkpoint(resume_path, model, optimizer, scheduler)
    elif resume_path:
        print(f"[WARN] Checkpoint not found: {resume_path}, starting from scratch")

    # Initialize metrics log for CSV
    metrics_log = []

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
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
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]  "):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(imgs)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        val_acc = v_correct / v_total if v_total > 0 else 0.0

        # Compute metrics
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)

        cm = confusion_matrix(all_labels_np, all_preds_np, labels=[0, 1])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        far = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        frr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        val_f1 = f1_score(all_labels_np, all_preds_np)
        try:
            val_auc = roc_auc_score(all_labels_np, all_probs_np)
        except ValueError:
            val_auc = 0.0

        print(f"Epoch {epoch:3d} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        print(f"          | FAR={far*100:.2f}% | FRR={frr*100:.2f}% | F1={val_f1:.4f} | AUC={val_auc:.4f}")

        scheduler.step()

        # Current metrics for saving
        current_metrics = {
            'best_acc': best_metrics['best_acc'],
            'best_auc': best_metrics['best_auc'],
            'best_f1': best_metrics['best_f1'],
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1,
        }

        # Save best accuracy checkpoint
        if val_acc >= best_metrics['best_acc']:
            best_metrics['best_acc'] = val_acc
            current_metrics['best_acc'] = val_acc
            ckpt_path = os.path.join(CKPT_DIR, 'best_acc.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path)
            print(f"  -> Saved best ACC model (val_acc={val_acc:.4f}) -> {ckpt_path}")

        # Save best AUC checkpoint
        if val_auc >= best_metrics['best_auc']:
            best_metrics['best_auc'] = val_auc
            current_metrics['best_auc'] = val_auc
            ckpt_path = os.path.join(CKPT_DIR, 'best_auc.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path)
            print(f"  -> Saved best AUC model (val_auc={val_auc:.4f}) -> {ckpt_path}")

        # Save best F1 checkpoint
        if val_f1 >= best_metrics['best_f1']:
            best_metrics['best_f1'] = val_f1
            current_metrics['best_f1'] = val_f1
            ckpt_path = os.path.join(CKPT_DIR, 'best_f1.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path)
            print(f"  -> Saved best F1 model (val_f1={val_f1:.4f}) -> {ckpt_path}")

        # Save latest checkpoint (for resuming interrupted training)
        latest_path = os.path.join(CKPT_DIR, 'latest.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, latest_path)

        # Log metrics every LOG_INTERVAL epochs
        if epoch % LOG_INTERVAL == 0:
            metrics_row = {
                'epoch': epoch,
                'train_loss': round(train_loss, 3),
                'train_acc': round(train_acc, 3),
                'val_acc': round(val_acc, 3),
                'FAR': round(far, 5),
                'FRR': round(frr, 5),
                'val_f1': round(val_f1, 3),
                'val_auc': round(val_auc, 3),
            }
            metrics_log.append(metrics_row)
            # Save CSV after each log interval
            metrics_df = pd.DataFrame(metrics_log)
            metrics_df.to_csv(METRICS_CSV, index=False)
            print(f"  -> Metrics saved to {METRICS_CSV}")

    print(f"\nTraining complete.")
    print(f"  Best ACC: {best_metrics['best_acc']:.4f}")
    print(f"  Best AUC: {best_metrics['best_auc']:.4f}")
    print(f"  Best F1:  {best_metrics['best_f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IC Card Tamper Detection')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/latest.pth)')
    args = parser.parse_args()

    train(resume_path=args.resume)
