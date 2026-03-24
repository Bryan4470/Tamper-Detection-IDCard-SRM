"""
Training script for IC Card Tamper Detection using Two-Stream SRM Network.
Run from the SRM/ directory:
    python train.py
    python train.py --resume ../checkpoints/last_epoch.pth
    python train.py --epochs 50 --lr 5e-5
"""

import os
import sys
import random
import argparse
import logging
import pickle
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# Change to src dir so xception weight path resolves correctly
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model_core import Two_Stream_Net

# ── Config ──────────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

SRC_DIR       = _CFG['data']['src_dir']
CKPT_DIR      = '../checkpoints'
BATCH_SIZE    = _CFG['data']['batch_size']
NUM_EPOCHS    = _CFG['training']['epochs']
LR            = _CFG['training']['learning_rate']
MIN_LR        = _CFG['training']['min_lr']
WARMUP_EPOCHS = _CFG['training']['warmup_epochs']
WEIGHT_DECAY  = _CFG['training']['weight_decay']
GRAD_CLIP     = _CFG['training']['grad_clip']
IMAGE_SIZE    = _CFG['data']['image_size']
NUM_CLASSES   = 2                  # genuine=0, tampered=1
VAL_SPLIT     = _CFG['data']['val_split']
NUM_WORKERS   = _CFG['data']['num_workers']
PERSIST_WRKRS = _CFG['data'].get('persistent_workers', False) and NUM_WORKERS > 0
SEED          = _CFG.get('seed', 42)
DEVICE        = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# Early stopping
ES_PATIENCE        = _CFG['early_stopping']['patience']
ES_MIN_DELTA       = _CFG['early_stopping']['min_delta']

# Evaluation
TAMPER_THRESHOLD   = _CFG['evaluation']['tamper_threshold']  # must match predict.py
FAR_THRESHOLD      = _CFG['evaluation']['far_threshold']
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(CKPT_DIR, exist_ok=True)

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True   # speed boost for fixed input size
    print(f"Random seed: {seed}")


def make_splits(src_dir, val_fraction, seed):
    """Load image paths from per-class CSV files under src_dir/genuine and src_dir/tamper.
    Returns (train_samples, val_samples) as [(abs_path, label), ...].
    Caches paths in dataset_cache.pkl and stratified splits in splits.pkl for fast reloads.
    """
    cache_file  = os.path.join(src_dir, 'dataset_cache.pkl')
    splits_file = os.path.join(src_dir, 'splits.pkl')

    # ── Dataset cache ─────────────────────────────────────────────────────────
    all_paths, all_labels = None, None
    if os.path.exists(cache_file):
        try:
            print(f"Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            all_paths  = cached['image_paths']
            all_labels = cached['labels']
            print(f"  {len(all_paths)} images "
                  f"({all_labels.count(0)} genuine, {all_labels.count(1)} tamper)")
        except Exception as e:
            logging.warning(f"Cache load failed ({e}), rebuilding...")

    if all_paths is None:
        print("Scanning CSVs (first run — will be cached)...")
        all_paths, all_labels = [], []
        for cls_name, label in [('genuine', 0), ('tamper', 1)]:
            cls_dir = os.path.join(src_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Missing folder: {cls_dir}")
                continue
            csv_files = sorted(f for f in os.listdir(cls_dir) if f.endswith('.csv'))
            for csv_file in csv_files:
                csv_path = os.path.join(cls_dir, csv_file)
                print(f"  {cls_name}/{csv_file}")
                try:
                    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
                    if 'image_path' not in df.columns:
                        print(f"    [SKIP] no 'image_path' column")
                        continue
                    count = 0
                    for img_path in df['image_path']:
                        if os.path.exists(img_path):
                            all_paths.append(img_path)
                            all_labels.append(label)
                            count += 1
                        else:
                            logging.warning(f"Not found: {img_path}")
                    print(f"    {count} images loaded")
                except Exception as e:
                    print(f"    [ERROR] {e}")
        print(f"\nTotal: {len(all_paths)} images "
              f"({all_labels.count(0)} genuine, {all_labels.count(1)} tamper)")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'image_paths': all_paths, 'labels': all_labels}, f)
            print(f"Saved dataset cache: {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    # ── Splits cache ──────────────────────────────────────────────────────────
    train_idx, val_idx = None, None
    if os.path.exists(splits_file):
        try:
            print(f"Loading splits from cache: {splits_file}")
            with open(splits_file, 'rb') as f:
                splits = pickle.load(f)
            if splits.get('n_total') == len(all_paths):
                train_idx = splits['train']
                val_idx   = splits['val']
            else:
                print("  Dataset size changed — regenerating splits...")
        except Exception as e:
            logging.warning(f"Splits cache load failed ({e}), regenerating...")

    if train_idx is None:
        rng = np.random.default_rng(seed)
        train_idx, val_idx = [], []
        for label in [0, 1]:
            cls_indices = [i for i, l in enumerate(all_labels) if l == label]
            perm = rng.permutation(len(cls_indices))
            n_val = max(1, int(len(cls_indices) * val_fraction))
            val_idx.extend(   cls_indices[perm[i]] for i in range(n_val))
            train_idx.extend( cls_indices[perm[i]] for i in range(n_val, len(cls_indices)))
        try:
            with open(splits_file, 'wb') as f:
                pickle.dump({'train': train_idx, 'val': val_idx,
                             'n_total': len(all_paths)}, f)
            print(f"Saved splits cache: {splits_file}")
        except Exception as e:
            logging.warning(f"Failed to save splits: {e}")

    train_samples = [(all_paths[i], all_labels[i]) for i in train_idx]
    val_samples   = [(all_paths[i], all_labels[i]) for i in val_idx]
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


train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5),           # slight scan angle variation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],     # ImageNet mean
                         [0.229, 0.224, 0.225]),    # ImageNet std
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def validate(model, loader, criterion, device, threshold=TAMPER_THRESHOLD):
    """Run validation and return FAR, FRR, F1, AUC, precision, recall, loss."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _, _ = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels_np = np.array(all_labels)
    probs_np  = np.array(all_probs)
    preds_np  = (probs_np >= threshold).astype(int)

    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far  = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # tamper accepted as genuine
    frr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # genuine rejected as tamper
    f1   = f1_score(labels_np, preds_np, zero_division=0)
    auc  = roc_auc_score(labels_np, probs_np) if len(np.unique(labels_np)) > 1 else 0.0
    prec = precision_score(labels_np, preds_np, zero_division=0)
    rec  = recall_score(labels_np, preds_np, zero_division=0)
    avg_loss = total_loss / len(labels_np)

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {'loss': avg_loss, 'f1': f1, 'auc': auc, 'far': far, 'frr': frr,
            'precision': prec, 'recall': rec, 'acc': acc,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, metrics,
                    best_val_f1, best_val_auc, best_frr_under_far, patience_counter):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_val_f1,
        'best_val_auc': best_val_auc,
        'best_frr_under_far': best_frr_under_far,
        'patience_counter': patience_counter,
        **{f'val_{k}': v for k, v in metrics.items()},
    }
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint and return restored training state."""
    print(f"Resuming from: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return {
        'start_epoch':        ckpt['epoch'] + 1,
        'best_val_f1':        ckpt.get('best_val_f1', 0.0),
        'best_val_auc':       ckpt.get('best_val_auc', 0.0),
        'best_frr_under_far': ckpt.get('best_frr_under_far', float('inf')),
        'patience_counter':   ckpt.get('patience_counter', 0),
    }


def train(resume_path=None, num_epochs=NUM_EPOCHS, lr=LR):
    set_random_seed(SEED)

    # Build stratified in-memory split from source directory
    train_samples, val_samples = make_splits(SRC_DIR, VAL_SPLIT, SEED)

    # Datasets & loaders
    train_ds = ICCardDataset(train_samples, split_name='train', transform=train_tf)
    val_ds   = ICCardDataset(val_samples,   split_name='val',   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=PERSIST_WRKRS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=PERSIST_WRKRS)

    # Model
    model = Two_Stream_Net().to(DEVICE)
    print(f"Model loaded. Running on: {DEVICE}")

    # Loss
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.25]).to(DEVICE)
    )

    # AdamW optimizer — start at MIN_LR so warmup ramps up correctly
    optimizer = optim.AdamW(model.parameters(), lr=MIN_LR, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine annealing via SequentialLR (avoids scheduler init conflict)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=MIN_LR / lr, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - WARMUP_EPOCHS, eta_min=MIN_LR
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    # Mixed precision
    use_amp = (DEVICE == 'cuda')
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # Tracking (may be overwritten by resume)
    start_epoch          = 1
    best_val_f1          = 0.0
    best_val_auc         = 0.0
    best_frr_under_far   = float('inf')
    patience_counter     = 0

    # Resume
    if resume_path:
        state = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, DEVICE)
        start_epoch        = state['start_epoch']
        best_val_f1        = state['best_val_f1']
        best_val_auc       = state['best_val_auc']
        best_frr_under_far = state['best_frr_under_far']
        patience_counter   = state['patience_counter']
        print(f"Resumed at epoch {start_epoch} | best_f1={best_val_f1:.4f} | "
              f"best_auc={best_val_auc:.4f} | patience={patience_counter}/{ES_PATIENCE}")

    ckpt_kwargs = lambda m: dict(
        best_val_f1=best_val_f1,
        best_val_auc=best_val_auc,
        best_frr_under_far=best_frr_under_far,
        patience_counter=patience_counter,
    )

    val_m = {}
    for epoch in range(start_epoch, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _ = model(imgs)
                loss = criterion(logits, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        scheduler.step()

        # ── Validate ───────────────────────────────────────────
        val_m = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch:3d} | lr={get_lr(optimizer):.2e} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_acc={val_m['acc']:.4f} | val_f1={val_m['f1']:.4f} | val_auc={val_m['auc']:.4f} | "
              f"prec={val_m['precision']:.4f} | rec={val_m['recall']:.4f} | "
              f"FAR={val_m['far']*100:.2f}% | FRR={val_m['frr']*100:.2f}%")

        if epoch % 5 == 0:
            log_path = os.path.join(CKPT_DIR, 'train_log.csv')
            write_header = not os.path.exists(log_path)
            with open(log_path, 'a') as log_f:
                if write_header:
                    log_f.write('epoch,lr,train_loss,train_acc,val_acc,val_f1,val_auc,far,frr\n')
                log_f.write(f"{epoch},{get_lr(optimizer):.2e},{train_loss:.4f},{train_acc:.4f},"
                            f"{val_m['acc']:.4f},{val_m['f1']:.4f},{val_m['auc']:.4f},"
                            f"{val_m['far']*100:.2f},{val_m['frr']*100:.2f}\n")

        # ── Checkpoints ────────────────────────────────────────
        if val_m['f1'] > best_val_f1 + ES_MIN_DELTA:
            best_val_f1 = val_m['f1']
            patience_counter = 0
            save_checkpoint(os.path.join(CKPT_DIR, 'best_f1.pth'),
                            epoch, model, optimizer, scheduler, scaler, val_m,
                            **ckpt_kwargs(val_m))
            print(f"  ✓ Saved best_f1.pth  (f1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        if val_m['auc'] > best_val_auc:
            best_val_auc = val_m['auc']
            save_checkpoint(os.path.join(CKPT_DIR, 'best_auc.pth'),
                            epoch, model, optimizer, scheduler, scaler, val_m,
                            **ckpt_kwargs(val_m))
            print(f"  ✓ Saved best_auc.pth (auc={best_val_auc:.4f})")

        if val_m['far'] <= FAR_THRESHOLD and val_m['frr'] < best_frr_under_far:
            best_frr_under_far = val_m['frr']
            save_checkpoint(os.path.join(CKPT_DIR, 'best_frr_under_far.pth'),
                            epoch, model, optimizer, scheduler, scaler, val_m,
                            **ckpt_kwargs(val_m))
            print(f"  ✓ Saved best_frr_under_far.pth "
                  f"(frr={best_frr_under_far*100:.2f}% @ far<={FAR_THRESHOLD*100:.0f}%)")

        # ── Early stopping ─────────────────────────────────────
        if patience_counter >= ES_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no F1 improvement for {ES_PATIENCE} epochs)")
            break

    if val_m:
        save_checkpoint(os.path.join(CKPT_DIR, 'last_epoch.pth'),
                        epoch, model, optimizer, scheduler, scaler, val_m,
                        **ckpt_kwargs(val_m))

    print(f"\nTraining complete.")
    print(f"  Best F1:  {best_val_f1:.4f}")
    print(f"  Best AUC: {best_val_auc:.4f}")
    if best_frr_under_far < float('inf'):
        print(f"  Best FRR under FAR<={FAR_THRESHOLD*100:.0f}%: {best_frr_under_far*100:.2f}%")
    else:
        print(f"  No epoch met FAR<={FAR_THRESHOLD*100:.0f}% constraint")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IC Card Tamper Detection')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'Learning rate (default: {LR})')
    args = parser.parse_args()

    train(resume_path=args.resume, num_epochs=args.epochs, lr=args.lr)
