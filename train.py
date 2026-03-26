"""
Training script for IC Card Tamper Detection.

Supports:
- Two-Stream Network (RGB + SRM)
- Three-Stream Network (RGB + SRM + CB)

Training Phases:
    Phase 1: Two-Stream training (RGB + SRM backbone)
    Phase 2: CB stream training (freeze backbone, train CB + fusion)
    Phase 3: Joint fine-tuning (all three streams)

Run from the SRM/ directory:
    # Phase 1: Two-Stream training (30 epochs)
    python train.py

    # Phase 2: CB stream training with frozen backbone (15 epochs)
    python train.py --use-three-stream --freeze-backbone --init-from checkpoints/best_auc.pth

    # Phase 3: Joint fine-tuning (30 epochs)
    python train.py --use-three-stream --resume checkpoints/three_stream_latest.pth

    # Custom epoch count
    python train.py --use-three-stream --freeze-backbone --init-from checkpoints/best_auc.pth --epochs 20

    # Resume training
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

from model_core import Two_Stream_Net, Three_Stream_Net
from loss.cb_consistency import CombinedThreeStreamLoss

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
DEVICE      = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Three-Stream specific config
CB_LOSS_WEIGHT   = 0.3      # Weight for CB consistency loss
PATCHES_PER_REGION = 16     # 4x4 grid per region
NUM_REGIONS = 5             # Number of ID card regions

# Phase 2: CB stream training (frozen backbone)
PHASE2_LR = 1e-3
PHASE2_EPOCHS = 15

# Phase 3: Joint fine-tuning (differential LR)
PHASE3_BACKBONE_LR = 1e-5   # Slow LR for pretrained backbone
PHASE3_CB_LR = 1e-4         # Faster LR for CB/fusion
PHASE3_EPOCHS = 30
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


def get_transforms(use_three_stream=False):
    """
    Get train and val transforms.
    For three-stream, disable ColorJitter to preserve CB channel information.
    """
    if use_three_stream:
        # No ColorJitter for three-stream (preserves CB channel)
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # No ColorJitter!
            transforms.ToTensor(),                          # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],          # -> [-1,1]
                                 [0.5, 0.5, 0.5]),
        ])
    else:
        # Standard transforms with ColorJitter for two-stream
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return train_tf, val_tf


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, ckpt_path, model_type='two_stream'):
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
        'model_type': model_type,
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


def train(
    resume_path=None,
    use_three_stream=False,
    freeze_backbone=False,
    init_from_two_stream=None,
    num_epochs=None,
):
    """
    Main training function.

    Args:
        resume_path: Path to resume training from (full checkpoint)
        use_three_stream: If True, use Three_Stream_Net instead of Two_Stream_Net
        freeze_backbone: If True, freeze RGB-SRM backbone (Phase 1 training)
        init_from_two_stream: Path to Two_Stream_Net checkpoint to initialize from
        num_epochs: Number of epochs (auto-set based on phase if None)
    """
    # Determine number of epochs based on training phase
    if num_epochs is None:
        if use_three_stream and freeze_backbone:
            num_epochs = PHASE2_EPOCHS  # 15 epochs for Phase 2 (CB training)
        elif use_three_stream:
            num_epochs = PHASE3_EPOCHS  # 30 epochs for Phase 3 (fine-tuning)
        else:
            num_epochs = NUM_EPOCHS     # 30 epochs for Phase 1 (Two-Stream)

    # ── Debug: Print training configuration ──
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Model type:       {'Three-Stream' if use_three_stream else 'Two-Stream'}")
    if use_three_stream:
        if freeze_backbone:
            print(f"  Training phase:   Phase 2 (CB stream training, backbone frozen)")
        else:
            print(f"  Training phase:   Phase 3 (Joint fine-tuning)")
    else:
        print(f"  Training phase:   Phase 1 (Two-Stream training)")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Device:           {DEVICE}")
    print(f"  Resume from:      {resume_path if resume_path else 'None'}")
    print(f"  Init from:        {init_from_two_stream if init_from_two_stream else 'None'}")
    print("=" * 60)

    # Build stratified in-memory split from source directory
    train_samples, val_samples = make_splits(SRC_DIR, VAL_SPLIT, SEED)

    # Get transforms
    train_tf, val_tf = get_transforms(use_three_stream=use_three_stream)

    # Datasets & loaders
    train_ds = ICCardDataset(train_samples, split_name='train', transform=train_tf)
    val_ds   = ICCardDataset(val_samples,   split_name='val',   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    if use_three_stream:
        model_type = 'three_stream'
        if init_from_two_stream and os.path.exists(init_from_two_stream):
            print(f"Initializing Three_Stream_Net from Two_Stream_Net: {init_from_two_stream}")
            model = Three_Stream_Net.from_two_stream(init_from_two_stream)
        else:
            model = Three_Stream_Net()

        if freeze_backbone:
            model.freeze_backbone(freeze=True)
            print("Phase 2 training: RGB-SRM backbone frozen, training CB stream + fusion")

        ckpt_prefix = 'three_stream_'
    else:
        model_type = 'two_stream'
        model = Two_Stream_Net()
        ckpt_prefix = ''

    model = model.to(DEVICE)
    print(f"\n[DEBUG] Model loaded: {model_type}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[DEBUG] Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Detailed parameter breakdown for three-stream
    if use_three_stream:
        cb_params = sum(p.numel() for n, p in model.named_parameters() if 'cb_stream' in n)
        fusion_params = sum(p.numel() for n, p in model.named_parameters() if 'three_stream_fusion' in n)
        backbone_params_count = total_params - cb_params - fusion_params
        cb_trainable = sum(p.numel() for n, p in model.named_parameters() if 'cb_stream' in n and p.requires_grad)
        fusion_trainable = sum(p.numel() for n, p in model.named_parameters() if 'three_stream_fusion' in n and p.requires_grad)
        print(f"[DEBUG] Parameter breakdown:")
        print(f"        - Backbone (RGB+SRM): {backbone_params_count:,} params")
        print(f"        - CB stream:          {cb_params:,} params ({cb_trainable:,} trainable)")
        print(f"        - Three-stream fusion:{fusion_params:,} params ({fusion_trainable:,} trainable)")

    # Loss function
    if use_three_stream:
        criterion = CombinedThreeStreamLoss(
            cls_weight=1.0,
            cb_weight=CB_LOSS_WEIGHT,
            patches_per_region=PATCHES_PER_REGION,
            use_region_level=True,
        )
        print(f"Using CombinedThreeStreamLoss (cb_weight={CB_LOSS_WEIGHT})")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if use_three_stream and freeze_backbone:
        # Phase 2: Train only CB stream and fusion (backbone frozen)
        lr = PHASE2_LR
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4
        )
    elif use_three_stream and not freeze_backbone:
        # Phase 3: Joint fine-tuning with differential learning rates
        backbone_params = []
        cb_fusion_params = []
        for name, param in model.named_parameters():
            if 'cb_stream' in name or 'three_stream_fusion' in name:
                cb_fusion_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': PHASE3_BACKBONE_LR},
            {'params': cb_fusion_params, 'lr': PHASE3_CB_LR},
        ], weight_decay=1e-4)
        lr = PHASE3_CB_LR  # For logging
        print(f"Differential LR: backbone={PHASE3_BACKBONE_LR}, cb/fusion={PHASE3_CB_LR}")
    else:
        # Phase 1: Two-Stream training
        lr = LR
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    print(f"[DEBUG] Optimizer: Adam (lr={lr}, weight_decay=1e-4)")
    print(f"[DEBUG] Scheduler: StepLR (step_size=10, gamma=0.5)")

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
    metrics_csv_path = METRICS_CSV.replace('.csv', f'_{model_type}.csv')

    print(f"\n[DEBUG] Starting training from epoch {start_epoch} to {num_epochs}")
    print(f"[DEBUG] Checkpoints will be saved to: {CKPT_DIR}/")
    print(f"[DEBUG] Metrics CSV: {metrics_csv_path}")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        total_cls_loss, total_cb_loss = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
        first_batch = True
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Debug: Print shapes on first batch of first epoch
            if first_batch and epoch == start_epoch:
                print(f"\n[DEBUG] First batch shapes:")
                print(f"        - Input images: {imgs.shape}")
                print(f"        - Labels: {labels.shape} (genuine={sum(labels==0).item()}, tamper={sum(labels==1).item()})")
                first_batch = False

            optimizer.zero_grad()

            if use_three_stream:
                logits, feats, att, cb_features = model(imgs, return_cb_features=True)

                # Debug: Print CB features shape on first batch
                if epoch == start_epoch and total == 0:
                    print(f"        - CB features: {cb_features.shape}")
                    print(f"        - Logits: {logits.shape}")

                loss_dict = criterion(logits, cb_features, labels)
                loss = loss_dict['total']
                total_cls_loss += loss_dict['classification'].item() * imgs.size(0)
                total_cb_loss += loss_dict['cb_consistency'].item() * imgs.size(0)
            else:
                logits, feats, att = model(imgs)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ───────────────────────────────────────────
        model.eval()
        v_correct, v_total = 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]  "):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                if use_three_stream:
                    logits, _, _ = model(imgs, return_cb_features=False)
                else:
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

        if use_three_stream:
            avg_cls_loss = total_cls_loss / total
            avg_cb_loss = total_cb_loss / total
            print(f"          | cls_loss={avg_cls_loss:.4f} | cb_loss={avg_cb_loss:.4f}")

        print(f"          | FAR={far*100:.2f}% | FRR={frr*100:.2f}% | F1={val_f1:.4f} | AUC={val_auc:.4f}")
        print(f"          | TP={tp} TN={tn} FP={fp} FN={fn}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"          | LR={current_lr:.2e}")

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
            ckpt_path = os.path.join(CKPT_DIR, f'{ckpt_prefix}best_acc.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path, model_type)
            print(f"  -> Saved best ACC model (val_acc={val_acc:.4f}) -> {ckpt_path}")

        # Save best AUC checkpoint
        if val_auc >= best_metrics['best_auc']:
            best_metrics['best_auc'] = val_auc
            current_metrics['best_auc'] = val_auc
            ckpt_path = os.path.join(CKPT_DIR, f'{ckpt_prefix}best_auc.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path, model_type)
            print(f"  -> Saved best AUC model (val_auc={val_auc:.4f}) -> {ckpt_path}")

        # Save best F1 checkpoint
        if val_f1 >= best_metrics['best_f1']:
            best_metrics['best_f1'] = val_f1
            current_metrics['best_f1'] = val_f1
            ckpt_path = os.path.join(CKPT_DIR, f'{ckpt_prefix}best_f1.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path, model_type)
            print(f"  -> Saved best F1 model (val_f1={val_f1:.4f}) -> {ckpt_path}")

        # Save latest checkpoint (for resuming interrupted training)
        latest_path = os.path.join(CKPT_DIR, f'{ckpt_prefix}latest.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, latest_path, model_type)

        # Log metrics every LOG_INTERVAL epochs
        if epoch % LOG_INTERVAL == 0:
            metrics_row = {
                'epoch': epoch,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 4),
                'val_acc': round(val_acc, 4),
                'FAR': round(far, 5),
                'FRR': round(frr, 5),
                'val_f1': round(val_f1, 4),
                'val_auc': round(val_auc, 4),
            }
            if use_three_stream:
                metrics_row['cls_loss'] = round(total_cls_loss / total, 4)
                metrics_row['cb_loss'] = round(total_cb_loss / total, 4)

            metrics_log.append(metrics_row)
            # Save CSV after each log interval
            metrics_df = pd.DataFrame(metrics_log)
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"  -> Metrics saved to {metrics_csv_path}")

    print(f"\nTraining complete.")
    print(f"  Best ACC: {best_metrics['best_acc']:.4f}")
    print(f"  Best AUC: {best_metrics['best_auc']:.4f}")
    print(f"  Best F1:  {best_metrics['best_f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IC Card Tamper Detection')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/latest.pth)')
    parser.add_argument('--use-three-stream', action='store_true',
                        help='Use Three-Stream Network (RGB + SRM + CB)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze RGB-SRM backbone (Phase 1 training for three-stream)')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Initialize Three_Stream_Net from Two_Stream_Net checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: 15 for Phase 1, 30 for Phase 2/Two-Stream)')
    args = parser.parse_args()

    train(
        resume_path=args.resume,
        use_three_stream=args.use_three_stream,
        freeze_backbone=args.freeze_backbone,
        init_from_two_stream=args.init_from,
        num_epochs=args.epochs,
    )
