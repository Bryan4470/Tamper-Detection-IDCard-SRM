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
IMAGE_SIZE  = 256
NUM_CLASSES = 2                  # genuine=0, tampered=1
VAL_SPLIT   = 0.15               # fraction of SRC_DIR images used for validation
SEED        = 42
DEVICE      = 'cuda:1' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 6             # dataloader workers (0 = main process only)

# Backbone-specific training configurations
BACKBONE_CONFIGS = {
    'xception': {
        'batch_size': 18,
        'num_epochs': 30,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'step',
        'scheduler_step': 10,
        'scheduler_gamma': 0.5,
        'warmup_epochs': 0,
    },
    'convnext': {
        'batch_size': 20,       # Can increase due to lower memory usage
        'num_epochs': 30,
        'lr': 5e-5,             # Lower LR recommended for ConvNeXt
        'weight_decay': 0.05,   # Higher weight decay recommended
        'optimizer': 'adamw',   # AdamW recommended for ConvNeXt
        'scheduler': 'cosine',  # Cosine annealing works well
        'warmup_epochs': 5,     # Warmup recommended
    },
}
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


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, ckpt_path, backbone='xception'):
    """Save checkpoint with all training state."""
    torch.save({
        'epoch': epoch,
        'backbone': backbone,
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


def init_eval_csv(csv_path, backbone):
    """Initialize CSV file for evaluation results."""
    with open(csv_path, 'w') as f:
        f.write('epoch,backbone,train_loss,train_acc,val_acc,far,frr,f1,auc,lr\n')
    print(f"Initialized eval CSV: {csv_path}")


def append_eval_csv(csv_path, epoch, backbone, train_loss, train_acc, val_acc, far, frr, f1, auc, lr):
    """Append evaluation results to CSV file."""
    with open(csv_path, 'a') as f:
        f.write(f'{epoch},{backbone},{train_loss:.6f},{train_acc:.6f},{val_acc:.6f},'
                f'{far:.6f},{frr:.6f},{f1:.6f},{auc:.6f},{lr:.2e}\n')
    print(f"  -> Saved eval results to {csv_path}")


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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(resume_path=None, backbone='xception'):
    # Get backbone-specific configuration
    config = BACKBONE_CONFIGS[backbone]
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    warmup_epochs = config['warmup_epochs']

    print(f"\n{'='*60}")
    print(f"Training with backbone: {backbone}")
    print(f"Batch size: {batch_size}, LR: {lr}, Weight decay: {weight_decay}")
    print(f"Optimizer: {config['optimizer']}, Warmup epochs: {warmup_epochs}")
    print(f"{'='*60}\n")

    # Build stratified in-memory split from source directory
    train_samples, val_samples = make_splits(SRC_DIR, VAL_SPLIT, SEED)

    # Datasets & loaders
    train_ds = ICCardDataset(train_samples, split_name='train', transform=train_tf)
    val_ds   = ICCardDataset(val_samples,   split_name='val',   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = Two_Stream_Net(backbone=backbone).to(DEVICE)
    print(f"Model loaded. Running on: {DEVICE}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if config.get('scheduler') == 'cosine':
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        step_scheduler_per_batch = True
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('scheduler_step', 10),
            gamma=config.get('scheduler_gamma', 0.5)
        )
        step_scheduler_per_batch = False

    # Initialize best metrics
    start_epoch = 1
    best_metrics = {'best_acc': 0.0, 'best_auc': 0.0, 'best_f1': 0.0}

    # Resume from checkpoint if specified
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_metrics = load_checkpoint(resume_path, model, optimizer, scheduler)
    elif resume_path:
        print(f"[WARN] Checkpoint not found: {resume_path}, starting from scratch")

    # Initialize eval CSV (append mode if resuming, else create new)
    eval_csv_path = os.path.join(CKPT_DIR, f'eval_results_{backbone}.csv')
    if start_epoch == 1 or not os.path.exists(eval_csv_path):
        init_eval_csv(eval_csv_path, backbone)
    else:
        print(f"Appending to existing eval CSV: {eval_csv_path}")

    for epoch in range(start_epoch, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, feats, att = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Step scheduler per batch for cosine schedule
            if step_scheduler_per_batch:
                scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

            # Update progress bar with current LR
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'lr': f'{current_lr:.2e}'})

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ───────────────────────────────────────────
        model.eval()
        v_correct, v_total = 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]  "):
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

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | lr={current_lr:.2e} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        print(f"          | FAR={far*100:.2f}% | FRR={frr*100:.2f}% | F1={val_f1:.4f} | AUC={val_auc:.4f}")

        # Save eval results to CSV every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            append_eval_csv(eval_csv_path, epoch, backbone, train_loss, train_acc,
                           val_acc, far, frr, val_f1, val_auc, current_lr)

        # Step scheduler per epoch for step scheduler
        if not step_scheduler_per_batch:
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
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path, backbone)
            print(f"  -> Saved best ACC model (val_acc={val_acc:.4f}) -> {ckpt_path}")

        # Save best AUC checkpoint
        if val_auc >= best_metrics['best_auc']:
            best_metrics['best_auc'] = val_auc
            current_metrics['best_auc'] = val_auc
            ckpt_path = os.path.join(CKPT_DIR, 'best_auc.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path, backbone)
            print(f"  -> Saved best AUC model (val_auc={val_auc:.4f}) -> {ckpt_path}")

        # Save best F1 checkpoint
        if val_f1 >= best_metrics['best_f1']:
            best_metrics['best_f1'] = val_f1
            current_metrics['best_f1'] = val_f1
            ckpt_path = os.path.join(CKPT_DIR, 'best_f1.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, ckpt_path, backbone)
            print(f"  -> Saved best F1 model (val_f1={val_f1:.4f}) -> {ckpt_path}")

        # Save latest checkpoint (for resuming interrupted training)
        latest_path = os.path.join(CKPT_DIR, 'latest.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, current_metrics, latest_path, backbone)

    print(f"\nTraining complete.")
    print(f"  Best ACC: {best_metrics['best_acc']:.4f}")
    print(f"  Best AUC: {best_metrics['best_auc']:.4f}")
    print(f"  Best F1:  {best_metrics['best_f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IC Card Tamper Detection')
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['xception', 'convnext'],
                        help='Backbone architecture to use (default: xception)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/latest.pth)')
    args = parser.parse_args()

    train(resume_path=args.resume, backbone=args.backbone)
