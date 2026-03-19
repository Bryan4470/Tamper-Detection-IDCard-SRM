#!/usr/bin/env python3
"""
Training Script for RGB+Cb Dual-Stream Tamper Detection Model

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume checkpoints/last.pth
    python scripts/train.py --evaluate --checkpoint checkpoints/best.pth --csv test.csv --output results/
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import argparse
import logging
import random
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

from src.data import EKYCDataLoader, load_test_dataset_from_csv
from src.models import get_model, CombinedLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    logger.info(f"Random seed: {seed}")


class Trainer:
    """Trainer for RGB+Cb dual-stream model."""

    def __init__(self, config: Dict[str, Any], resume_path: Optional[str] = None):
        self.config = config

        # Set random seed
        seed_cfg = config.get('random_seed', {})
        if seed_cfg.get('enabled', True):
            set_random_seed(seed_cfg.get('seed', 42), seed_cfg.get('deterministic', False))

        self.device = self._setup_device()

        # Performance tracking
        self.best_val_f1 = 0.0
        self.best_val_auc = 0.0
        self.best_frr_under_far = float('inf')

        # Thresholds
        eval_cfg = config.get('evaluation', {})
        self.threshold = eval_cfg.get('classification_threshold', 0.5)
        self.far_threshold = eval_cfg.get('far_constraint_threshold', 0.01)

        # Early stopping
        self.patience_counter = 0
        self.start_epoch = 0

        # Directories
        if resume_path:
            self.save_dir = Path(resume_path).parent
            self.log_dir = Path(config['logging']['log_dir'])
            timestamp = self.save_dir.name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_dir = Path(config['logging']['log_dir'])
            self.save_dir = Path(config['logging']['save_dir']) / timestamp
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir / f"{config['logging']['experiment_name']}_{timestamp}")

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()

        self.scaler = torch.cuda.amp.GradScaler() if config['device'].get('mixed_precision', False) else None

        if resume_path:
            self._load_checkpoint(resume_path)

        logger.info(f"Save: {self.save_dir}")
        logger.info(f"Cb patches: {self.model.get_num_patches()}")

    def _setup_device(self) -> torch.device:
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['gpu_id']}")
            logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device('cpu')
            logger.info("CPU mode")
        return device

    def _setup_data(self):
        logger.info("Loading data...")
        data_cfg = self.config['data']
        aug_cfg = self.config.get('augmentation', {})
        loader = EKYCDataLoader(
            root_dir=data_cfg['root_dir'],
            image_size=data_cfg['image_size'],
            batch_size=data_cfg['batch_size'],
            val_split=data_cfg['val_split'],
            test_split=data_cfg['test_split'],
            random_state=data_cfg['random_state'],
            num_workers=data_cfg['num_workers'],
            aug_config=aug_cfg
        )

        self.train_loader, self.val_loader, _ = loader.get_dataloaders()
        logger.info(f"Train: {len(self.train_loader)} batches, Val: {len(self.val_loader)} batches")

        # Test datasets
        test_cfg = data_cfg.get('test_dataset', {})
        if test_cfg.get('enabled', False):
            csv_dir = test_cfg['csv_dir']
            csv_files = test_cfg.get('csv_files', [])
            # If csv_files is empty or not specified, load all CSV files in csv_dir
            if not csv_files:
                csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
                logger.info(f"Auto-detected {len(csv_files)} CSV files in {csv_dir}")
            csv_paths = [os.path.join(csv_dir, f) for f in csv_files]
            self.test_loader = load_test_dataset_from_csv(
                csv_paths=csv_paths,
                batch_size=data_cfg['batch_size'],
                image_size=data_cfg['image_size'],
                num_workers=data_cfg['num_workers'],
                return_individual=False
            )
        else:
            self.test_loader = None

    def _setup_model(self):
        logger.info("Building model...")
        self.model = get_model(self.config).to(self.device)
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parameters: {params:,}")

    def _setup_optimizer(self):
        cfg = self.config['training']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg['epochs'], eta_min=cfg['scheduler']['min_lr']
        )
        self.grad_clip = cfg.get('grad_clip_max_norm', 1.0)

    def _setup_loss(self):
        cfg = self.config['training']['loss']
        self.criterion = CombinedLoss(
            cls_weight=cfg.get('classification_weight', 1.0),
            cb_weight=cfg.get('cb_consistency_weight', 0.3),
            cb_margin=cfg.get('cb_margin', 0.5),
            use_focal_loss=cfg.get('use_focal', False),
            patches_per_region=cfg.get('patches_per_region', 16),
            use_region_level=cfg.get('use_region_level', True),
            contrastive_weight=cfg.get('contrastive_weight', 0.0),
            temperature=cfg.get('temperature', 0.07),
        )

    def _load_checkpoint(self, path: str):
        logger.info(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_f1 = ckpt.get('best_val_f1', 0.0)
        self.best_val_auc = ckpt.get('best_val_auc', 0.0)
        self.patience_counter = ckpt.get('patience_counter', 0)
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = total_cls = total_cb = total_con = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self.model(images, return_features=True)
                loss_dict = self.criterion(
                    out['logits'], out['cb_features'], labels,
                    fused_features=out.get('fused_features')
                )
                loss = loss_dict['total']

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            total_cls += loss_dict['classification'].item()
            total_cb += loss_dict['cb_consistency'].item()
            total_con += loss_dict['contrastive'].item()
            all_preds.extend(out['logits'].argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        n = len(self.train_loader)
        return {'loss': total_loss/n, 'cls_loss': total_cls/n, 'cb_loss': total_cb/n,
                'con_loss': total_con/n, 'accuracy': accuracy_score(all_labels, all_preds)}

    def validate(self, loader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = total_cls = total_cb = total_con = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.model(images, return_features=True)
                loss_dict = self.criterion(
                    out['logits'], out['cb_features'], labels,
                    fused_features=out.get('fused_features')
                )

                total_loss += loss_dict['total'].item()
                total_cls += loss_dict['classification'].item()
                total_cb += loss_dict['cb_consistency'].item()
                total_con += loss_dict['contrastive'].item()
                probs = torch.softmax(out['logits'], dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        labels_np, probs_np = np.array(all_labels), np.array(all_probs)
        preds = (probs_np >= self.threshold).astype(int)

        cm = confusion_matrix(labels_np, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        n = len(loader)
        return {
            'loss': total_loss/n, 'cls_loss': total_cls/n, 'cb_loss': total_cb/n,
            'con_loss': total_con/n,
            'accuracy': accuracy_score(labels_np, preds),
            'precision': precision_score(labels_np, preds, zero_division=0),
            'recall': recall_score(labels_np, preds, zero_division=0),
            'f1_score': f1_score(labels_np, preds, zero_division=0),
            'auc_roc': roc_auc_score(labels_np, probs_np) if len(np.unique(labels_np)) > 1 else 0.0,
            'frr': fn/(fn+tp) if (fn+tp) > 0 else 0.0,
            'far': fp/(fp+tn) if (fp+tn) > 0 else 0.0,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }

    def save_checkpoint(self, epoch: int, metrics: Dict, filename: str):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_f1': self.best_val_f1,
            'best_val_auc': self.best_val_auc,
            'patience_counter': self.patience_counter,
            **{f'val_{k}': v for k, v in metrics.items()}
        }
        if self.scaler:
            ckpt['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(ckpt, self.save_dir / filename)

    def train(self):
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping']['patience']

        logger.info("\n" + "="*70)
        logger.info("Starting training...")
        logger.info("="*70)

        for epoch in range(self.start_epoch, epochs):
            train_m = self.train_epoch(epoch)
            val_m = self.validate(self.val_loader)
            self.scheduler.step()

            # Early stopping
            if val_m['f1_score'] > self.best_val_f1:
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Logging
            self.writer.add_scalar('Train/Loss', train_m['loss'], epoch)
            self.writer.add_scalar('Val/F1', val_m['f1_score'], epoch)
            self.writer.add_scalar('Val/AUC', val_m['auc_roc'], epoch)

            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info(f"Train - Loss: {train_m['loss']:.4f}, Cls: {train_m['cls_loss']:.4f}, Cb: {train_m['cb_loss']:.4f}, Con: {train_m['con_loss']:.4f}")
            logger.info(f"Val   - F1: {val_m['f1_score']:.4f}, AUC: {val_m['auc_roc']:.4f}, FRR: {val_m['frr']:.4f}, FAR: {val_m['far']:.4f}")
            logger.info(f"Patience: {self.patience_counter}/{patience}")

            # Save checkpoints
            if val_m['f1_score'] > self.best_val_f1:
                self.best_val_f1 = val_m['f1_score']
                self.save_checkpoint(epoch, val_m, 'best_f1.pth')
                logger.info("Saved best_f1.pth")

            if val_m['auc_roc'] > self.best_val_auc:
                self.best_val_auc = val_m['auc_roc']
                self.save_checkpoint(epoch, val_m, 'best_auc.pth')
                logger.info("Saved best_auc.pth")

            if val_m['far'] <= self.far_threshold and val_m['frr'] < self.best_frr_under_far:
                self.best_frr_under_far = val_m['frr']
                self.save_checkpoint(epoch, val_m, 'best_frr_under_far.pth')
                logger.info("Saved best_frr_under_far.pth")

            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.save_checkpoint(epoch, val_m, 'last_epoch.pth')

        logger.info("\n" + "="*70)
        logger.info(f"Training complete! Best F1: {self.best_val_f1:.4f}, Best AUC: {self.best_val_auc:.4f}")
        logger.info("="*70)

        # Evaluate on test set
        if self.test_loader:
            self._evaluate_test()

        self.writer.close()

    def _evaluate_test(self):
        logger.info("\nEvaluating on test set...")
        eval_dir = self.save_dir / "eval_results"
        eval_dir.mkdir(exist_ok=True)

        for name in ['best_f1', 'best_auc', 'last_epoch']:
            path = self.save_dir / f'{name}.pth'
            if not path.exists():
                continue

            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            m = self.validate(self.test_loader)

            logger.info(f"{name}: F1={m['f1_score']:.4f}, AUC={m['auc_roc']:.4f}, FRR={m['frr']:.4f}, FAR={m['far']:.4f}")

            with open(eval_dir / f'{name}_results.txt', 'w') as f:
                f.write(f"F1: {m['f1_score']:.4f}\nAUC: {m['auc_roc']:.4f}\n")
                f.write(f"FRR: {m['frr']:.4f}\nFAR: {m['far']:.4f}\n")
                f.write(f"TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}\n")


def main():
    parser = argparse.ArgumentParser(description='Train RGB+Cb Model')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config, args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
