"""
Inference script for IC Card Tamper Detection.
Usage:
    # Single image
    python predict.py --image path/to/card.jpg

    # Whole folder
    python predict.py --folder path/to/test/images

    # Evaluate test split with metrics
    python predict.py --eval
"""

import os
import sys
import argparse
import hashlib
import logging
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model_core import Two_Stream_Net

# ── Config ──────────────────────────────────────────────────────────────────
_CKPT_PREFERRED  = '../checkpoints/best_frr_under_far.pth'
_CKPT_FALLBACK   = '../checkpoints/best_f1.pth'
CKPT_PATH        = _CKPT_PREFERRED if os.path.exists(_CKPT_PREFERRED) else _CKPT_FALLBACK
TEST_DIR         = '/mnt3/auto-ekyc/id_physical_tamper_new/data/testing_dataset'
IMAGE_SIZE       = 256
TAMPER_THRESHOLD = 0.2   # adjust after threshold tuning
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES          = {0: 'genuine', 1: 'tamper'}
# ────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_model(ckpt_path=CKPT_PATH):
    model = Two_Stream_Net().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded: {ckpt_path}  "
          f"(epoch={ckpt['epoch']}, "
          f"f1={ckpt.get('val_f1', 'n/a')}, "
          f"auc={ckpt.get('val_auc', 'n/a')})")
    return model


def predict_image(model, image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    with torch.no_grad():
        logits, feats, att_map = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_class = 1 if probs[1].item() >= TAMPER_THRESHOLD else 0
    confidence = probs[pred_class].item()
    return {
        'path':        image_path,
        'prediction':  CLASSES[pred_class],
        'confidence':  confidence,
        'prob_genuine':  probs[0].item(),
        'prob_tampered': probs[1].item(),
    }


class _EvalDataset(Dataset):
    def __init__(self, samples):   # samples: [(path, label), ...]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return transform(img), label


def _load_test_samples(test_dir, use_cache=True):
    """Load test samples from all CSVs in test_dir (image_path + fraud_type columns).
    Caches results in test_dir/.test_cache/ keyed by CSV file list hash.
    """
    csv_paths = sorted(
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir) if f.endswith('.csv')
    )

    cache_key  = hashlib.md5('|'.join(csv_paths).encode()).hexdigest()
    cache_dir  = os.path.join(test_dir, '.test_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'test_data_{cache_key}.pkl')

    if use_cache and os.path.exists(cache_file):
        try:
            print(f"Loading test dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            return cached['image_paths'], cached['labels']
        except Exception as e:
            print(f"Cache load failed ({e}), rebuilding...")

    class_to_idx = {'genuine': 0, 'tamper': 1}
    all_paths, all_labels = [], []

    print("Loading test dataset from CSVs...")
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            if 'image_path' not in df.columns or 'fraud_type' not in df.columns:
                continue
            count = 0
            for _, row in df.iterrows():
                img_path   = row['image_path']
                fraud_type = row['fraud_type'].strip().lower()
                if fraud_type not in class_to_idx:
                    continue
                if os.path.exists(img_path):
                    all_paths.append(img_path)
                    all_labels.append(class_to_idx[fraud_type])
                    count += 1
                else:
                    logging.warning(f"Not found: {img_path}")
            print(f"  {os.path.basename(csv_path)}: {count} images")
        except Exception as e:
            print(f"  [ERROR] {csv_path}: {e}")

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'image_paths': all_paths, 'labels': all_labels}, f)
        print(f"Saved test cache: {cache_file}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return all_paths, all_labels


def evaluate_test_split(model):
    all_paths, all_labels = _load_test_samples(TEST_DIR)
    samples = list(zip(all_paths, all_labels))

    counts = {
        'genuine': sum(1 for l in all_labels if l == 0),
        'tamper':  sum(1 for l in all_labels if l == 1),
    }

    loader = DataLoader(_EvalDataset(samples), batch_size=32,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits, _, _ = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    labels_np = np.array(all_labels)
    probs_np  = np.array(all_probs)
    preds_np  = (probs_np >= TAMPER_THRESHOLD).astype(int)

    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # tamper accepted as genuine
    frr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # genuine rejected as tamper
    f1  = f1_score(labels_np, preds_np, zero_division=0)
    auc = roc_auc_score(labels_np, probs_np) if len(np.unique(labels_np)) > 1 else 0.0

    print(f"\nEvaluating on: {TEST_DIR}")
    print(f"  genuine images : {counts['genuine']}")
    print(f"  tamper  images : {counts['tamper']}")
    print(f"  total          : {counts['genuine'] + counts['tamper']}")
    print(f"  threshold      : {TAMPER_THRESHOLD}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Genuine  Tamper")
    print(f"Actual Genuine   {tn:4d}     {fp:4d}")
    print(f"Actual Tamper    {fn:4d}     {tp:4d}")
    print(f"\nFAR (tamper accepted as genuine): {far*100:.2f}%")
    print(f"FRR (genuine rejected as tamper): {frr*100:.2f}%")
    print(f"F1 Score:                         {f1:.4f}")
    print(f"AUC-ROC:                          {auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='IC Card Tamper Detection')
    parser.add_argument('--image',      type=str, help='Path to single image')
    parser.add_argument('--folder',     type=str, help='Path to folder of images')
    parser.add_argument('--eval',       action='store_true',
                        help='Evaluate on test split with metrics')
    parser.add_argument('--checkpoint', type=str, default=CKPT_PATH,
                        help='Checkpoint to load (default: best_f1.pth). '
                             'Options: best_f1.pth, best_auc.pth, best_frr_under_far.pth, last_epoch.pth')
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    if args.image:
        result = predict_image(model, args.image)
        print(f"\nImage:      {result['path']}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"  genuine:  {result['prob_genuine']*100:.1f}%")
        print(f"  tampered: {result['prob_tampered']*100:.1f}%")

    elif args.folder:
        EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = [f for f in os.listdir(args.folder)
                 if os.path.splitext(f)[1].lower() in EXTS]
        for fname in sorted(files):
            result = predict_image(model, os.path.join(args.folder, fname))
            print(f"{fname:40s}  →  {result['prediction']:8s}  "
                  f"genuine={result['prob_genuine']*100:.1f}%  "
                  f"tamper={result['prob_tampered']*100:.1f}%")

    elif args.eval:
        evaluate_test_split(model)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
