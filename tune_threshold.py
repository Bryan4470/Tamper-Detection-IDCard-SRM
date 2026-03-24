"""
Threshold tuning script for IC Card Tamper Detection.

Sweeps TAMPER_THRESHOLD from 0.05 to 0.95 and reports FAR, FRR, F1, and
accuracy at each step. Highlights the Equal Error Rate (EER) point and the
best F1 threshold. No retraining needed — just needs a trained checkpoint.

Usage:
    python tune_threshold.py
    python tune_threshold.py --checkpoint ../checkpoints/best_auc.pth
    python tune_threshold.py --step 0.01
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model_core import Two_Stream_Net

# ── Config ───────────────────────────────────────────────────────────────────
CKPT_PATH  = '../checkpoints/best_f1.pth'
TEST_DIR   = r'C:\Users\bryancfk\extracted_images_test'
IMAGE_SIZE = 256
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
# ─────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class _EvalDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return transform(img), label


def collect_probs(ckpt_path):
    model = Two_Stream_Net().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded: {ckpt_path}  "
          f"(epoch={ckpt['epoch']}, "
          f"f1={ckpt.get('val_f1', 'n/a')}, "
          f"auc={ckpt.get('val_auc', 'n/a')})")

    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    samples = []
    counts = {'genuine': 0, 'tamper': 0}

    for cls_name, label in [('genuine', 0), ('tamper', 1)]:
        cls_dir = os.path.join(TEST_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir)
                 if os.path.splitext(f)[1].lower() in EXTS]
        counts[cls_name] = len(files)
        samples.extend((os.path.join(cls_dir, f), label) for f in files)

    loader = DataLoader(_EvalDataset(samples), batch_size=32,
                        shuffle=False, num_workers=0, pin_memory=True)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits, _, _ = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    print(f"\nTest set  —  genuine: {counts['genuine']}, tamper: {counts['tamper']}, "
          f"total: {counts['genuine'] + counts['tamper']}")
    auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
    print(f"AUC-ROC (threshold-independent): {auc:.4f}\n")

    return np.array(all_labels), np.array(all_probs)


def sweep(labels, probs, step):
    thresholds = np.arange(step, 1.0, step)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        far = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # tamper missed
        frr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # genuine falsely rejected
        f1  = f1_score(labels, preds, zero_division=0)
        acc = (tp + tn) / len(labels)

        results.append(dict(t=t, far=far, frr=frr, f1=f1, acc=acc,
                            tp=tp, tn=tn, fp=fp, fn=fn))

    return results


def find_eer(results):
    best, best_diff = None, float('inf')
    for r in results:
        diff = abs(r['far'] - r['frr'])
        if diff < best_diff:
            best_diff = diff
            best = r
    return best


def find_best_f1(results):
    return max(results, key=lambda r: r['f1'])


def find_far_budget(results, max_far):
    """Highest threshold where FAR <= max_far (most conservative that still meets budget)."""
    candidates = [r for r in results if r['far'] <= max_far]
    return max(candidates, key=lambda r: r['t']) if candidates else None


def print_table(results):
    print(f"{'Threshold':>10}  {'FAR':>7}  {'FRR':>7}  {'F1':>7}  {'Acc':>7}  "
          f"{'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
    print("-" * 80)
    for r in results:
        print(f"  {r['t']:.3f}      {r['far']*100:6.2f}%  {r['frr']*100:6.2f}%  "
              f"{r['f1']:.4f}  {r['acc']*100:6.2f}%  "
              f"{r['tp']:5d}  {r['tn']:5d}  {r['fp']:5d}  {r['fn']:5d}")


def main():
    parser = argparse.ArgumentParser(description='Tune tamper detection threshold')
    parser.add_argument('--checkpoint', type=str, default=CKPT_PATH)
    parser.add_argument('--step', type=float, default=0.05,
                        help='Threshold sweep step size (default: 0.05)')
    parser.add_argument('--max-far', type=float, default=None,
                        help='Show best threshold where FAR <= this value (e.g. 0.05)')
    args = parser.parse_args()

    labels, probs = collect_probs(args.checkpoint)
    results = sweep(labels, probs, args.step)

    print_table(results)

    eer    = find_eer(results)
    best_f1 = find_best_f1(results)

    print("\n── Recommended operating points ──────────────────────────────────────────")
    print(f"  EER point     threshold={eer['t']:.3f}  "
          f"FAR={eer['far']*100:.2f}%  FRR={eer['frr']*100:.2f}%  F1={eer['f1']:.4f}")
    print(f"  Best F1       threshold={best_f1['t']:.3f}  "
          f"FAR={best_f1['far']*100:.2f}%  FRR={best_f1['frr']*100:.2f}%  F1={best_f1['f1']:.4f}")

    if args.max_far is not None:
        budgeted = find_far_budget(results, args.max_far)
        if budgeted:
            print(f"  FAR ≤ {args.max_far*100:.0f}% budget  threshold={budgeted['t']:.3f}  "
                  f"FAR={budgeted['far']*100:.2f}%  FRR={budgeted['frr']*100:.2f}%  F1={budgeted['f1']:.4f}")
        else:
            print(f"  FAR ≤ {args.max_far*100:.0f}% budget  — no threshold achieves this on the test set")

    print("\nSet TAMPER_THRESHOLD in predict.py to your chosen value.")


if __name__ == '__main__':
    main()
