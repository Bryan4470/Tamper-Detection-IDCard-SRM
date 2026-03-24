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
CKPT_PATH        = '../checkpoints/best_f1.pth'
TEST_DIR         = r'C:\Users\bryancfk\extracted_images_test'
IMAGE_SIZE       = 256
TAMPER_THRESHOLD = 0.2   # adjust after threshold tuning
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES          = {0: 'genuine', 1: 'tamper'}
# ────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


def evaluate_test_split(model):
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
