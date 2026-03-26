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
from torchvision import transforms
from PIL import Image
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'src'))
os.chdir(os.path.join(_THIS_DIR, 'src'))

from model_core import Two_Stream_Net

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_PATH        = '../checkpoints/best_acc.pth'
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


def load_model(backbone=None, ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = CKPT_PATH
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Auto-detect backbone from checkpoint if not specified
    if backbone is None:
        backbone = ckpt.get('backbone', 'xception')  # Default to xception for old checkpoints

    model = Two_Stream_Net(backbone=backbone).to(DEVICE)
    state_dict = ckpt['model_state_dict']

    # Handle lazy-initialized linear layers in DualCrossModalAttention
    # Check if checkpoint has linear layers that need to be initialized
    for cma_name in ['dual_cma0', 'dual_cma1']:
        linear1_key = f'{cma_name}.linear1.weight'
        if linear1_key in state_dict:
            # Get spatial size from the linear layer shape
            spatial_size = state_dict[linear1_key].shape[0]
            # Initialize the linear layers in the model
            cma_module = getattr(model, cma_name)
            cma_module.linear1 = torch.nn.Linear(spatial_size, spatial_size).to(DEVICE)
            cma_module.linear2 = torch.nn.Linear(spatial_size, spatial_size).to(DEVICE)
            cma_module._linear_initialized = True

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {backbone} checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
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


def evaluate_test_split(model):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

    # Debug: Check if test directory is accessible
    print(f"[DEBUG] Checking TEST_DIR: {TEST_DIR}")
    if not os.path.exists(TEST_DIR):
        print(f"[ERROR] TEST_DIR does not exist: {TEST_DIR}")
        return
    print(f"[DEBUG] TEST_DIR exists, listing files...")

    class_to_idx = {'genuine': 0, 'tamper': 1}
    all_preds, all_labels, all_probs = [], [], []
    counts = {'genuine': 0, 'tamper': 0}

    csv_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith('.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {csv_files}")

    total_processed = 0
    for csv_file in csv_files:
        try:
            print(f"[DEBUG] Reading CSV: {csv_file}")
            df = pd.read_csv(os.path.join(TEST_DIR, csv_file), dtype=str, keep_default_na=False)
            if 'image_path' not in df.columns or 'fraud_type' not in df.columns:
                print(f"[SKIP] {csv_file}: missing 'image_path' or 'fraud_type' column")
                continue
            print(f"[DEBUG] CSV has {len(df)} rows")
            for idx, row in df.iterrows():
                img_path   = row['image_path']
                fraud_type = row['fraud_type'].strip().lower()
                if fraud_type not in class_to_idx:
                    continue
                if not os.path.exists(img_path):
                    print(f"[SKIP] Image not found: {img_path}")
                    continue
                label = class_to_idx[fraud_type]
                counts[fraud_type] += 1
                total_processed += 1
                if total_processed % 50 == 0:
                    print(f"[DEBUG] Processed {total_processed} images...")
                result = predict_image(model, img_path)
                all_preds.append(1 if result['prediction'] == 'tamper' else 0)
                all_labels.append(label)
                all_probs.append(result['prob_tampered'])
        except Exception as e:
            print(f"[ERROR] {csv_file}: {e}")

    print(f"[DEBUG] Total images processed: {total_processed}")

    print(f"\nEvaluating on: {TEST_DIR}")
    print(f"  genuine images : {counts['genuine']}")
    print(f"  tamper  images : {counts['tamper']}")
    print(f"  total          : {counts['genuine'] + counts['tamper']}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['genuine', 'tamper']))
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"                 Predicted")
    print(f"                 Genuine  Tamper")
    print(f"Actual Genuine   {tn:4d}     {fp:4d}")
    print(f"Actual Tamper    {fn:4d}     {tp:4d}")

    # Compute metrics
    far = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # tampered accepted as genuine
    frr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # genuine rejected as tampered
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # only one class present

    print(f"\n{'='*40}")
    print(f"FAR (tamper accepted as genuine): {far*100:.2f}%")
    print(f"FRR (genuine rejected as tamper): {frr*100:.2f}%")
    print(f"F1 Score:                         {f1:.4f}")
    print(f"AUC:                              {auc:.4f}")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description='IC Card Tamper Detection')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['xception', 'convnext'],
                        help='Backbone architecture (auto-detected from checkpoint if not specified)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: checkpoints/best_acc.pth)')
    parser.add_argument('--image',  type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--eval',   action='store_true',
                        help='Evaluate on data/test/ split with metrics')
    args = parser.parse_args()

    model = load_model(backbone=args.backbone, ckpt_path=args.checkpoint)

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
