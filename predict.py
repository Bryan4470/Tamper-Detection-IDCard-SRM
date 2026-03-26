"""
Inference script for IC Card Tamper Detection.

Supports both Two-Stream and Three-Stream models (auto-detected from checkpoint).

Usage:
    # Single image
    python predict.py --image path/to/card.jpg

    # Whole folder
    python predict.py --folder path/to/test/images

    # Evaluate test split with metrics
    python predict.py --eval

    # Specify checkpoint (auto-detects model type)
    python predict.py --checkpoint checkpoints/three_stream_best_auc.pth --eval
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

from model_core import Two_Stream_Net, Three_Stream_Net

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_PATH        = '../checkpoints/best_auc.pth'
TEST_DIR         = '/mnt3/auto-ekyc/id_physical_tamper_new/data/testing_dataset'
IMAGE_SIZE       = 256
TAMPER_THRESHOLD = 0.1   # adjust after threshold tuning
DEVICE           = 'cuda:1' if torch.cuda.is_available() else 'cpu'
CLASSES          = {0: 'genuine', 1: 'tamper'}
# ────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def detect_model_type(ckpt_path):
    """
    Auto-detect model type from checkpoint.

    Returns:
        'three_stream' or 'two_stream'
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Check for explicit model_type field
    if 'model_type' in ckpt:
        return ckpt['model_type']

    # Check state_dict keys for CB stream components
    state_dict = ckpt.get('model_state_dict', ckpt)
    for key in state_dict.keys():
        if 'cb_stream' in key or 'three_stream_fusion' in key:
            return 'three_stream'

    return 'two_stream'


def load_model(ckpt_path=CKPT_PATH):
    """
    Load model from checkpoint, auto-detecting model type.

    Returns:
        model: Loaded model in eval mode
        model_type: 'two_stream' or 'three_stream'
    """
    model_type = detect_model_type(ckpt_path)
    print(f"Detected model type: {model_type}")

    if model_type == 'three_stream':
        model = Three_Stream_Net().to(DEVICE)
    else:
        model = Two_Stream_Net().to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    epoch = ckpt.get('epoch', 'unknown')
    val_acc = ckpt.get('val_acc', 0.0)
    val_auc = ckpt.get('val_auc', 0.0)
    print(f"Loaded checkpoint from epoch {epoch} (val_acc={val_acc:.4f}, val_auc={val_auc:.4f})")

    return model, model_type


def predict_image(model, image_path, model_type='two_stream'):
    """
    Predict a single image.

    Args:
        model: Loaded model
        image_path: Path to image
        model_type: 'two_stream' or 'three_stream'

    Returns:
        Dictionary with prediction results
    """
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    with torch.no_grad():
        if model_type == 'three_stream':
            logits, feats, att_map = model(tensor, return_cb_features=False)
        else:
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


def evaluate_test_split(model, model_type='two_stream'):
    """
    Evaluate model on test split with full metrics.
    """
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

    class_to_idx = {'genuine': 0, 'tamper': 1}
    all_preds, all_labels, all_probs = [], [], []
    counts = {'genuine': 0, 'tamper': 0}

    csv_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith('.csv'))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(os.path.join(TEST_DIR, csv_file), dtype=str, keep_default_na=False)
            if 'image_path' not in df.columns or 'fraud_type' not in df.columns:
                print(f"[SKIP] {csv_file}: missing 'image_path' or 'fraud_type' column")
                continue
            for _, row in df.iterrows():
                img_path   = row['image_path']
                fraud_type = row['fraud_type'].strip().lower()
                if fraud_type not in class_to_idx or not os.path.exists(img_path):
                    continue
                label = class_to_idx[fraud_type]
                counts[fraud_type] += 1
                result = predict_image(model, img_path, model_type)
                all_preds.append(1 if result['prediction'] == 'tamper' else 0)
                all_labels.append(label)
                all_probs.append(result['prob_tampered'])
        except Exception as e:
            print(f"[ERROR] {csv_file}: {e}")

    print(f"\nEvaluating on: {TEST_DIR}")
    print(f"  Model type   : {model_type}")
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

    return {
        'far': far,
        'frr': frr,
        'f1': f1,
        'auc': auc,
    }


def main():
    parser = argparse.ArgumentParser(description='IC Card Tamper Detection')
    parser.add_argument('--image',  type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--eval',   action='store_true',
                        help='Evaluate on data/test/ split with metrics')
    parser.add_argument('--checkpoint', type=str, default=CKPT_PATH,
                        help='Path to model checkpoint')
    args = parser.parse_args()

    model, model_type = load_model(args.checkpoint)

    if args.image:
        result = predict_image(model, args.image, model_type)
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
            result = predict_image(model, os.path.join(args.folder, fname), model_type)
            print(f"{fname:40s}  →  {result['prediction']:8s}  "
                  f"genuine={result['prob_genuine']*100:.1f}%  "
                  f"tamper={result['prob_tampered']*100:.1f}%")

    elif args.eval:
        evaluate_test_split(model, model_type)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
