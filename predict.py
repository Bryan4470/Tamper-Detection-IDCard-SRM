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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model_core import Two_Stream_Net

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_PATH        = '../checkpoints/best_model.pth'
TEST_DIR         = r'C:\Users\bryancfk\extracted_images_test'
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


def load_model():
    model = Two_Stream_Net().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
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
    from sklearn.metrics import classification_report, confusion_matrix

    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_preds, all_labels = [], []
    counts = {'genuine': 0, 'tamper': 0}

    for cls_name, label in [('genuine', 0), ('tamper', 1)]:
        cls_dir = os.path.join(TEST_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir)
                 if os.path.splitext(f)[1].lower() in EXTS]
        counts[cls_name] = len(files)
        for fname in files:
            result = predict_image(model, os.path.join(cls_dir, fname))
            pred_label = 1 if result['prediction'] == 'tamper' else 0
            all_preds.append(pred_label)
            all_labels.append(label)

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

    far = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # tampered accepted as genuine
    frr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # genuine rejected as tampered
    print(f"\nFAR (tamper accepted as genuine): {far*100:.2f}%")
    print(f"FRR (genuine rejected as tamper): {frr*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='IC Card Tamper Detection')
    parser.add_argument('--image',  type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--eval',   action='store_true',
                        help='Evaluate on data/test/ split with metrics')
    args = parser.parse_args()

    model = load_model()

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
