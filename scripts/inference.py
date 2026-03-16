#!/usr/bin/env python3
"""
Inference Script for RGB+Cb Tamper Detection Model

Usage:
    python scripts/inference.py --model checkpoint.pth --image path/to/image.jpg
    python scripts/inference.py --model checkpoint.pth --directory path/to/images/
    python scripts/inference.py --model checkpoint.pth --csv path/to/data.csv --output results.csv
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import shutil
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import requests
import base64
from io import BytesIO
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from src.models import get_model


def call_crop_api(img_bytes: bytes) -> str:
    """Call AI engine API to crop ID card from image."""
    url = "http://10.1.1.49:8100/mq_beta/mykad_front"
    files = {"image": img_bytes}
    params = {"is_image": "0", "returnDetailed": "1"}
    response = requests.post(url, files=files, data=params, timeout=10)
    result = response.json()
    return result.get('card_cropped')


def base64_to_pil(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data)).convert('RGB')


class TamperDetector:
    """Inference wrapper for tamper detection model."""

    def __init__(self, model_path: str, config_path: Optional[str] = None,
                 device: str = None, threshold: float = 0.5, crop: bool = False):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = threshold
        self.crop = crop

        # Load config
        if config_path is None:
            config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load model
        print(f"Loading model: {model_path}")
        self.model = get_model(self.config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model = self.model.to(self.device).eval()
        self.image_size = self.config['data']['image_size']
        print(f"Device: {self.device}")

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image to tensor."""
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor.unsqueeze(0)

    def load_image(self, image_path: str) -> Image.Image:
        """Load image, optionally cropping via API."""
        if self.crop:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            cropped_base64 = call_crop_api(img_bytes)
            if not cropped_base64:
                raise Exception("Failed to get cropped image from API")
            return base64_to_pil(cropped_base64)
        else:
            return Image.open(image_path).convert('RGB')

    def preprocess(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image from path."""
        img = self.load_image(image_path)
        return self.preprocess_image(img)

    def predict(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            return {'error': f'File not found: {image_path}'}

        try:
            img_tensor = self.preprocess(image_path).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probs = F.softmax(output['logits'], dim=1)

            prob_genuine = probs[0, 0].item()
            prob_tampered = probs[0, 1].item()

            return {
                'image_path': str(image_path),
                'prediction': 'genuine' if prob_genuine >= self.threshold else 'tampered',
                'confidence': prob_tampered,
                'prob_genuine': prob_genuine,
                'prob_tampered': prob_tampered
            }
        except Exception as e:
            return {'error': str(e), 'image_path': str(image_path)}

    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        results = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Inference"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images, valid_paths = [], []

            for path in batch_paths:
                if not os.path.exists(path):
                    results.append({'error': 'File not found', 'image_path': path})
                    continue
                try:
                    batch_images.append(self.preprocess(path))
                    valid_paths.append(path)
                except Exception as e:
                    results.append({'error': str(e), 'image_path': path})

            if not batch_images:
                continue

            batch_tensor = torch.cat(batch_images, dim=0).to(self.device)

            with torch.no_grad():
                output = self.model(batch_tensor)
                probs = F.softmax(output['logits'], dim=1)

            for j, path in enumerate(valid_paths):
                prob_genuine = probs[j, 0].item()
                prob_tampered = probs[j, 1].item()
                results.append({
                    'image_path': str(path),
                    'prediction': 'genuine' if prob_genuine >= self.threshold else 'tampered',
                    'confidence': prob_tampered,
                    'prob_genuine': prob_genuine,
                    'prob_tampered': prob_tampered
                })

        return results

    def predict_directory(self, directory: str, batch_size: int = 32) -> List[Dict]:
        extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        image_paths = [str(p) for p in Path(directory).rglob('*') if p.suffix in extensions]
        print(f"Found {len(image_paths)} images")
        return self.predict_batch(image_paths, batch_size)


def save_error_cases(results: List[Dict], labels: List[int], output_path: str, threshold: float = 0.5):
    """Copy false positive and false negative images into subfolders next to the output CSV."""
    probs = np.array([r['prob_tampered'] for r in results])
    preds = (probs >= threshold).astype(int)

    base_dir = Path(output_path).parent / "error_cases"
    fp_dir = base_dir / "false_positives"  # genuine predicted as tampered
    fn_dir = base_dir / "false_negatives"  # tampered predicted as genuine
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    for r, label, pred, prob in zip(results, labels, preds, probs):
        src = r['image_path']
        if not os.path.exists(src):
            continue
        fname = Path(src).name
        # False positive: genuine (0) predicted as tampered (1)
        if label == 0 and pred == 1:
            stem, ext = os.path.splitext(fname)
            dst = fp_dir / f"{stem}_conf{prob:.3f}{ext}"
            shutil.copy2(src, dst)
        # False negative: tampered (1) predicted as genuine (0)
        elif label == 1 and pred == 0:
            stem, ext = os.path.splitext(fname)
            dst = fn_dir / f"{stem}_conf{prob:.3f}{ext}"
            shutil.copy2(src, dst)

    fp_count = len(list(fp_dir.iterdir()))
    fn_count = len(list(fn_dir.iterdir()))
    print(f"Error cases saved to {base_dir}/")
    print(f"  false_positives/: {fp_count} images (genuine predicted as tampered)")
    print(f"  false_negatives/: {fn_count} images (tampered predicted as genuine)")


def compute_and_save_metrics(results: List[Dict], labels: List[int], output_path: str, threshold: float = 0.5):
    """Compute evaluation metrics from results with ground truth labels and save to file."""
    probs = np.array([r['prob_tampered'] for r in results])
    preds = (probs >= threshold).astype(int)
    labels_np = np.array(labels)

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(labels_np, preds),
        'precision': precision_score(labels_np, preds, zero_division=0),
        'recall': recall_score(labels_np, preds, zero_division=0),
        'f1_score': f1_score(labels_np, preds, zero_division=0),
        'auc_roc': roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0,
        'frr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        'far': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

    print("\n" + "=" * 50)
    print("Evaluation Metrics")
    print("=" * 50)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1_score']:.4f}")
    print(f"AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"FRR      : {metrics['frr']:.4f}")
    print(f"FAR      : {metrics['far']:.4f}")
    print(f"TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print("=" * 50)

    eval_path = str(output_path).replace('.csv', '_eval_results.txt')
    with open(eval_path, 'w') as f:
        f.write(f"F1: {metrics['f1_score']:.4f}\nAUC: {metrics['auc_roc']:.4f}\n")
        f.write(f"FRR: {metrics['frr']:.4f}\nFAR: {metrics['far']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\nRecall: {metrics['recall']:.4f}\n")
        f.write(f"TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n")
    print(f"Eval results saved to {eval_path}")

    save_error_cases(results, labels, output_path, threshold)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Tamper Detection Inference')
    parser.add_argument('--model', required=True, help='Path to checkpoint')
    parser.add_argument('--config', default=None, help='Path to config.yaml')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--directory', help='Directory of images')
    parser.add_argument('--csv', help='CSV file with image_path column')
    parser.add_argument('--csv-dir', help='Directory containing CSV files to run inference on')
    parser.add_argument('--output', help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--crop', action='store_true', help='Auto-crop via AI engine API before prediction')
    args = parser.parse_args()

    if not any([args.image, args.directory, args.csv, args.csv_dir]):
        parser.error("Specify --image, --directory, --csv, or --csv-dir")

    detector = TamperDetector(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        threshold=args.threshold,
        crop=args.crop
    )

    results = []

    if args.image:
        result = detector.predict(args.image)
        print(f"\nResult: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        results = [result]

    elif args.directory:
        results = detector.predict_directory(args.directory, args.batch_size)
        genuine = sum(1 for r in results if r.get('prediction') == 'genuine')
        tampered = sum(1 for r in results if r.get('prediction') == 'tampered')
        print(f"\nSummary: {genuine} genuine, {tampered} tampered ({len(results)} total)")

        if args.output:
            import pandas as pd
            frames = []
            for csv_file in Path(args.directory).glob('*.csv'):
                try:
                    df_c = pd.read_csv(csv_file)
                    if 'image_path' in df_c.columns and 'fraud_type' in df_c.columns:
                        frames.append(df_c[['image_path', 'fraud_type']])
                except Exception:
                    continue
            if frames:
                df = pd.concat(frames).drop_duplicates('image_path')
                class_to_idx = {'genuine': 0, 'tamper': 1}
                valid_results, labels = [], []
                for r in results:
                    if 'error' in r:
                        continue
                    row = df[df['image_path'] == r['image_path']]
                    if row.empty:
                        continue
                    fraud_type = str(row.iloc[0]['fraud_type']).strip().lower()
                    if fraud_type not in class_to_idx:
                        continue
                    valid_results.append(r)
                    labels.append(class_to_idx[fraud_type])
                if labels:
                    compute_and_save_metrics(valid_results, labels, args.output, args.threshold)

    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        image_paths = df['image_path'].tolist()
        results = detector.predict_batch(image_paths, args.batch_size)
        genuine = sum(1 for r in results if r.get('prediction') == 'genuine')
        tampered = sum(1 for r in results if r.get('prediction') == 'tampered')
        print(f"\nSummary: {genuine} genuine, {tampered} tampered ({len(results)} total)")

        if args.output and 'fraud_type' in df.columns:
            class_to_idx = {'genuine': 0, 'tamper': 1}
            valid_results, labels = [], []
            for r in results:
                if 'error' in r:
                    continue
                row = df[df['image_path'] == r['image_path']]
                if row.empty:
                    continue
                fraud_type = str(row.iloc[0]['fraud_type']).strip().lower()
                if fraud_type not in class_to_idx:
                    continue
                valid_results.append(r)
                labels.append(class_to_idx[fraud_type])
            if labels:
                compute_and_save_metrics(valid_results, labels, args.output, args.threshold)

    elif args.csv_dir:
        import pandas as pd
        frames = []
        for csv_file in sorted(Path(args.csv_dir).glob('*.csv')):
            try:
                df_c = pd.read_csv(csv_file)
                if 'image_path' in df_c.columns and 'fraud_type' in df_c.columns:
                    frames.append(df_c[['image_path', 'fraud_type']])
            except Exception:
                continue
        if not frames:
            parser.error(f"No valid CSV files with image_path and fraud_type found in {args.csv_dir}")
        df = pd.concat(frames).drop_duplicates('image_path')
        print(f"Loaded {len(df)} images from {len(frames)} CSV files")
        image_paths = df['image_path'].tolist()
        results = detector.predict_batch(image_paths, args.batch_size)
        genuine = sum(1 for r in results if r.get('prediction') == 'genuine')
        tampered = sum(1 for r in results if r.get('prediction') == 'tampered')
        print(f"\nSummary: {genuine} genuine, {tampered} tampered ({len(results)} total)")

        if args.output:
            class_to_idx = {'genuine': 0, 'tamper': 1}
            valid_results, labels = [], []
            for r in results:
                if 'error' in r:
                    continue
                row = df[df['image_path'] == r['image_path']]
                if row.empty:
                    continue
                fraud_type = str(row.iloc[0]['fraud_type']).strip().lower()
                if fraud_type not in class_to_idx:
                    continue
                valid_results.append(r)
                labels.append(class_to_idx[fraud_type])
            if labels:
                compute_and_save_metrics(valid_results, labels, args.output, args.threshold)

    if args.output and results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'prediction', 'confidence',
                                                   'prob_genuine', 'prob_tampered'])
            writer.writeheader()
            writer.writerows([r for r in results if 'error' not in r])
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
