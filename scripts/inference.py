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


def main():
    parser = argparse.ArgumentParser(description='Tamper Detection Inference')
    parser.add_argument('--model', required=True, help='Path to checkpoint')
    parser.add_argument('--config', default=None, help='Path to config.yaml')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--directory', help='Directory of images')
    parser.add_argument('--csv', help='CSV file with image_path column')
    parser.add_argument('--output', help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--crop', action='store_true', help='Auto-crop via AI engine API before prediction')
    args = parser.parse_args()

    if not any([args.image, args.directory, args.csv]):
        parser.error("Specify --image, --directory, or --csv")

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

    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        image_paths = df['image_path'].tolist()
        results = detector.predict_batch(image_paths, args.batch_size)
        genuine = sum(1 for r in results if r.get('prediction') == 'genuine')
        tampered = sum(1 for r in results if r.get('prediction') == 'tampered')
        print(f"\nSummary: {genuine} genuine, {tampered} tampered ({len(results)} total)")

    if args.output and results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'prediction', 'confidence',
                                                   'prob_genuine', 'prob_tampered'])
            writer.writeheader()
            writer.writerows([r for r in results if 'error' not in r])
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
