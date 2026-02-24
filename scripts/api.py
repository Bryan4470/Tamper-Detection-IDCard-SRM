#!/usr/bin/env python3
"""
FastAPI Inference Server for RGB+Cb Tamper Detection Model
"""

import sys
from pathlib import Path
import base64
from io import BytesIO

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import yaml

from src.models import get_model


class TamperDetector:
    def __init__(self, model_path: str, config_path: str = None, device: str = None, threshold: float = 0.5):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = threshold

        if config_path is None:
            config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        print(f"Loading model: {model_path}")
        self.model = get_model(self.config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model = self.model.to(self.device).eval()
        self.image_size = self.config['data']['image_size']

        # Pre-compute normalization tensors on GPU
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        print(f"Model loaded on {self.device}")

        # Warm-up inference
        self._warmup()

    def _warmup(self):
        """Warm-up CUDA kernels with dummy inference"""
        print("Warming up model...")
        dummy = torch.zeros(1, 3, self.image_size, self.image_size).to(self.device)
        with torch.no_grad():
            self.model(dummy)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print("Warm-up complete")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor.unsqueeze(0)

    def predict(self, image: Image.Image) -> dict:
        img_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = F.softmax(output['logits'], dim=1)

        prob_genuine = probs[0, 0].item()
        prob_tampered = probs[0, 1].item()

        return {
            'prediction': 'genuine' if prob_genuine >= self.threshold else 'tampered',
            'confidence': prob_tampered,
            'prob_genuine': prob_genuine,
            'prob_tampered': prob_tampered
        }


def base64_to_pil(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data)).convert('RGB')


def call_crop_api(img_bytes: bytes) -> str:
    url = "http://10.1.1.49:8100/mq_beta/mykad_front"
    files = {"image": img_bytes}
    params = {"is_image": "0", "returnDetailed": "1"}
    response = requests.post(url, files=files, data=params, timeout=10)
    result = response.json()
    return result.get('card_cropped')


app = FastAPI(title="Tamper Detection API")
detector = None


@app.on_event("startup")
async def startup():
    import os
    global detector
    model_path = os.environ.get("MODEL_PATH", "/mnt2/auto-ekyc/id_physical_tamper_new/_dual_stream/trained_model/tamper_with_color_space.pth")
    config_path = os.environ.get("CONFIG_PATH", "/app/configs/config.yaml")
    threshold = float(os.environ.get("THRESHOLD", "0.5"))
    detector = TamperDetector(model_path, config_path, threshold=threshold)


@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: bool = False):
    """
    Single image prediction.
    - file: Image file to predict
    - crop: If true, auto-crop via AI engine API before prediction
    """
    try:
        contents = await file.read()

        if crop:
            cropped_base64 = call_crop_api(contents)
            if not cropped_base64:
                raise HTTPException(status_code=400, detail="Failed to get cropped image from API")
            image = base64_to_pil(cropped_base64)
        else:
            image = Image.open(BytesIO(contents))

        result = detector.predict(image)
        result['filename'] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(
    directory: str = None,
    csv: str = None,
    output: str = None,
    crop: bool = False
):
    """
    Batch inference from directory or CSV file.
    - directory: Path to folder containing images
    - csv: Path to CSV file with 'image_path' column
    - output: Optional path to save results CSV
    - crop: If true, auto-crop via AI engine API before prediction
    """
    import csv as csv_module
    import glob

    if not directory and not csv:
        raise HTTPException(status_code=400, detail="Must provide 'directory' or 'csv' parameter")

    # Collect image paths
    image_paths = []
    if directory:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(f"{directory}/{ext}"))
            image_paths.extend(glob.glob(f"{directory}/{ext.upper()}"))
    elif csv:
        with open(csv, 'r') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                image_paths.append(row['image_path'])

    # Process images
    results = []
    for img_path in image_paths:
        try:
            if crop:
                # Read image bytes and call crop API
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                cropped_base64 = call_crop_api(img_bytes)
                if not cropped_base64:
                    raise Exception("Failed to get cropped image from API")
                image = base64_to_pil(cropped_base64)
            else:
                image = Image.open(img_path)

            result = detector.predict(image)
            results.append({
                'image_path': img_path,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'prob_genuine': result['prob_genuine'],
                'prob_tampered': result['prob_tampered']
            })
        except Exception as e:
            results.append({
                'image_path': img_path,
                'prediction': 'error',
                'confidence': 0,
                'prob_genuine': 0,
                'prob_tampered': 0,
                'error': str(e)
            })

    # Save to CSV if output path provided
    if output and results:
        with open(output, 'w', newline='') as f:
            writer = csv_module.DictWriter(f, fieldnames=['image_path', 'prediction', 'confidence', 'prob_genuine', 'prob_tampered'])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in ['image_path', 'prediction', 'confidence', 'prob_genuine', 'prob_tampered']})

    return {
        'total': len(results),
        'output': output,
        'results': results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7500)
