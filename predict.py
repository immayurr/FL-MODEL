"""
predict.py
──────────
Standalone inference script — load a saved global model and classify
a single MRI image.

Usage
─────
    python predict.py --model results/models/global_model.pt \\
                      --image path/to/mri.jpg

    python predict.py --model results/models/global_model.pt \\
                      --image path/to/mri.jpg \\
                      --config path/to/config.yaml \\
                      --device cpu

Output example
──────────────
    Prediction : glioma
    Confidence : 0.8741

    All class scores:
      glioma        0.8741  ████████████████████████████
      meningioma    0.0821  ██
      pituitary     0.0312  █
      no_tumor      0.0126
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from data.dataset import build_transforms
from models.cnn_model import build_model
from utils.config import load_config


def predict(model_path: str, image_path: str, config_path: str, device_str: str) -> None:
    # ── Config ───────────────────────────────────────────────────────
    cfg = load_config(config_path)
    class_names: list[str] = cfg.data.classes
    device = torch.device(device_str)

    # ── Path validation ───────────────────────────────────────────────
    model_file = Path(model_path)
    image_file = Path(image_path)

    if not model_file.exists():
        sys.exit(f"\nError: model file not found — {model_file.resolve()}")
    if not image_file.exists():
        sys.exit(f"\nError: image file not found — {image_file.resolve()}")

    # ── Load model ────────────────────────────────────────────────────
    model = build_model(
        num_classes=cfg.model.num_classes,
        input_size=cfg.model.input_size,
    ).to(device)

    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ── Preprocess image ──────────────────────────────────────────────
    # Uses the same eval transform as training — Normalize with ImageNet
    # stats must match the pipeline used during prepare_data.py / training.
    transform = build_transforms(cfg.model.input_size, augment=False)
    img = Image.open(image_file).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)

    # ── Inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    predicted_idx = int(torch.tensor(probs).argmax())
    confidence = probs[predicted_idx]

    # ── Output ────────────────────────────────────────────────────────
    print(f"\nPrediction : {class_names[predicted_idx]}")
    print(f"Confidence : {confidence:.4f}")
    print(f"\nAll class scores:")
    for cls, prob in zip(class_names, probs):
        bar_len = int(prob * 32)
        bar = "█" * bar_len
        print(f"  {cls:<14s}  {prob:.4f}  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brain tumor MRI inference using a trained federated global model."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the saved model weights (.pt file).",
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the MRI image to classify.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device — 'cpu' or 'cuda' (default: cpu).",
    )
    args = parser.parse_args()
    predict(args.model, args.image, args.config, args.device)


if __name__ == "__main__":
    main()
