"""
models/cnn_model.py
───────────────────
BrainTumorCNN — a three-block convolutional network for 4-class MRI
classification (glioma / meningioma / pituitary / no_tumor).

Architecture
────────────
    Input  (B, 3, S, S)
     ↓
    Block 1: Conv2d 3→32,  BN, ReLU, MaxPool2d  →  (B, 32, S/2, S/2)
    Block 2: Conv2d 32→64, BN, ReLU, MaxPool2d  →  (B, 64, S/4, S/4)
    Block 3: Conv2d 64→128,BN, ReLU, MaxPool2d  →  (B, 128, S/8, S/8)
     ↓
    Flatten  →  128 * (S//8)²
    Linear 128*(S//8)² → 256, ReLU, Dropout(0.5)
    Linear 256 → num_classes (logits)

    S = input_size (default 128)

Changes from the original SimpleCNN
────────────────────────────────────
- Layers are defined as nn.Sequential blocks — easier to read, profile,
  and extend.
- Flattening uses nn.Flatten() instead of x.view(-1, ...) so the batch
  dimension is always handled correctly and the model is ONNX-exportable.
- The fully-connected head size is derived from input_size at construction
  time, not hard-coded — changing input_size in config.yaml just works.
- build_model() factory function prevents coupling to constructor args.
"""

import torch
import torch.nn as nn


class BrainTumorCNN(nn.Module):

    def __init__(self, num_classes: int = 4, input_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ── Block 2 ──────────────────────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ── Block 3 ──────────────────────────────────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Three MaxPool2d halve the spatial size three times: S → S//8
        reduced = input_size // 8
        flat_dim = 128 * reduced * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int = 4, input_size: int = 128) -> BrainTumorCNN:
    """
    Factory — returns a freshly initialised BrainTumorCNN.
    Call this instead of the constructor so future changes (e.g. swapping
    to ResNet) only need to happen in one place.
    """
    return BrainTumorCNN(num_classes=num_classes, input_size=input_size)


# ── Quick smoke-test ─────────────────────────────────────────────────── #

if __name__ == "__main__":
    model = build_model(num_classes=4, input_size=128)
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    assert out.shape == (2, 4), f"Unexpected output shape: {out.shape}"
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}  ✓")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {n_params:,}")
