"""
data/dataset.py
───────────────
Dataset class and DataLoader factories for the federated brain tumor project.

Two loader types are exposed:
  build_loaders(client_id, cfg)  — train + val loaders for a specific client
  build_test_loader(cfg)         — global held-out test loader (server only)

Design notes
────────────
- Training uses data augmentation (random flips, rotation, colour jitter).
  Validation and test use a simple resize + normalise pipeline.
- ImageNet mean/std normalisation is applied to all splits. Even without
  transfer learning this centres the pixel distribution and improves
  convergence stability compared to ToTensor() alone.
- num_workers=0 is intentional — MPI spawns multiple Python processes
  and each already occupies a core; extra DataLoader workers cause
  contention on a single machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Standard ImageNet statistics
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Transform pipelines ──────────────────────────────────────────────── #

def build_transforms(input_size: int, augment: bool) -> transforms.Compose:
    """
    Return a transform pipeline appropriate for training or evaluation.

    augment=True  — resize to (input_size+16) then random-crop, plus
                    flips, rotation, and colour jitter.
    augment=False — direct resize to input_size, no random ops.
    Both end with ToTensor + Normalize.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((input_size + 16, input_size + 16)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ── Dataset ──────────────────────────────────────────────────────────── #

class MRIDataset(Dataset):
    """
    Generic folder-based dataset for MRI classification.

    Expected directory layout::

        root/
            glioma/       image1.jpg  image2.jpg ...
            meningioma/   ...
            pituitary/    ...
            no_tumor/     ...

    Parameters
    ----------
    root        : Path to the split folder.
    class_names : Ordered list of class names — defines integer label mapping.
    transform   : torchvision transform pipeline.
    """

    def __init__(
        self,
        root: str | Path,
        class_names: list[str],
        transform: transforms.Compose,
    ):
        self.root = Path(root)
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(class_names)
        }
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for cls, idx in self.class_to_idx.items():
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in _IMG_EXTS:
                    self.samples.append((img_path, idx))

        if not self.samples:
            raise ValueError(
                f"No images found under {self.root}\n"
                "Run prepare_data.py first to create the partitions."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ── DataLoader factories ─────────────────────────────────────────────── #

def build_loaders(
    client_id: int,
    cfg,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders for one federated client.
    Reads from data/partitions/client_{client_id}/train|val/.
    """
    client_root = Path(cfg.data.partitions_dir) / f"client_{client_id}"

    train_ds = MRIDataset(
        root=client_root / "train",
        class_names=cfg.data.classes,
        transform=build_transforms(cfg.model.input_size, augment=True),
    )
    val_ds = MRIDataset(
        root=client_root / "val",
        class_names=cfg.data.classes,
        transform=build_transforms(cfg.model.input_size, augment=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def build_test_loader(cfg) -> DataLoader:
    """
    Build a DataLoader for the global held-out test set.
    Points directly at data/raw/brain_tumor/test/ — never partitioned,
    never touched by prepare_data.py.
    """
    test_root = Path(cfg.data.raw_dir) / "test"
    test_ds = MRIDataset(
        root=test_root,
        class_names=cfg.data.classes,
        transform=build_transforms(cfg.model.input_size, augment=False),
    )
    return DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
