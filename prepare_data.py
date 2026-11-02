"""
prepare_data.py
───────────────
One-time, idempotent data preparation.  Run this once before training.

What it does
────────────
1. Reads class folders from data/raw/brain_tumor/train/.
2. Per class, copies 20 % of images to data/raw/brain_tumor/val/.
   (Copies — never moves — so the original train/ is not modified.)
3. Partitions the remaining 80 % of training images across NUM_CLIENTS
   using stratified IID splitting: each class is shuffled independently
   then interleaved across clients, guaranteeing equal class proportions.
4. Copies each client's val slice to data/partitions/client_N/val/ so
   clients can evaluate locally without touching the global val/ folder.
5. Writes data/partitions/manifest.json recording all counts and the seed.

What it never does
──────────────────
- Move or delete any file.
- Touch the original data/raw/brain_tumor/test/ folder.
- Re-run if manifest.json already exists (idempotency guard).

Usage
─────
    python prepare_data.py
    python prepare_data.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from utils.config import load_config
from utils.seed import set_all_seeds


# ── File copy helper ─────────────────────────────────────────────────── #

def _copy_files(src_dir: Path, dst_dir: Path, filenames: list[str]) -> int:
    """Copy `filenames` from src_dir to dst_dir. Returns count copied."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        shutil.copy2(src_dir / name, dst_dir / name)
    return len(filenames)


# ── Main logic ───────────────────────────────────────────────────────── #

def prepare(cfg) -> None:
    raw_root = Path(cfg.data.raw_dir)
    partitions_root = Path(cfg.data.partitions_dir)
    manifest_path = partitions_root / "manifest.json"

    # ── Guard: skip if already prepared ─────────────────────────────
    if manifest_path.exists():
        print(f"[prepare_data] Manifest already exists at {manifest_path}.")
        print("[prepare_data] Skipping — delete manifest.json to re-prepare.")
        return

    set_all_seeds(cfg.data.seed)

    classes: list[str] = cfg.data.classes
    val_ratio: float = cfg.data.val_split_ratio
    num_clients: int = cfg.data.num_clients
    train_src = raw_root / "train"
    val_dst = raw_root / "val"

    # Manifest skeleton
    manifest: dict = {
        "seed": cfg.data.seed,
        "val_split_ratio": val_ratio,
        "num_clients": num_clients,
        "classes": classes,
        "clients": {
            str(i + 1): {"train": {}, "val": {}} for i in range(num_clients)
        },
        "global_val": {},
        "global_test": {},
    }

    print(f"\n[prepare_data] Starting data preparation")
    print(f"  Classes      : {classes}")
    print(f"  Val ratio    : {val_ratio}")
    print(f"  Num clients  : {num_clients}")
    print(f"  Seed         : {cfg.data.seed}\n")

    for cls in classes:
        cls_src = train_src / cls
        if not cls_src.is_dir():
            raise FileNotFoundError(
                f"Class folder not found: {cls_src}\n"
                f"Expected layout: {raw_root}/train/<class_name>/"
            )

        all_images: list[str] = sorted(
            p.name for p in cls_src.iterdir() if p.is_file()
        )
        random.shuffle(all_images)

        n_val = max(1, int(len(all_images) * val_ratio))
        val_images = all_images[:n_val]
        train_images = all_images[n_val:]

        # ── Global val split ─────────────────────────────────────────
        copied_val = _copy_files(cls_src, val_dst / cls, val_images)
        manifest["global_val"][cls] = copied_val

        # ── Partition training images across clients (IID) ────────────
        # Interleave: image[0] → client 1, image[1] → client 2, …
        # This guarantees identical class distributions regardless of
        # how images happen to be sorted on disk.
        for client_idx in range(num_clients):
            client_id = str(client_idx + 1)
            chunk = train_images[client_idx::num_clients]
            dst = partitions_root / f"client_{client_id}" / "train" / cls
            count = _copy_files(cls_src, dst, chunk)
            manifest["clients"][client_id]["train"][cls] = count

        # ── Per-client val slice (from the global val images) ─────────
        for client_idx in range(num_clients):
            client_id = str(client_idx + 1)
            val_chunk = val_images[client_idx::num_clients]
            dst = partitions_root / f"client_{client_id}" / "val" / cls
            count = _copy_files(cls_src, dst, val_chunk)
            manifest["clients"][client_id]["val"][cls] = count

        # ── Record global test counts (untouched folder) ──────────────
        test_cls = raw_root / "test" / cls
        test_count = len(list(test_cls.iterdir())) if test_cls.is_dir() else 0
        manifest["global_test"][cls] = test_count

        print(
            f"  {cls:>14s}  total={len(all_images):4d}  "
            f"train={len(train_images):4d}  val={len(val_images):3d}  "
            f"test={test_count:3d}"
        )

    # ── Write manifest ───────────────────────────────────────────────
    partitions_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[prepare_data] Done.")
    print(f"  Manifest     → {manifest_path}")
    print(f"  Partitions   → {partitions_root}/client_{{1..{num_clients}}}/")
    print(f"  Global val   → {val_dst}/")
    print(f"  Global test  → {raw_root}/test/  (untouched)\n")


# ── Entry point ──────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare federated data partitions for brain tumor MRI."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepare(cfg)
