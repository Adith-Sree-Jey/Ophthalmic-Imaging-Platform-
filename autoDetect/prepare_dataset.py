"""
prepare_dataset.py

Collects anterior_segment / red_glow / slit_lamp images from all NS grade
folders and builds a flat train/val/test split ready for training.

Output structure:
    autoDetect/
    └── dataset/
        ├── train/
        │   ├── anterior_segment/
        │   ├── red_glow/
        │   └── slit_lamp/
        ├── val/
        │   ├── anterior_segment/
        │   ├── red_glow/
        │   └── slit_lamp/
        └── test/
            ├── anterior_segment/
            ├── red_glow/
            └── slit_lamp/

Usage:
    python prepare_dataset.py
"""

from __future__ import annotations

import random
import shutil
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent


def _repo_relative_env_path(name: str, default: Path) -> Path:
    configured = Path(os.getenv(name, str(default))).expanduser()
    return configured if configured.is_absolute() else REPO_ROOT / configured


DATA_ROOT = _repo_relative_env_path("OPHTHALMIC_CATARACT_DATA_ROOT", REPO_ROOT / "Data")
OUTPUT_DIR = _repo_relative_env_path(
    "OPHTHALMIC_AUTODETECT_DATASET_DIR", MODULE_DIR / "dataset"
)

GRADES     = ["NS1", "NS2", "NS3", "NS4"]
CLASSES    = ["anterior_segment", "red_glow", "slit_lamp"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # remainder

SEED = 42
# ---------------------------------------------------------------------------


def collect_images(data_root: Path) -> dict[str, list[Path]]:
    """Gather all image paths grouped by class."""
    class_images: dict[str, list[Path]] = {cls: [] for cls in CLASSES}

    for grade in GRADES:
        for cls in CLASSES:
            folder = data_root / grade / cls
            if not folder.exists():
                print(f"  WARNING: folder not found — {folder}")
                continue
            images = [
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            ]
            class_images[cls].extend(images)
            print(f"  {grade}/{cls}: {len(images)} images")

    return class_images


def split(images: list[Path], seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = shuffled[:n_train]
    val   = shuffled[n_train: n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


def copy_split(images: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        shutil.copy2(src, dest_dir / src.name)


def main() -> None:
    print("=" * 60)
    print("  Ophthalmic Imaging - Image Type Classifier Dataset Preparation")
    print("=" * 60)

    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing dataset at {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print("\nCollecting images...")
    class_images = collect_images(DATA_ROOT)

    print("\nSplitting and copying...")
    totals = {"train": 0, "val": 0, "test": 0}

    for cls, images in class_images.items():
        train, val, test = split(images, SEED)
        copy_split(train, OUTPUT_DIR / "train" / cls)
        copy_split(val,   OUTPUT_DIR / "val"   / cls)
        copy_split(test,  OUTPUT_DIR / "test"  / cls)

        totals["train"] += len(train)
        totals["val"]   += len(val)
        totals["test"]  += len(test)

        print(f"  {cls}: {len(train)} train | {len(val)} val | {len(test)} test")

    print("\n" + "=" * 60)
    print(f"  Train : {totals['train']} images")
    print(f"  Val   : {totals['val']}   images")
    print(f"  Test  : {totals['test']}  images")
    print(f"  Total : {sum(totals.values())} images")
    print(f"\n  Dataset saved → {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
