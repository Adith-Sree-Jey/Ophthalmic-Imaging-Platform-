from __future__ import annotations

import cv2
import numpy as np

from retina_segmentation.src.data_loader import get_train_val_dataloaders


def _write_training_pairs(root, count: int = 10) -> tuple[str, str]:
    image_dir = root / "image"
    mask_dir = root / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(count):
        name = f"sample_{index:02d}.png"
        image = np.full((16, 16, 3), index, dtype=np.uint8)
        mask = np.full((16, 16), 255 if index % 2 else 0, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / name), image)
        assert cv2.imwrite(str(mask_dir / name), mask)
    return str(image_dir), str(mask_dir)


def test_train_validation_split_is_disjoint_deterministic_and_train_only(tmp_path):
    image_dir, mask_dir = _write_training_pairs(tmp_path / "train")

    train_loader, validation_loader = get_train_val_dataloaders(
        image_dir,
        mask_dir,
        val_fraction=0.2,
        split_seed=42,
        batch_size=2,
        img_size=16,
        add_green=False,
        pin_memory=False,
    )
    repeat_train, repeat_validation = get_train_val_dataloaders(
        image_dir,
        mask_dir,
        val_fraction=0.2,
        split_seed=42,
        batch_size=2,
        img_size=16,
        add_green=False,
        pin_memory=False,
    )

    train_indices = set(train_loader.dataset.indices)
    validation_indices = set(validation_loader.dataset.indices)
    assert len(train_indices) == 8
    assert len(validation_indices) == 2
    assert train_indices.isdisjoint(validation_indices)
    assert train_indices | validation_indices == set(range(10))
    assert train_loader.dataset.dataset.augment is True
    assert validation_loader.dataset.dataset.augment is False
    assert train_loader.dataset.indices == repeat_train.dataset.indices
    assert validation_loader.dataset.indices == repeat_validation.dataset.indices

