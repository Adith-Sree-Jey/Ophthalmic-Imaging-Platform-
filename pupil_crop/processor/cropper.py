"""Crop utilities for pupil detections."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def crop_pupil(image: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 10) -> np.ndarray:
    """Crop a pupil region using bbox coordinates with clamped padding."""
    x, y, w, h = bbox
    image_height, image_width = image.shape[:2]

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, image_width)
    y2 = min(y + h + padding, image_height)

    return image[y1:y2, x1:x2].copy()


def crop_all_detections(
    image: np.ndarray, detections: list[dict], padding: int
) -> list[np.ndarray]:
    """Crop all detected pupil regions from an image."""
    crops: list[np.ndarray] = []
    for detection in detections:
        bbox = detection["bbox"]
        crop = crop_pupil(image, bbox, padding=padding)
        crops.append(crop)
    return crops


def save_crop(crop: np.ndarray, filename: str, index: int, output_dir: str) -> str:
    """Save a crop as a PNG file and return the absolute saved path."""
    crops_dir = Path(output_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    output_path = crops_dir / f"{stem}_pupil_{index}.png"
    cv2.imwrite(str(output_path), crop)
    return str(output_path.resolve())
