"""Image loading helpers built on OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Read an image from disk using OpenCV in BGR format."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_all_images(folder: str) -> list[tuple[str, np.ndarray]]:
    """Load all JPG and PNG images from a folder."""
    folder_path = Path(folder)
    supported_extensions = {".jpg", ".jpeg", ".png"}
    images: list[tuple[str, np.ndarray]] = []

    if not folder_path.exists():
        return images

    for image_path in sorted(folder_path.iterdir()):
        if image_path.suffix.lower() not in supported_extensions:
            continue

        image = load_image(str(image_path))
        if validate_image(image):
            images.append((image_path.name, image))

    return images


def validate_image(image: np.ndarray) -> bool:
    """Return True when an image array exists and has a valid shape."""
    return image is not None and image.ndim in (2, 3) and image.size > 0
