"""Enhancement utilities for pupil crops."""

from __future__ import annotations

import cv2
import numpy as np


def quality_check(crop: np.ndarray) -> bool:
    """Return True when a crop is large enough for downstream processing."""
    return crop is not None and crop.ndim in (2, 3) and crop.shape[0] >= 20 and crop.shape[1] >= 20


def enhance_crop(crop: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement followed by sharpening."""
    lab_image = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, kernel)


def resize_crop(crop: np.ndarray, target_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize a crop to a fixed target size."""
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
