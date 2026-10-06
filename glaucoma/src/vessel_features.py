from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class VesselFeatures:
    cdr: float
    vessel_density: float
    fractal_dimension: float
    mean_tortuosity: float
    vessel_area_ratio: float
    thin_vessel_ratio: float
    disc_region_density: float
    cdr_interpretation: str
    risk_level: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_vessel_features(
    vessel_mask: np.ndarray,
    original_image: Optional[np.ndarray] = None,
) -> VesselFeatures:
    binary = (vessel_mask > 127).astype(np.uint8) if vessel_mask.max() > 1 else vessel_mask.astype(np.uint8)
    height, width = binary.shape
    total_pixels = height * width

    vessel_pixels = int(binary.sum())
    vessel_area_ratio = float(vessel_pixels) / float(total_pixels or 1)

    disc_mask, cup_mask = _estimate_disc_and_cup(binary, original_image, height, width)
    disc_area = int(disc_mask.sum())
    cup_area = int(cup_mask.sum())
    cdr = float(cup_area) / float(disc_area) if disc_area > 0 else 0.5

    if disc_area > 0:
        disc_vessels = int((binary & disc_mask).sum())
        disc_region_density = float(disc_vessels) / float(disc_area)
    else:
        disc_region_density = vessel_area_ratio

    center_y, center_x = height // 2, width // 2
    radius_y, radius_x = int(height * 0.30), int(width * 0.30)
    zone = binary[center_y - radius_y:center_y + radius_y, center_x - radius_x:center_x + radius_x]
    zone_pixels = int(zone.size)
    zone_vessels = int(zone.sum())
    vessel_density = float(zone_vessels) / float(zone_pixels or 1)

    fractal_dimension = _box_counting_fractal_dim(binary)
    mean_tortuosity = _compute_tortuosity(binary)
    thin_vessel_ratio = _thin_vessel_ratio(binary)

    if cdr < 0.4:
        cdr_interpretation = "within normal limits (CDR < 0.4)"
        risk_level = "low"
    elif cdr < 0.5:
        cdr_interpretation = "low-normal range (CDR 0.4-0.5)"
        risk_level = "low"
    elif cdr < 0.6:
        cdr_interpretation = "borderline elevated (CDR 0.5-0.6) - monitor"
        risk_level = "moderate"
    elif cdr < 0.7:
        cdr_interpretation = "elevated (CDR 0.6-0.7) - glaucoma suspect"
        risk_level = "high"
    else:
        cdr_interpretation = "significantly elevated (CDR >= 0.7) - urgent review"
        risk_level = "high"

    return VesselFeatures(
        cdr=round(cdr, 3),
        vessel_density=round(vessel_density, 3),
        fractal_dimension=round(fractal_dimension, 3),
        mean_tortuosity=round(mean_tortuosity, 4),
        vessel_area_ratio=round(vessel_area_ratio, 3),
        thin_vessel_ratio=round(thin_vessel_ratio, 3),
        disc_region_density=round(disc_region_density, 3),
        cdr_interpretation=cdr_interpretation,
        risk_level=risk_level,
    )


def _estimate_disc_and_cup(
    binary: np.ndarray,
    original: Optional[np.ndarray],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    disc_mask = np.zeros((height, width), dtype=np.uint8)
    cup_mask = np.zeros((height, width), dtype=np.uint8)

    if original is not None and original.ndim == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            disc_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(disc_mask, [disc_contour], -1, 1, -1)
            moments = cv2.moments(disc_contour)
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                disc_radius = int(np.sqrt(cv2.contourArea(disc_contour) / np.pi))
                cup_radius = int(disc_radius * 0.5)
                cv2.circle(cup_mask, (center_x, center_y), cup_radius, 1, -1)
    else:
        center_y, center_x = height // 2, width // 2
        disc_radius = min(height, width) // 6
        cup_radius = int(disc_radius * 0.55)
        cv2.circle(disc_mask, (center_x, center_y), disc_radius, 1, -1)
        cv2.circle(cup_mask, (center_x, center_y), cup_radius, 1, -1)

    return disc_mask, cup_mask


def _box_counting_fractal_dim(binary: np.ndarray) -> float:
    sizes = [2, 4, 8, 16, 32, 64, 128]
    counts = []
    usable_sizes = []

    for size in sizes:
        height_boxes = binary.shape[0] // size
        width_boxes = binary.shape[1] // size
        if height_boxes == 0 or width_boxes == 0:
            break

        cropped = binary[:height_boxes * size, :width_boxes * size]
        reshaped = cropped.reshape(height_boxes, size, width_boxes, size)
        box_has_vessel = reshaped.any(axis=(1, 3))
        counts.append(int(box_has_vessel.sum()))
        usable_sizes.append(size)

    if len(counts) < 2:
        return 1.5

    log_sizes = np.log(np.array(usable_sizes, dtype=float))
    log_counts = np.log(np.array(counts, dtype=float) + 1e-6)
    coefficients = np.polyfit(log_sizes, log_counts, 1)
    return float(np.clip(-coefficients[0], 1.0, 2.0))


def _compute_tortuosity(binary: np.ndarray) -> float:
    try:
        from skimage.morphology import skeletonize

        skeleton = skeletonize(binary).astype(np.uint8) * 255
    except ImportError:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skeleton = cv2.erode(binary * 255, kernel, iterations=2)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    tortuosities = []
    for contour in contours:
        if len(contour) < 10:
            continue
        points = contour[:, 0, :]
        arc_length = cv2.arcLength(contour, closed=False)
        chord_length = float(np.linalg.norm(points[0] - points[-1]))
        if chord_length > 5:
            tortuosities.append((arc_length / chord_length) - 1.0)

    return float(np.mean(tortuosities)) if tortuosities else 0.1


def _thin_vessel_ratio(binary: np.ndarray) -> float:
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thick_only = cv2.morphologyEx(binary * 255, cv2.MORPH_OPEN, large_kernel)
    thin_only = cv2.subtract(binary * 255, thick_only)

    all_vessel = float(binary.sum()) + 1e-6
    thin_pixels = float((thin_only > 0).sum())
    return float(thin_pixels / all_vessel)
