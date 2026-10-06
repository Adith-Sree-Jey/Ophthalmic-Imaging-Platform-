"""
YOLOv8 Pupil Detector with High-Quality Output Processing.

Pipeline
--------
1. YOLOv8 inference  → bounding box + confidence
2. Smart crop        → square crop centred on detection with padding
3. Quality pipeline  → denoise → CLAHE → sharpen → upscale → normalize
4. Quality gate      → rejects crops that are too blurry / too small
5. Save              → lossless PNG (never JPEG) for downstream model

Usage
-----
    from yolo_detector import YOLODetector

    detector = YOLODetector(model_path="runs/detect/train/weights/best.pt")
    results  = detector.detect(image)          # list[dict]
    crops    = detector.detect_and_crop(image, filename, output_dir)
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Optional import — graceful error if ultralytics not installed
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum side-length (px) of a crop accepted for downstream use
MIN_CROP_SIZE: int = 64

# Target output size fed to the next model
OUTPUT_SIZE: tuple[int, int] = (512, 512)

# Laplacian variance threshold — below this the crop is considered too blurry
BLUR_THRESHOLD: float = 20.0

# CLAHE parameters
CLAHE_CLIP: float = 2.0
CLAHE_TILE: tuple[int, int] = (8, 8)

# Super-resolution upscale factor used when crop is smaller than OUTPUT_SIZE
SR_SCALE: int = 2


# ---------------------------------------------------------------------------
# YOLODetector
# ---------------------------------------------------------------------------

class YOLODetector:
    """Detect the pupil with YOLOv8 and return high-quality crops.

    Parameters
    ----------
    model_path:
        Path to a fine-tuned ``best.pt`` weights file.
    confidence_threshold:
        Minimum YOLO confidence to accept a detection (0-1).
    output_size:
        (width, height) in pixels for the saved crop.  Defaults to
        ``OUTPUT_SIZE`` (512 × 512).
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        output_size: tuple[int, int] = OUTPUT_SIZE,
    ) -> None:
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed.  "
                "Run: pip install ultralytics"
            )
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                "Train the model first with train.py"
            )

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.output_size = output_size
        self._quality = QualityPipeline(output_size=output_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run YOLOv8 inference and return raw detection dicts.

        Parameters
        ----------
        image:
            BGR image (H, W, 3).

        Returns
        -------
        list[dict] each with:
            ``bbox``        – (x, y, w, h) integers
            ``confidence``  – float
            ``class_id``    – int
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[dict] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                detections.append({
                    "bbox":       (x1, y1, x2 - x1, y2 - y1),
                    "confidence": round(conf, 4),
                    "class_id":   cls,
                })
        return detections

    def detect_and_crop(
        self,
        image: np.ndarray,
        filename: str,
        output_dir: str,
    ) -> list[dict]:
        """Detect, crop, enhance, quality-check, and save.

        Parameters
        ----------
        image:
            BGR image.
        filename:
            Original filename (used for output naming).
        output_dir:
            Root output directory.  Crops go to ``output_dir/crops/``.

        Returns
        -------
        list[dict] — one entry per accepted detection:
            ``bbox``        – (x, y, w, h)
            ``confidence``  – float
            ``crop_path``   – absolute path to saved PNG
            ``quality_score`` – float (higher = sharper)
            ``size``        – (w, h) of saved crop
        """
        detections = self.detect(image)
        if not detections:
            return []

        crops_dir = Path(output_dir) / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        stem = Path(filename).stem

        for idx, det in enumerate(detections):
            x, y, w, h = det["bbox"]

            # --- 1. Square crop with padding ---
            crop_raw = _square_crop(image, x, y, w, h, padding_frac=0.08)
            if crop_raw is None or crop_raw.size == 0:
                continue

            # --- 2. Quality pipeline ---
            crop_enhanced, quality_score = self._quality.process(crop_raw)

            # --- 3. Quality gate ---
            if not self._quality.passes_gate(crop_enhanced, quality_score):
                continue

            # --- 4. Save as lossless PNG ---
            out_path = crops_dir / f"{stem}_pupil_{idx}.png"
            cv2.imwrite(
                str(out_path),
                crop_enhanced,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],   # 0 = no compression, max quality
            )

            results.append({
                "bbox":          det["bbox"],
                "confidence":    det["confidence"],
                "crop_path":     str(out_path.resolve()),
                "quality_score": round(quality_score, 2),
                "size":          (crop_enhanced.shape[1], crop_enhanced.shape[0]),
            })

        return results


# ---------------------------------------------------------------------------
# Quality pipeline
# ---------------------------------------------------------------------------

class QualityPipeline:
    """Enhance a raw pupil crop so it is clean input for a downstream model.

    Steps (in order)
    ----------------
    1. Denoise          — bilateral filter (preserves edges)
    2. CLAHE            — contrast-limited adaptive histogram equalisation
    3. Unsharp mask     — sharpening without amplifying noise
    4. Upscale          — Lanczos if crop < output_size, else just resize
    5. Normalise range  – stretch to full 0-255 dynamic range

    Parameters
    ----------
    output_size:
        (width, height) of the final crop in pixels.
    """

    def __init__(self, output_size: tuple[int, int] = OUTPUT_SIZE) -> None:
        self.output_size = output_size
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP,
            tileGridSize=CLAHE_TILE,
        )

    def process(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply the full quality pipeline.

        Parameters
        ----------
        crop:
            Raw BGR crop from the original image.

        Returns
        -------
        (enhanced_crop, quality_score)
            enhanced_crop  – BGR uint8 array of shape ``(*output_size, 3)``
            quality_score  – Laplacian variance (higher = sharper)
        """
        out = crop.copy()

        # Step 1 — Bilateral denoise (keeps pupil-iris edge crisp)
        out = cv2.bilateralFilter(out, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 2 — CLAHE on L channel (LAB colourspace)
        out = self._apply_clahe(out)

        # Step 3 — Unsharp mask sharpening
        out = self._unsharp_mask(out, strength=0.6)

        # Step 4 — Upscale / resize to target
        out = self._smart_resize(out)

        # Step 5 — Normalise dynamic range
        out = self._normalise(out)

        # Compute quality score on grayscale of final output
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        quality_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return out, quality_score

    def passes_gate(self, crop: np.ndarray, quality_score: float) -> bool:
        """Return True when the crop meets minimum quality requirements.

        Parameters
        ----------
        crop:
            Enhanced crop.
        quality_score:
            Laplacian variance from ``process()``.
        """
        h, w = crop.shape[:2]
        if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
            return False
        if quality_score < BLUR_THRESHOLD:
            return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_clahe(self, bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel of the LAB image."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        merged = cv2.merge((l_ch, a_ch, b_ch))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _unsharp_mask(
        self,
        bgr: np.ndarray,
        strength: float = 0.6,
        blur_size: int = 0,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Unsharp masking — sharpens without amplifying noise.

        Parameters
        ----------
        strength:
            How much sharpening to apply (0 = none, 1 = heavy).
        """
        blurred = cv2.GaussianBlur(bgr, (blur_size, blur_size), sigma) \
            if blur_size > 0 \
            else cv2.GaussianBlur(bgr, (0, 0), sigma)
        sharpened = cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0)
        return sharpened

    def _smart_resize(self, bgr: np.ndarray) -> np.ndarray:
        """Resize to output_size.

        Uses Lanczos (high quality) when upscaling,
        area interpolation (avoids aliasing) when downscaling.
        """
        target_w, target_h = self.output_size
        h, w = bgr.shape[:2]

        if w < target_w or h < target_h:
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_AREA

        return cv2.resize(bgr, (target_w, target_h), interpolation=interpolation)

    @staticmethod
    def _normalise(bgr: np.ndarray) -> np.ndarray:
        """Stretch pixel values to use the full 0-255 dynamic range."""
        out = bgr.astype(np.float32)
        min_val = out.min()
        max_val = out.max()
        if max_val > min_val:
            out = (out - min_val) / (max_val - min_val) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_crop(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    padding_frac: float = 0.08,
) -> np.ndarray | None:
    """Crop a square region centred on the detection with padding.

    Extends the shorter axis to match the longer one so the crop is
    always square, then adds a percentage padding on all sides.

    Parameters
    ----------
    image:
        Full BGR image.
    x, y, w, h:
        Bounding box (top-left origin).
    padding_frac:
        Fraction of the square side to add as padding (default 8 %).

    Returns
    -------
    Cropped BGR array or None if the box is invalid.
    """
    if w <= 0 or h <= 0:
        return None

    img_h, img_w = image.shape[:2]

    # Centre of detection
    cx = x + w // 2
    cy = y + h // 2

    # Make square using the larger dimension
    side = max(w, h)
    pad  = int(side * padding_frac)
    half = side // 2 + pad

    # Clamp to image bounds
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, img_w)
    y2 = min(cy + half, img_h)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2].copy()