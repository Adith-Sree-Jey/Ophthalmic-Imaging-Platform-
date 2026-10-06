"""
Universal clinical slit-lamp pupil detector.

Uses TWO colour-based strategies in parallel, picks the best result,
then falls back to HoughCircles on grayscale if both fail.

Strategy 1 — R-B adaptive:
    Iris is coloured (R > B for brown/red iris). Pupil is neutral dark (R ≈ B).
    Threshold = global_mean(R-B) - 30.

Strategy 2 — Saturation + Value:
    Pupil is optically black → near-zero saturation.
    Iris and sclera both have significant colour → higher saturation.
    Threshold: S < 60 AND V < 120.

Both strategies produce contour candidates scored by:
    circularity × (1 / (1 + distance_to_centre / 400))
The highest-scoring candidate across both methods wins.

Drop-in replacement — same class name and return schema as the original.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class MediaPipeDetector:
    """Detect the pupil in clinical anterior-segment / slit-lamp eye images."""

    def __init__(self, confidence_threshold: float = 0.0) -> None:
        """Initialise detector.

        Parameters
        ----------
        confidence_threshold:
            Minimum confidence to accept a detection.
            Keep at 0.0 for slit-lamp images — the contrast-based
            confidence is naturally low (~0.03-0.10) because the whole
            image is dark.
        """
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect the pupil and return a list with one detection dict.

        Parameters
        ----------
        image : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        list[dict] with keys:
            ``bbox``        – (x, y, w, h) integers, top-left origin
            ``confidence``  – float in [0.0, 1.0]
        Empty list when no reliable pupil is found.
        """
        circle = self._colour_detect(image)

        if circle is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 2)
            circle = self._hough_fallback(image, blurred)

        if circle is None:
            return []

        confidence = self._compute_confidence(image, circle)
        if confidence < self.confidence_threshold:
            return []

        cx, cy, r = circle
        r = max(r, 1)
        img_h, img_w = image.shape[:2]
        x = max(int(cx - r), 0)
        y = max(int(cy - r), 0)
        w = min(int(2 * r), img_w - x)
        h = min(int(2 * r), img_h - y)

        return [{"bbox": (x, y, w, h), "confidence": round(confidence, 4)}]

    def close(self) -> None:
        """No-op — kept for interface compatibility."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _colour_detect(self, image: np.ndarray) -> tuple[int, int, int] | None:
        """Run both colour strategies and return the best pupil candidate.

        Parameters
        ----------
        image : np.ndarray
            BGR image.

        Returns
        -------
        (cx, cy, radius) or None.
        """
        all_candidates: list[tuple[float, int, int, int]] = []

        # --- Strategy 1: R-B adaptive threshold ---
        b_ch, _, r_ch = cv2.split(image)
        diff_rb = r_ch.astype(np.int16) - b_ch.astype(np.int16)
        rb_threshold = float(diff_rb.mean()) - 30.0
        mask_rb = (diff_rb < rb_threshold).astype(np.uint8) * 255
        all_candidates += self._contour_candidates(mask_rb, image.shape)

        # --- Strategy 2: HSV saturation + value ---
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, S, V = cv2.split(hsv)
        mask_sv = ((S < 60) & (V < 120)).astype(np.uint8) * 255
        all_candidates += self._contour_candidates(mask_sv, image.shape)

        if not all_candidates:
            return None

        # Pick highest score
        all_candidates.sort(key=lambda t: t[0], reverse=True)
        _, cx, cy, r = all_candidates[0]
        return (cx, cy, r)

    def _contour_candidates(
        self,
        mask_raw: np.ndarray,
        image_shape: tuple,
    ) -> list[tuple[float, int, int, int]]:
        """Apply morphology, find contours, score and return candidates.

        Parameters
        ----------
        mask_raw : np.ndarray
            Binary mask (uint8, 0/255).
        image_shape : tuple
            (H, W, C) of the source image.

        Returns
        -------
        List of (score, cx, cy, radius) tuples.
        """
        img_h, img_w = image_shape[:2]
        img_cx, img_cy = img_w // 2, img_h // 2

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  kernel_open)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[tuple[float, int, int, int]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (40_000 < area < 700_000):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = (4 * math.pi * area) / (perimeter ** 2)
            if circularity < 0.50:
                continue

            (cx_f, cy_f), r_f = cv2.minEnclosingCircle(contour)
            dist = math.sqrt((cx_f - img_cx) ** 2 + (cy_f - img_cy) ** 2)
            centre_weight = 1.0 / (1.0 + dist / 400.0)
            score = circularity * centre_weight

            candidates.append((score, int(cx_f), int(cy_f), int(r_f)))

        return candidates

    def _hough_fallback(
        self,
        original: np.ndarray,
        gray_blurred: np.ndarray,
    ) -> tuple[int, int, int] | None:
        """HoughCircles fallback when colour methods find nothing.

        Filters to circles in the central 60 % of frame and picks
        the darkest centre.

        Parameters
        ----------
        original : np.ndarray
            BGR image.
        gray_blurred : np.ndarray
            Pre-blurred grayscale image.

        Returns
        -------
        (cx, cy, radius) or None.
        """
        img_h, img_w = gray_blurred.shape

        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=200,
            param1=30,
            param2=10,
            minRadius=150,
            maxRadius=400,
        )

        if circles is None:
            return None

        circles = np.round(circles[0]).astype(int)
        cx_min, cx_max = img_w * 0.20, img_w * 0.80
        cy_min, cy_max = img_h * 0.20, img_h * 0.80
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        best: tuple[int, int, int] | None = None
        lowest_mean = float("inf")

        for cx, cy, r in circles:
            if not (cx_min < cx < cx_max and cy_min < cy < cy_max):
                continue
            mask = np.zeros(gray_orig.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), max(r // 3, 10), 255, -1)
            mean_val = cv2.mean(gray_orig, mask=mask)[0]
            if mean_val < lowest_mean:
                lowest_mean = mean_val
                best = (int(cx), int(cy), int(r))

        return best

    def _compute_confidence(
        self,
        image: np.ndarray,
        circle: tuple[int, int, int],
    ) -> float:
        """Confidence = normalised brightness contrast, pupil vs iris ring.

        Parameters
        ----------
        image : np.ndarray
            Original BGR image.
        circle : tuple
            (cx, cy, radius).

        Returns
        -------
        float in [0.0, 1.0].
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cx, cy, r = circle
        r = max(r, 5)
        img_h, img_w = gray.shape

        pupil_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(pupil_mask, (cx, cy), max(r // 2, 3), 255, -1)
        pupil_mean = cv2.mean(gray, mask=pupil_mask)[0]

        outer_r = min(int(r * 1.5), min(img_h, img_w) // 2)
        iris_outer = np.zeros(gray.shape, dtype=np.uint8)
        iris_inner = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(iris_outer, (cx, cy), outer_r, 255, -1)
        cv2.circle(iris_inner, (cx, cy), r,       255, -1)
        iris_mask = cv2.subtract(iris_outer, iris_inner)
        iris_mean = cv2.mean(gray, mask=iris_mask)[0]

        contrast = (iris_mean - pupil_mean) / 255.0
        return float(np.clip(contrast, 0.0, 1.0))