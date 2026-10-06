#!/usr/bin/env python3
"""
crop_roi_v3.py — ROI Cropping for Cataract NS Grading
======================================================
Fixed version: detects DARK pupil, not bright iris.

Input:  Data/NS{1,2,3,4}/{modality}/*.jpg
Output: Data_Cropped_v3/NS{1,2,3,4}/{modality}/*.jpg

Usage:
    python crop_roi_v3.py --input_dir Data --output_dir Data_Cropped_v3 --visualize --workers 4
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from datetime import datetime
from PIL import Image


@dataclass
class CropResult:
    """Result of a single crop operation."""
    success: bool
    input_path: str
    output_path: Optional[str]
    method_used: str
    confidence: float
    error_msg: Optional[str] = None
    roi_center: Optional[Tuple[int, int]] = None
    roi_radius: Optional[int] = None


def load_image_pil(path: str) -> np.ndarray:
    """
    Load image using PIL (handles Tamil patient names with special chars).
    Returns BGR image for OpenCV processing.
    """
    try:
        pil_img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        raise ValueError(f"Could not load image: {path} - {e}")


# ═══════════════════════════════════════════════════════════════════
# ANTERIOR SEGMENT CROPPER — Detects DARK pupil
# ═══════════════════════════════════════════════════════════════════

class AnteriorSegmentCropper:
    """
    ROI cropping for anterior segment images.
    
    Key insight: Pupil is DARK, iris is BRIGHT.
    - Pupil radius: 8-28% of image width (small dark circle)
    - Iris radius: 40-50% of image width (we EXCLUDE this)
    - Score circles by DARKNESS — pupil is dark, iris is bright
    """
    
    OUTPUT_SIZE = 260
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess for dark circle (pupil) detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Blur to reduce noise for HoughCircles
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        
        return enhanced, blurred
    
    def detect_pupil(self, img: np.ndarray) -> Optional[Tuple[int, int, int, float]]:
        """
        Detect pupil using HoughCircles — look for DARK circles.
        
        Returns:
            (cx, cy, radius, confidence) or None
        """
        h, w = img.shape[:2]
        enhanced, blurred = self.preprocess(img)
        
        # Pupil radius: 8-28% of image width
        # This EXCLUDES the iris (40-50% of width)
        min_r = int(w * 0.08)
        max_r = int(w * 0.28)
        
        # Maximum distance from image center
        max_dist_from_center = min(w, h) * 0.35
        
        best_circle = None
        best_score = -1
        
        # Try multiple param sets for robustness
        param_sets = [
            {'dp': 1.2, 'param1': 60, 'param2': 25},
            {'dp': 1.5, 'param1': 50, 'param2': 20},
            {'dp': 1.0, 'param1': 70, 'param2': 30},
            {'dp': 1.3, 'param1': 45, 'param2': 22},
        ]
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=w // 3,
                param1=params['param1'],
                param2=params['param2'],
                minRadius=min_r,
                maxRadius=max_r
            )
            
            if circles is None:
                continue
            
            circles = np.uint16(np.around(circles))
            
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                
                # FILTER 1: Must be roughly centered
                dist_from_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                if dist_from_center > max_dist_from_center:
                    continue
                
                # FILTER 2: Must be DARK inside
                mask = np.zeros_like(enhanced)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                mean_inside = cv2.mean(enhanced, mask=mask)[0]
                
                # Pupil should be dark (< 80 intensity)
                # Iris is bright (> 100 intensity)
                if mean_inside > 80:
                    continue
                
                # SCORE: darker = better, more centered = better
                darkness_score = 1.0 - (mean_inside / 255.0)
                center_score = 1.0 - (dist_from_center / max_dist_from_center)
                score = darkness_score * 0.6 + center_score * 0.4
                
                if score > best_score:
                    best_score = score
                    best_circle = (cx, cy, r, score)
        
        return best_circle
    
    def find_darkest_region(self, img: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """
        Fallback: scan image for darkest circular region — that's the pupil.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Search in central 60% of image
        search_h = int(h * 0.6)
        search_w = int(w * 0.6)
        start_x = int(w * 0.2)
        start_y = int(h * 0.2)
        
        best_pos = (w // 2, h // 2)
        best_dark = 255
        probe_r = int(min(w, h) * 0.12)
        
        for y in range(start_y, start_y + search_h, 15):
            for x in range(start_x, start_x + search_w, 15):
                x1 = max(0, x - probe_r)
                y1 = max(0, y - probe_r)
                x2 = min(w, x + probe_r)
                y2 = min(h, y + probe_r)
                region = gray[y1:y2, x1:x2]
                mean = np.mean(region)
                if mean < best_dark:
                    best_dark = mean
                    best_pos = (x, y)
        
        radius = int(min(w, h) * 0.18)
        return best_pos, radius
    
    def crop_to_roi(self, img: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
        """
        Crop image centered on pupil.
        Pupil should fill 60-70% of the cropped image.
        """
        h, w = img.shape[:2]
        
        # Crop size = 2.2x pupil radius (gives ~60-70% pupil fill)
        crop_half = int(r * 2.2)
        
        # Minimum crop size (40% of image)
        min_crop_half = int(min(w, h) * 0.20)
        crop_half = max(crop_half, min_crop_half)
        
        # Calculate bounds
        x1 = max(0, cx - crop_half)
        y1 = max(0, cy - crop_half)
        x2 = min(w, cx + crop_half)
        y2 = min(h, cy + crop_half)
        
        # Make square
        side = min(x2 - x1, y2 - y1)
        cx_crop = (x1 + x2) // 2
        cy_crop = (y1 + y2) // 2
        
        x1 = max(0, cx_crop - side // 2)
        y1 = max(0, cy_crop - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        
        if x2 - x1 < side:
            x1 = max(0, x2 - side)
        if y2 - y1 < side:
            y1 = max(0, y2 - side)
        
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (self.OUTPUT_SIZE, self.OUTPUT_SIZE),
                            interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    def crop(self, img_path: str, output_path: str) -> CropResult:
        """Crop anterior segment image to pupil/lens region."""
        try:
            img = load_image_pil(img_path)
            
            result = self.detect_pupil(img)
            
            if result is not None:
                cx, cy, r, confidence = result
                method = "hough_circles"
            else:
                (cx, cy), r = self.find_darkest_region(img)
                confidence = 0.35
                method = "darkest_region_fallback"
            
            cropped = self.crop_to_roi(img, cx, cy, r)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cropped)
            
            return CropResult(
                success=True,
                input_path=img_path,
                output_path=output_path,
                method_used=method,
                confidence=confidence,
                roi_center=(cx, cy),
                roi_radius=r
            )
            
        except Exception as e:
            return CropResult(
                success=False,
                input_path=img_path,
                output_path=None,
                method_used="error",
                confidence=0.0,
                error_msg=str(e)
            )


# ═══════════════════════════════════════════════════════════════════
# RED GLOW CROPPER
# ═══════════════════════════════════════════════════════════════════

class RedGlowCropper:
    """
    ROI cropping for red glow (red reflex) images.
    Detects the bright orange/red reflex region.
    """
    
    OUTPUT_SIZE = 224
    
    def detect_reflex(self, img: np.ndarray) -> Optional[Tuple[int, int, int, float]]:
        """Detect the bright red/orange reflex region."""
        h, w = img.shape[:2]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Red/orange reflex: H in [0-25] or [155-180], S>50, V>80
        lower_red1 = np.array([0, 50, 80])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([155, 50, 80])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Also detect by brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        combined = cv2.bitwise_or(mask, bright_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            area_score = min(area / (w * h * 0.25), 1.0)
            dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            center_score = 1 - (dist / (np.sqrt(w**2 + h**2) / 2))
            
            score = area_score * 0.5 + center_score * 0.5
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
        return (int(cx), int(cy), int(radius), best_score)
    
    def crop(self, img_path: str, output_path: str) -> CropResult:
        """Crop red glow image to reflex region."""
        try:
            img = load_image_pil(img_path)
            h, w = img.shape[:2]
            
            result = self.detect_reflex(img)
            
            if result is not None:
                cx, cy, r, confidence = result
                method = "reflex_detection"
            else:
                cx, cy = w // 2, h // 2
                r = int(min(w, h) * 0.35)
                confidence = 0.3
                method = "center_fallback"
            
            crop_half = int(r * 1.2)
            crop_half = max(crop_half, int(min(w, h) * 0.25))
            
            x1 = max(0, cx - crop_half)
            y1 = max(0, cy - crop_half)
            x2 = min(w, cx + crop_half)
            y2 = min(h, cy + crop_half)
            
            side = min(x2 - x1, y2 - y1)
            cx_crop = (x1 + x2) // 2
            cy_crop = (y1 + y2) // 2
            x1 = max(0, cx_crop - side // 2)
            y1 = max(0, cy_crop - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)
            
            cropped = img[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (self.OUTPUT_SIZE, self.OUTPUT_SIZE),
                                interpolation=cv2.INTER_LANCZOS4)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, resized)
            
            return CropResult(
                success=True,
                input_path=img_path,
                output_path=output_path,
                method_used=method,
                confidence=confidence,
                roi_center=(cx, cy),
                roi_radius=r
            )
            
        except Exception as e:
            return CropResult(
                success=False,
                input_path=img_path,
                output_path=None,
                method_used="error",
                confidence=0.0,
                error_msg=str(e)
            )


# ═══════════════════════════════════════════════════════════════════
# SLIT LAMP CROPPER
# ═══════════════════════════════════════════════════════════════════

class SlitLampCropper:
    """
    ROI cropping for slit lamp images.
    Detects the bright vertical slit beam crossing the lens.
    """
    
    OUTPUT_SIZE = 260
    
    def detect_slit_beam(self, img: np.ndarray) -> Optional[Tuple[int, int, int, float]]:
        """Detect the bright slit beam."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_result = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            if bw == 0:
                continue
            aspect_ratio = bh / bw
            
            if aspect_ratio > 1.5:
                cx = x + bw // 2
                cy = y + bh // 2
                
                dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                center_score = 1 - (dist / (np.sqrt(w**2 + h**2) / 2))
                aspect_score = min(aspect_ratio / 4.0, 1.0)
                
                score = center_score * 0.6 + aspect_score * 0.4
                
                if score > best_score:
                    best_score = score
                    radius = max(bh // 2, int(min(w, h) * 0.25))
                    best_result = (cx, cy, radius, score)
        
        return best_result
    
    def crop(self, img_path: str, output_path: str) -> CropResult:
        """Crop slit lamp image to beam/lens region."""
        try:
            img = load_image_pil(img_path)
            h, w = img.shape[:2]
            
            result = self.detect_slit_beam(img)
            
            if result is not None:
                cx, cy, r, confidence = result
                method = "slit_beam_detection"
            else:
                cx = int(w * 0.55)
                cy = h // 2
                r = int(min(w, h) * 0.30)
                confidence = 0.3
                method = "center_fallback"
            
            crop_half = int(r * 1.3)
            crop_half = max(crop_half, int(min(w, h) * 0.25))
            
            x1 = max(0, cx - crop_half)
            y1 = max(0, cy - crop_half)
            x2 = min(w, cx + crop_half)
            y2 = min(h, cy + crop_half)
            
            side = min(x2 - x1, y2 - y1)
            cx_crop = (x1 + x2) // 2
            cy_crop = (y1 + y2) // 2
            x1 = max(0, cx_crop - side // 2)
            y1 = max(0, cy_crop - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)
            
            cropped = img[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (self.OUTPUT_SIZE, self.OUTPUT_SIZE),
                                interpolation=cv2.INTER_LANCZOS4)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, resized)
            
            return CropResult(
                success=True,
                input_path=img_path,
                output_path=output_path,
                method_used=method,
                confidence=confidence,
                roi_center=(cx, cy),
                roi_radius=r
            )
            
        except Exception as e:
            return CropResult(
                success=False,
                input_path=img_path,
                output_path=None,
                method_used="error",
                confidence=0.0,
                error_msg=str(e)
            )


# ═══════════════════════════════════════════════════════════════════
# MULTIMODAL CROPPER — Processes all grades and modalities
# ═══════════════════════════════════════════════════════════════════

class MultiModalCropper:
    """
    Processes all images across all grades and modalities.
    Input: Data/NS{1,2,3,4}/{modality}/*.jpg
    """
    
    MODALITY_CROPPERS = {
        'anterior_segment': AnteriorSegmentCropper,
        'red_glow': RedGlowCropper,
        'slit_lamp': SlitLampCropper,
    }
    
    def __init__(self, input_dir: str, output_dir: str, visualize: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.visualize = visualize
        self.results: Dict[str, List[CropResult]] = {
            'anterior_segment': [],
            'red_glow': [],
            'slit_lamp': []
        }
    
    def find_images(self) -> Dict[str, List[Tuple[str, str]]]:
        """Find all images."""
        images = {mod: [] for mod in self.MODALITY_CROPPERS.keys()}
        grades = ['NS1', 'NS2', 'NS3', 'NS4']
        
        modality_aliases = {
            'anterior_segment': ['anterior_segment', 'anterior segment'],
            'red_glow': ['red_glow', 'red glow'],
            'slit_lamp': ['slit_lamp', 'slit lamp'],
        }
        
        for grade in grades:
            grade_dir = self.input_dir / grade
            if not grade_dir.exists():
                continue
            
            for modality, aliases in modality_aliases.items():
                modality_dir = None
                for alias in aliases:
                    test_dir = grade_dir / alias
                    if test_dir.exists():
                        modality_dir = test_dir
                        break
                
                if modality_dir is None:
                    continue
                
                for img_file in modality_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        output_path = self.output_dir / grade / modality / img_file.name
                        images[modality].append((str(img_file), str(output_path)))
        
        return images
    
    def process_modality(self, modality: str, image_pairs: List[Tuple[str, str]],
                         workers: int = 4) -> List[CropResult]:
        """Process all images for a single modality."""
        if not image_pairs:
            return []
        
        cropper = self.MODALITY_CROPPERS[modality]()
        results = []
        
        print(f"  Found {len(image_pairs)} images")
        
        def process_one(pair):
            inp, out = pair
            if os.path.exists(out):
                return CropResult(
                    success=True,
                    input_path=inp,
                    output_path=out,
                    method_used="skipped_existing",
                    confidence=1.0
                )
            return cropper.crop(inp, out)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, pair): pair for pair in image_pairs}
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 100 == 0 or (i + 1) == len(image_pairs):
                    print(f"    Processed {i + 1}/{len(image_pairs)}")
        
        return results
    
    def process_all(self, workers: int = 4):
        """Process all modalities."""
        print(f"\nROI Cropping V3 — Dark Pupil Detection")
        print(f"Input:  {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print("=" * 55)
        
        all_images = self.find_images()
        total_images = sum(len(imgs) for imgs in all_images.values())
        print(f"\nFound {total_images} total images")
        
        for modality, image_pairs in all_images.items():
            print(f"\nProcessing {modality}...")
            results = self.process_modality(modality, image_pairs, workers)
            self.results[modality] = results
            
            if not results:
                print(f"  No images found")
                continue
            
            success = sum(1 for r in results if r.success)
            skipped = sum(1 for r in results if r.method_used == "skipped_existing")
            methods = {}
            low_conf = 0
            
            for r in results:
                if r.method_used != "skipped_existing":
                    methods[r.method_used] = methods.get(r.method_used, 0) + 1
                if r.success and r.confidence < 0.4 and r.method_used != "skipped_existing":
                    low_conf += 1
            
            print(f"  Success: {success}/{len(results)} (skipped existing: {skipped})")
            if methods:
                print(f"  Methods: {methods}")
            if low_conf > 0:
                print(f"  ⚠️  {low_conf} images with low confidence")
        
        self.save_summary()
        
        if self.visualize:
            self.save_visualizations()
    
    def save_summary(self):
        """Save detailed results to JSON."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'total': sum(len(r) for r in self.results.values()),
            'success': sum(sum(1 for x in r if x.success) for r in self.results.values()),
            'methods': {},
            'modalities': {}
        }
        
        for modality, results in self.results.items():
            summary['modalities'][modality] = {
                'total': len(results),
                'success': sum(1 for r in results if r.success),
                'methods': {},
                'low_confidence': [],
                'failures': []
            }
            
            for r in results:
                if r.method_used != "skipped_existing":
                    method = r.method_used
                    summary['modalities'][modality]['methods'][method] = \
                        summary['modalities'][modality]['methods'].get(method, 0) + 1
                    
                    key = f"{modality}_{method}"
                    summary['methods'][key] = summary['methods'].get(key, 0) + 1
                
                if r.success and r.confidence < 0.4 and r.method_used != "skipped_existing":
                    summary['modalities'][modality]['low_confidence'].append({
                        'path': r.input_path,
                        'confidence': r.confidence
                    })
                
                if not r.success:
                    summary['modalities'][modality]['failures'].append({
                        'path': r.input_path,
                        'error': r.error_msg
                    })
        
        os.makedirs(self.output_dir, exist_ok=True)
        summary_path = self.output_dir / 'crop_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved → {summary_path}")
    
    def save_visualizations(self):
        """Save sample visualizations."""
        viz_dir = self.output_dir / 'visualizations'
        os.makedirs(viz_dir, exist_ok=True)
        
        for modality, results in self.results.items():
            valid_results = [r for r in results 
                           if r.success and r.method_used != "skipped_existing"]
            
            if not valid_results:
                continue
            
            for i, r in enumerate(valid_results[:5]):
                try:
                    orig = load_image_pil(r.input_path)
                    cropped = cv2.imread(r.output_path)
                    
                    if orig is None or cropped is None:
                        continue
                    
                    if r.roi_center and r.roi_radius:
                        cv2.circle(orig, r.roi_center, r.roi_radius, (0, 255, 0), 3)
                        cv2.circle(orig, r.roi_center, 5, (0, 0, 255), -1)
                    
                    display_h = 300
                    scale = display_h / orig.shape[0]
                    orig_resized = cv2.resize(orig, (int(orig.shape[1] * scale), display_h))
                    crop_resized = cv2.resize(cropped, (display_h, display_h))
                    
                    combined = np.hstack([
                        orig_resized,
                        np.ones((display_h, 10, 3), dtype=np.uint8) * 128,
                        crop_resized
                    ])
                    
                    cv2.putText(combined, f"{modality} - {r.method_used} ({r.confidence:.2f})",
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    out_path = viz_dir / f"{modality}_sample_{i+1}.jpg"
                    cv2.imwrite(str(out_path), combined)
                    
                except Exception as e:
                    print(f"  Visualization error: {e}")
        
        print(f"Visualizations saved → {viz_dir}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ROI Cropping V3 — Dark Pupil Detection')
    parser.add_argument('--input_dir', required=True, 
                        help='Input directory (contains NS1, NS2, NS3, NS4)')
    parser.add_argument('--output_dir', required=True, 
                        help='Output directory for cropped images')
    parser.add_argument('--visualize', action='store_true', 
                        help='Save sample visualizations')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    cropper = MultiModalCropper(args.input_dir, args.output_dir, args.visualize)
    cropper.process_all(args.workers)


if __name__ == '__main__':
    main()