#!/usr/bin/env python3
"""
Production inference for Multi-Model V7.

Usage:
    python inference_multimodal.py --checkpoint checkpoints_v7/best_qwk_model.pth \
        --ant path/to/anterior.jpg --rg path/to/red_glow.jpg --sl path/to/slit_lamp.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model.model_multimodal import MultiModalCataractModelV7, coral_logits_to_probs

# Pupil crop — optional import so file works even without ultralytics installed
try:
    from pupil_crop.Yolo.yolo_detector import YOLODetector, QualityPipeline
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


LABEL_NAMES = ["NS1", "NS2", "NS3", "NS4"]
ANT_SIZE = 260
RG_SIZE = 224
SL_SIZE = 260
NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass
class InferenceConfig:
    checkpoint: str
    device: str = "cuda"
    use_tta: bool = True
    ce_weight: float = 0.65
    temperature: float = 1.1
    rejection_threshold: float = 0.55
    borderline_gap: float = 0.15
    binary_boost: float = 0.05
    # Pupil crop settings — set yolo_weights to None to skip cropping
    yolo_weights: Optional[str] = None
    yolo_confidence: float = 0.10      # low threshold suits slit-lamp images
    output_dir: Optional[str] = "output"  # folder to save visuals + JSON


def get_base_transform(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size), A.Normalize(**NORM), ToTensorV2()])


def get_tta_transforms(size: int) -> List[A.Compose]:
    base = [A.Normalize(**NORM), ToTensorV2()]
    return [
        A.Compose([A.Resize(size, size)] + base),
        A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1)] + base),
        A.Compose([A.Resize(size, size), A.VerticalFlip(p=1)] + base),
        A.Compose([A.Resize(size, size), A.Rotate(limit=15, p=1)] + base),
        A.Compose([A.Resize(size, size), A.CLAHE(clip_limit=3, p=1)] + base),
    ]


def combine_v7_probabilities(
    ce_logits: torch.Tensor,
    coral_logits: torch.Tensor,
    binary_logits: torch.Tensor,
    ce_weight: float = 0.65,
    temperature: float = 1.1,
    binary_boost: float = 0.05,
) -> torch.Tensor:
    """
    Combine the three V7 heads into class probabilities.

    This mirrors the V7 evaluation-time logic so report generation uses the
    same decision rule as the model evaluation scripts.
    """
    ce_logits_scaled = ce_logits / temperature
    ce_probs = F.softmax(ce_logits_scaled, dim=1)
    coral_probs = coral_logits_to_probs(coral_logits)

    combined = ce_weight * ce_probs + (1.0 - ce_weight) * coral_probs

    binary_prob = torch.sigmoid(binary_logits.squeeze(-1))
    confidence_gate = (binary_prob - 0.5).abs() * 2.0
    ns1_reduction = binary_boost * confidence_gate * binary_prob

    combined = combined.clone()
    combined[:, 0] = combined[:, 0] * (1.0 - ns1_reduction)
    combined[:, 1] = combined[:, 1] + ns1_reduction * 0.5
    combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
    return combined


class CataractInferenceEngineV7:
    """V7 inference engine with TTA and report-friendly output."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        print(f"Loading V7 model from {config.checkpoint}...")
        self.model = MultiModalCataractModelV7(num_classes=4).to(self.device)
        state = torch.load(config.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.transform_ant = get_base_transform(ANT_SIZE)
        self.transform_rg = get_base_transform(RG_SIZE)
        self.transform_sl = get_base_transform(SL_SIZE)

        if config.use_tta:
            self.tta_ant = get_tta_transforms(ANT_SIZE)
            self.tta_rg = get_tta_transforms(RG_SIZE)
            self.tta_sl = get_tta_transforms(SL_SIZE)

        print(
            f"Engine ready. device={self.device}, "
            f"TTA={config.use_tta}, temp={config.temperature}, ce_weight={config.ce_weight}"
        )

        # Pupil crop — initialise YOLO detector if weights path provided
        self._yolo: Optional[YOLODetector] = None
        if config.yolo_weights is not None:
            if not _YOLO_AVAILABLE:
                print("WARNING: yolo_weights set but ultralytics not installed — skipping pupil crop")
            else:
                try:
                    self._yolo = YOLODetector(
                        model_path=config.yolo_weights,
                        confidence_threshold=config.yolo_confidence,
                        output_size=(ANT_SIZE, ANT_SIZE),   # crop direct to 260x260
                    )
                    print(f"Pupil crop enabled: {config.yolo_weights} (conf>={config.yolo_confidence})")
                except Exception as exc:
                    print(f"WARNING: Could not load YOLO weights — skipping pupil crop. ({exc})")

    def _crop_anterior(self, ant_rgb: np.ndarray) -> tuple[np.ndarray, bool, Optional[tuple]]:
        """Run YOLO pupil detection and return cropped anterior image.

        Returns
        -------
        (image, was_cropped, bbox_xywh)
            image        — RGB uint8, cropped if YOLO succeeded else original
            was_cropped  — True when YOLO found a pupil and crop was applied
            bbox_xywh    — (x, y, w, h) of best detection in original image coords,
                           or None if no detection
        """
        if self._yolo is None:
            return ant_rgb, False, None

        ant_bgr = cv2.cvtColor(ant_rgb, cv2.COLOR_RGB2BGR)

        try:
            detections = self._yolo.detect(ant_bgr)
        except Exception as exc:
            print(f"  Pupil crop: YOLO inference failed ({exc}) — using full image")
            return ant_rgb, False, None

        if not detections:
            print("  Pupil crop: no detection — using full image")
            return ant_rgb, False, None

        best = max(detections, key=lambda d: d["confidence"])
        x, y, w, h = best["bbox"]

        from pupil_crop.Yolo.yolo_detector import _square_crop
        crop_bgr = _square_crop(ant_bgr, x, y, w, h, padding_frac=0.20)

        if crop_bgr is None or crop_bgr.size == 0:
            print("  Pupil crop: invalid crop region — using full image")
            return ant_rgb, False, None

        quality = QualityPipeline(output_size=(ANT_SIZE, ANT_SIZE))
        crop_bgr_enhanced, quality_score = quality.process(crop_bgr)

        if not quality.passes_gate(crop_bgr_enhanced, quality_score):
            print(f"  Pupil crop: quality too low (score={quality_score:.1f}) — using full image")
            return ant_rgb, False, (x, y, w, h)

        crop_rgb = cv2.cvtColor(crop_bgr_enhanced, cv2.COLOR_BGR2RGB)
        print(f"  Pupil crop: OK (conf={best['confidence']:.2f}, quality={quality_score:.1f})")
        return crop_rgb, True, (x, y, w, h)

    def _crop_red_glow(self, rg_rgb: np.ndarray) -> tuple[np.ndarray, bool, Optional[tuple]]:
        """Crop red glow image to the bright reflex region using HSV colour detection.

        Mirrors RedGlowCropper.detect_reflex() from crop_roi_v3.py exactly.
        Output is resized to RG_SIZE x RG_SIZE (224x224).

        Returns
        -------
        (image, was_cropped, bbox_xywh)
            image       — RGB uint8, cropped+resized if detection succeeded else original
            was_cropped — True when reflex region was found
            bbox_xywh   — (x, y, w, h) bounding rect of detected region, or None
        """
        img_bgr = cv2.cvtColor(rg_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        try:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # Red/orange reflex: hue in [0-25] or [155-180], S>50, V>80
            mask1 = cv2.inRange(hsv, np.array([0,   50,  80]), np.array([25,  255, 255]))
            mask2 = cv2.inRange(hsv, np.array([155, 50,  80]), np.array([180, 255, 255]))
            color_mask = cv2.bitwise_or(mask1, mask2)

            # Also detect by brightness
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            combined = cv2.bitwise_or(color_mask, bright_mask)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour = None
            best_score   = 0.0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:
                    continue
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area_score   = min(area / (w * h * 0.25), 1.0)
                dist         = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
                center_score = 1 - (dist / (np.sqrt(w ** 2 + h ** 2) / 2))
                score        = area_score * 0.5 + center_score * 0.5
                if score > best_score:
                    best_score   = score
                    best_contour = contour

            if best_contour is not None:
                (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
                cx, cy, radius   = int(cx), int(cy), int(radius)
                method = "reflex_detection"
            else:
                # Fallback: centre crop
                cx, cy  = w // 2, h // 2
                radius  = int(min(w, h) * 0.35)
                method  = "center_fallback"

            # Square crop with 1.2x padding around reflex radius
            crop_half = max(int(radius * 1.2), int(min(w, h) * 0.25))
            x1 = max(0, cx - crop_half)
            y1 = max(0, cy - crop_half)
            x2 = min(w, cx + crop_half)
            y2 = min(h, cy + crop_half)

            # Force square
            side    = min(x2 - x1, y2 - y1)
            cx_crop = (x1 + x2) // 2
            cy_crop = (y1 + y2) // 2
            x1 = max(0, cx_crop - side // 2)
            y1 = max(0, cy_crop - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)

            cropped_bgr  = img_bgr[y1:y2, x1:x2]
            resized_bgr  = cv2.resize(cropped_bgr, (RG_SIZE, RG_SIZE),
                                      interpolation=cv2.INTER_LANCZOS4)
            cropped_rgb  = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

            was_cropped = (method == "reflex_detection")
            bbox        = (x1, y1, x2 - x1, y2 - y1)
            status      = "OK" if was_cropped else "fallback-centre"
            print(f"  Red glow crop: {status} (score={best_score:.2f})")
            return cropped_rgb, was_cropped, bbox

        except Exception as exc:
            print(f"  Red glow crop: failed ({exc}) — using full image")
            return rg_rgb, False, None

    def _save_outputs(
        self,
        case_id: str,
        ant_path: str,
        rg_path: str,
        sl_path: str,
        ant_cropped_rgb: np.ndarray,
        ant_was_cropped: bool,
        bbox_xywh: Optional[tuple],
        rg_cropped_rgb: np.ndarray,
        rg_was_cropped: bool,
        rg_bbox_xywh: Optional[tuple],
        result: Dict,
    ) -> Dict[str, str]:
        """Save all visual outputs for one case to output_dir.

        Saves:
          1. Cropped anterior image (what the CNN actually saw)
          2. Original anterior with YOLO bounding box drawn on it
          3. Side-by-side panel: original ant | cropped ant | red glow | slit lamp
          4. JSON result file

        Returns dict of saved file paths added to result under 'saved_files'.
        """
        if self.config.output_dir is None:
            return {}

        safe_id = str(case_id).replace(" ", "_").replace("\\", "_").replace("/", "_")
        out_root = Path(self.config.output_dir)
        (out_root / "crops").mkdir(parents=True, exist_ok=True)
        (out_root / "annotated").mkdir(parents=True, exist_ok=True)
        (out_root / "panels").mkdir(parents=True, exist_ok=True)
        (out_root / "json").mkdir(parents=True, exist_ok=True)

        saved = {}
        grade = result["grade"]
        conf  = result["confidence"]

        # ── 1. Cropped anterior (what CNN saw) ───────────────────────────────
        crop_path = out_root / "crops" / f"{safe_id}_anterior_crop.png"
        Image.fromarray(ant_cropped_rgb).save(str(crop_path))
        saved["anterior_crop"] = str(crop_path)

        # ── 1b. Cropped red glow (what CNN saw) ──────────────────────────────
        rg_crop_path = out_root / "crops" / f"{safe_id}_red_glow_crop.png"
        Image.fromarray(rg_cropped_rgb).save(str(rg_crop_path))
        saved["red_glow_crop"] = str(rg_crop_path)

        # ── 2. Original anterior with YOLO bbox annotation ───────────────────
        ant_orig_rgb = self._load_img(ant_path)
        ann = cv2.cvtColor(ant_orig_rgb, cv2.COLOR_RGB2BGR)
        if bbox_xywh is not None:
            x, y, w, h = bbox_xywh
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"Pupil ({grade} {conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(ann, (x, y - th - 10), (x + tw + 6, y), (0, 255, 0), -1)
            cv2.putText(ann, label, (x + 3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.putText(ann, "No pupil detected — full image used",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        ann_path = out_root / "annotated" / f"{safe_id}_anterior_annotated.png"
        cv2.imwrite(str(ann_path), ann)
        saved["anterior_annotated"] = str(ann_path)

        # ── 2b. Original red glow with reflex bbox annotation ────────────────
        rg_orig_rgb = self._load_img(rg_path)
        rg_ann = cv2.cvtColor(rg_orig_rgb, cv2.COLOR_RGB2BGR)
        if rg_bbox_xywh is not None:
            rx, ry, rw, rh = rg_bbox_xywh
            cv2.rectangle(rg_ann, (rx, ry), (rx + rw, ry + rh), (0, 165, 255), 3)
            rg_label = "Reflex region"
            (tw, th), _ = cv2.getTextSize(rg_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(rg_ann, (rx, ry - th - 10), (rx + tw + 6, ry), (0, 165, 255), -1)
            cv2.putText(rg_ann, rg_label, (rx + 3, ry - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.putText(rg_ann, "No reflex detected — full image used",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        rg_ann_path = out_root / "annotated" / f"{safe_id}_red_glow_annotated.png"
        cv2.imwrite(str(rg_ann_path), rg_ann)
        saved["red_glow_annotated"] = str(rg_ann_path)

        # ── 3. Side-by-side panel ─────────────────────────────────────────────
        PANEL_H = 320
        PANEL_W = 320
        LABEL_H = 48
        BORDER  = 3

        def _make_panel(img_rgb: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
            img_resized = cv2.resize(
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                (PANEL_W, PANEL_H), interpolation=cv2.INTER_LANCZOS4
            )
            label_bar = np.full((LABEL_H, PANEL_W, 3), 30, dtype=np.uint8)
            cv2.putText(label_bar, title, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(label_bar, subtitle, (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 160), 1)
            return np.vstack([img_resized, label_bar])

        sl_rgb = self._load_img(sl_path)

        review_tag = "  !! REVIEW" if result["needs_review"] else ""
        panels = [
            _make_panel(ant_orig_rgb,    "Anterior (original)", "before crop"),
            _make_panel(ant_cropped_rgb, "Anterior (cropped)" if ant_was_cropped else "Anterior (full)",
                        "input to CNN"),
            _make_panel(rg_orig_rgb,     "Red glow (original)", "before crop"),
            _make_panel(rg_cropped_rgb,  "Red glow (cropped)" if rg_was_cropped else "Red glow (full)",
                        "input to CNN"),
            _make_panel(sl_rgb,          "Slit lamp", "input to CNN (no crop)"),
        ]

        # Separator lines between panels
        sep = np.full((PANEL_H + LABEL_H, BORDER, 3), 80, dtype=np.uint8)
        row = panels[0]
        for p in panels[1:]:
            row = np.hstack([row, sep, p])

        # Header bar with grade result
        header_h = 52
        header = np.full((header_h, row.shape[1], 3), 20, dtype=np.uint8)
        header_text = (
            f"Case: {case_id}    Grade: {grade}    Conf: {conf:.1%}"
            f"    Ant crop: {'YES' if ant_was_cropped else 'NO'}"
            f"    RG crop: {'YES' if rg_was_cropped else 'NO'}{review_tag}"
        )
        cv2.putText(header, header_text, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 220, 255), 1)

        panel_img = np.vstack([header, row])
        panel_path = out_root / "panels" / f"{safe_id}_panel.png"
        cv2.imwrite(str(panel_path), panel_img)
        saved["panel"] = str(panel_path)

        # ── 4. JSON result ────────────────────────────────────────────────────
        json_path = out_root / "json" / f"{safe_id}_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        saved["json"] = str(json_path)

        print(f"  Outputs saved → {out_root / 'panels' / f'{safe_id}_panel.png'}")
        return saved

    def _load_img(self, path: str) -> np.ndarray:
        try:
            return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise ValueError(f"Could not load {path}: {exc}") from exc

    def _to_tensor(self, img: np.ndarray, transform: A.Compose) -> torch.Tensor:
        return transform(image=img)["image"].unsqueeze(0).to(self.device)

    def _run_model(
        self, ant_np: np.ndarray, rg_np: np.ndarray, sl_np: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.config.use_tta:
                ce_list, coral_list, binary_list, attn_list = [], [], [], []
                for t_ant, t_rg, t_sl in zip(self.tta_ant, self.tta_rg, self.tta_sl):
                    ant_t = self._to_tensor(ant_np, t_ant)
                    rg_t = self._to_tensor(rg_np, t_rg)
                    sl_t = self._to_tensor(sl_np, t_sl)
                    ce_out, coral_out, binary_out, attn = self.model(
                        ant_t, rg_t, sl_t, return_attention=True
                    )
                    ce_list.append(ce_out)
                    coral_list.append(coral_out)
                    binary_list.append(binary_out)
                    attn_list.append(attn)

                ce_logits = torch.stack(ce_list).mean(dim=0)
                coral_logits = torch.stack(coral_list).mean(dim=0)
                binary_logits = torch.stack(binary_list).mean(dim=0)
                attn_weights = torch.stack(attn_list).mean(dim=0).squeeze(0)
            else:
                ant_t = self._to_tensor(ant_np, self.transform_ant)
                rg_t = self._to_tensor(rg_np, self.transform_rg)
                sl_t = self._to_tensor(sl_np, self.transform_sl)
                ce_logits, coral_logits, binary_logits, attn_weights = self.model(
                    ant_t, rg_t, sl_t, return_attention=True
                )
                attn_weights = attn_weights.squeeze(0)

        return ce_logits, coral_logits, binary_logits, attn_weights

    def predict_case(
        self,
        ant_path: str,
        rg_path: str,
        sl_path: str,
        case_id: Optional[str] = None,
    ) -> Dict:
        ant_np = self._load_img(ant_path)
        rg_np = self._load_img(rg_path)
        sl_np = self._load_img(sl_path)

        # Keep original for annotation before it gets cropped
        ant_orig_rgb = ant_np.copy()

        # Crop anterior segment to pupil region before CNN
        ant_np, ant_was_cropped, bbox_xywh = self._crop_anterior(ant_np)

        # Crop red glow to reflex region before CNN
        rg_np, rg_was_cropped, rg_bbox_xywh = self._crop_red_glow(rg_np)

        ce_logits, coral_logits, binary_logits, attn_weights = self._run_model(ant_np, rg_np, sl_np)
        probs_tensor = combine_v7_probabilities(
            ce_logits=ce_logits,
            coral_logits=coral_logits,
            binary_logits=binary_logits,
            ce_weight=self.config.ce_weight,
            temperature=self.config.temperature,
            binary_boost=self.config.binary_boost,
        )

        probs = probs_tensor.squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        needs_review = False
        review_reason = None

        if confidence < self.config.rejection_threshold:
            needs_review = True
            review_reason = f"Low confidence ({confidence:.2f} < {self.config.rejection_threshold})"

        if pred_idx in [1, 2]:
            gap = abs(probs[1] - probs[2])
            if gap < self.config.borderline_gap:
                needs_review = True
                review_reason = f"NS2/NS3 borderline (gap={gap:.2f})"

        binary_prob = float(torch.sigmoid(binary_logits.squeeze()).item())
        attn = attn_weights.detach().cpu().numpy()

        result = {
            "case_id": case_id,
            "grade": LABEL_NAMES[pred_idx],
            "grade_index": pred_idx,
            "confidence": round(confidence, 4),
            "probabilities": {LABEL_NAMES[i]: round(float(probs[i]), 4) for i in range(4)},
            "binary_ns2plus_prob": round(binary_prob, 4),
            "needs_review": needs_review,
            "review_reason": review_reason,
            "attention": {
                "anterior": round(float(attn[0]), 3),
                "red_glow": round(float(attn[1]), 3),
                "slit_lamp": round(float(attn[2]), 3),
            },
            "input_paths": {
                "anterior_path": ant_path,
                "red_glow_path": rg_path,
                "slit_lamp_path": sl_path,
            },
            "pupil_crop_applied": ant_was_cropped,
            "bbox_xywh": list(bbox_xywh) if bbox_xywh is not None else None,
            "red_glow_crop_applied": rg_was_cropped,
            "red_glow_bbox_xywh": list(rg_bbox_xywh) if rg_bbox_xywh is not None else None,
        }

        # Save visual outputs to output_dir
        if self.config.output_dir is not None:
            saved = self._save_outputs(
                case_id=case_id or Path(ant_path).stem,
                ant_path=ant_path,
                rg_path=rg_path,
                sl_path=sl_path,
                ant_cropped_rgb=ant_np,
                ant_was_cropped=ant_was_cropped,
                bbox_xywh=bbox_xywh,
                rg_cropped_rgb=rg_np,
                rg_was_cropped=rg_was_cropped,
                rg_bbox_xywh=rg_bbox_xywh,
                result=result,
            )
            result["saved_files"] = saved

        return result

    def predict(self, ant_path: str, rg_path: str, sl_path: str) -> Dict:
        return self.predict_case(ant_path=ant_path, rg_path=rg_path, sl_path=sl_path)

    def predict_batch(self, records: List[Dict]) -> List[Dict]:
        results = []
        for i, rec in enumerate(records):
            case_id = rec.get("group_id") or rec.get("case_id") or f"sample_{i}"
            try:
                res = self.predict_case(
                    ant_path=rec["anterior_path"],
                    rg_path=rec["red_glow_path"],
                    sl_path=rec["slit_lamp_path"],
                    case_id=case_id,
                )
                res["group_id"] = case_id
                res["success"] = True
            except Exception as exc:
                res = {"group_id": case_id, "case_id": case_id, "success": False, "error": str(exc)}
            results.append(res)
        return results


def run_case_inference(
    checkpoint: str,
    ant_path: str,
    rg_path: str,
    sl_path: str,
    case_id: Optional[str] = None,
    device: str = "cuda",
    use_tta: bool = True,
) -> Dict:
    engine = CataractInferenceEngineV7(
        InferenceConfig(checkpoint=checkpoint, device=device, use_tta=use_tta)
    )
    return engine.predict_case(ant_path=ant_path, rg_path=rg_path, sl_path=sl_path, case_id=case_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ant", help="Anterior segment image")
    parser.add_argument("--rg", help="Red glow image")
    parser.add_argument("--sl", help="Slit lamp image")
    parser.add_argument("--case_id", help="Optional case identifier")
    parser.add_argument("--input_csv", help="CSV for batch inference")
    parser.add_argument("--output_csv", default="inference_results_v7.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--yolo_weights", default=None,
                        help="Path to YOLO best.pt for pupil crop (omit to skip)")
    parser.add_argument("--yolo_confidence", type=float, default=0.10)
    parser.add_argument("--output_dir", default="output",
                        help="Folder to save crops, annotated images, panels, JSON")
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--ce_weight", type=float, default=0.65)
    parser.add_argument("--binary_boost", type=float, default=0.05)
    args = parser.parse_args()

    config = InferenceConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        use_tta=not args.no_tta,
        ce_weight=args.ce_weight,
        temperature=args.temperature,
        binary_boost=args.binary_boost,
        yolo_weights=args.yolo_weights,
        yolo_confidence=args.yolo_confidence,
        output_dir=args.output_dir,
    )

    engine = CataractInferenceEngineV7(config)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        records = df.to_dict("records")

        print(f"\nBatch inference on {len(records)} samples...")
        results = engine.predict_batch(records)

        rows = []
        for result in results:
            row = {"group_id": result.get("group_id"), "success": result.get("success", True)}
            if result.get("success", True):
                row["grade"] = result["grade"]
                row["confidence"] = result["confidence"]
                row["binary_ns2plus"] = result["binary_ns2plus_prob"]
                row["needs_review"] = result["needs_review"]
                row["review_reason"] = result["review_reason"] or ""
                row["attn_anterior"] = result["attention"]["anterior"]
                row["attn_red_glow"] = result["attention"]["red_glow"]
                row["attn_slit_lamp"] = result["attention"]["slit_lamp"]
                for key, value in result["probabilities"].items():
                    row[f"prob_{key}"] = value
            else:
                row["error"] = result.get("error", "")
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.output_csv, index=False)

        successes = out_df[out_df["success"]]
        print(f"\nSaved to {args.output_csv}")
        print(f"  Success: {len(successes)}/{len(records)}")
        print(f"  Need review: {int(successes['needs_review'].sum()) if len(successes) else 0}")
        if len(successes) > 0:
            print(f"  Distribution: {successes['grade'].value_counts().to_dict()}")

    elif args.ant and args.rg and args.sl:
        result = engine.predict_case(
            ant_path=args.ant,
            rg_path=args.rg,
            sl_path=args.sl,
            case_id=args.case_id,
        )
        print("\n" + "=" * 50)
        print("  V7 INFERENCE RESULT")
        print("=" * 50)
        print(json.dumps(result, indent=2))
    else:
        print("Provide --input_csv or (--ant, --rg, --sl)")


if __name__ == "__main__":
    main()
