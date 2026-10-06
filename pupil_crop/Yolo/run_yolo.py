"""
Main inference pipeline — YOLO pupil detection + high-quality crop export.

Run
---
    python Yolo\run_yolo.py ^
        --model  "runs\detect\pupil_detector\weights\best.pt" ^
        --input  "input" ^
        --output "output" ^
        --conf   0.10 ^
        --size   512
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ---------------------------------------------------------------------------
# CONFIG — edit when not using command-line args
# ---------------------------------------------------------------------------
CONFIG = {
    "model_path":     "runs/detect/pupil_detector/weights/best.pt",
    "input_dir":      "./input",
    "output_dir":     "./output",
    "confidence":     0.10,     # lowered from 0.25 — slit-lamp images score lower
    "output_size":    (512, 512),
    "save_annotated": True,
    "csv_report":     True,
}


# ---------------------------------------------------------------------------
# Quality pipeline (self-contained — no import needed)
# ---------------------------------------------------------------------------

import cv2
import numpy as np


def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def _unsharp_mask(bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0)


def _normalise(bgr: np.ndarray) -> np.ndarray:
    out = bgr.astype(np.float32)
    mn, mx = out.min(), out.max()
    if mx > mn:
        out = (out - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_crop(crop: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Full quality pipeline on a raw crop."""
    # 1. Bilateral denoise — preserves edges
    out = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)
    # 2. CLAHE contrast enhancement
    out = _apply_clahe(out)
    # 3. Unsharp mask sharpening
    out = _unsharp_mask(out, strength=0.5)
    # 4. Resize — Lanczos when upscaling, AREA when downscaling
    h, w = out.shape[:2]
    tw, th = output_size
    interp = cv2.INTER_LANCZOS4 if (w < tw or h < th) else cv2.INTER_AREA
    out = cv2.resize(out, (tw, th), interpolation=interp)
    # 5. Normalise dynamic range
    out = _normalise(out)
    return out


def quality_score(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def square_crop(
    image: np.ndarray,
    x: int, y: int, w: int, h: int,
    padding_frac: float = 0.08,
) -> np.ndarray | None:
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = image.shape[:2]
    cx, cy = x + w // 2, y + h // 2
    side = max(w, h)
    pad  = int(side * padding_frac)
    half = side // 2 + pad
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, img_w)
    y2 = min(cy + half, img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    ann = image.copy()
    for d in detections:
        x, y, w, h = d["bbox"]
        conf  = d["confidence"]
        score = d.get("quality_score", 0)
        cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label = f"Pupil {conf:.2f}  Q={score:.0f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        lx, ly = x, max(y - 10, th + 6)
        cv2.rectangle(ann, (lx, ly - th - 4), (lx + tw + 4, ly + 2),
                      (0, 255, 0), -1)
        cv2.putText(ann, label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return ann


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    model_path: str,
    input_dir: str,
    output_dir: str,
    confidence: float,
    output_size: tuple[int, int],
    save_annotated: bool,
    csv_report: bool,
) -> None:

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[ERROR] Model not found: {model_file.resolve()}")
        sys.exit(1)

    print(f"\n[run_yolo] Loading model: {model_path}")
    model = YOLO(str(model_file))

    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    crops_dir   = output_path / "crops"
    vis_dir     = output_path / "visualizations"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated:
        vis_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in sorted(input_path.iterdir())
        if p.suffix.lower() in SUPPORTED
    ]

    if not images:
        print(f"[ERROR] No images found in {input_dir}")
        sys.exit(1)

    print(f"[run_yolo] Confidence threshold : {confidence}")
    print(f"[run_yolo] Output size          : {output_size[0]}x{output_size[1]} px")
    print(f"[run_yolo] Processing {len(images)} images …\n")

    all_rows: list[dict] = []
    n_detected = 0
    n_skipped  = 0
    t_start    = time.time()

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            n_skipped += 1
            continue

        # YOLO inference
        yolo_results = model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )

        # Collect raw detections
        raw_dets: list[dict] = []
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                raw_dets.append({
                    "bbox":       (x1, y1, x2 - x1, y2 - y1),
                    "confidence": round(float(box.conf[0]), 4),
                    "class_id":   int(box.cls[0]),
                })

        if not raw_dets:
            print(f"  ⚠️  {img_path.name[:60]} → no detection at conf={confidence}")
            n_skipped += 1
            continue

        # Process each detection
        accepted: list[dict] = []
        stem = img_path.stem

        for idx, det in enumerate(raw_dets):
            x, y, w, h = det["bbox"]

            # Square crop + padding
            crop_raw = square_crop(image, x, y, w, h, padding_frac=0.08)
            if crop_raw is None:
                continue

            # Enhance
            crop_enhanced = enhance_crop(crop_raw, output_size)
            q_score = quality_score(crop_enhanced)

            # Quality gate — reject only extremely blurry crops
            if q_score < 5.0:
                print(f"  ⚠️  {img_path.name[:50]} "
                      f"→ crop rejected (blur score={q_score:.1f})")
                continue

            # Save as lossless PNG
            out_path = crops_dir / f"{stem}_pupil_{idx}.png"
            cv2.imwrite(
                str(out_path),
                crop_enhanced,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )

            accepted.append({
                **det,
                "quality_score": round(q_score, 2),
                "crop_path":     str(out_path.resolve()),
                "size":          output_size,
            })

        if not accepted:
            n_skipped += 1
            print(f"  ⚠️  {img_path.name[:60]} → detected but quality gate failed")
            continue

        # Save annotated visualisation
        if save_annotated:
            ann = draw_detections(image, accepted)
            vis_path = vis_dir / f"{stem}_annotated.png"
            cv2.imwrite(str(vis_path), ann, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        n_detected += len(accepted)
        best = accepted[0]
        print(f"  ✅ {img_path.name[:55]}")
        print(f"     conf={best['confidence']:.2f}  "
              f"Q={best['quality_score']:.0f}  "
              f"bbox={best['bbox']}")

        for r in accepted:
            all_rows.append({
                "filename":      img_path.name,
                "confidence":    r["confidence"],
                "bbox_x":        r["bbox"][0],
                "bbox_y":        r["bbox"][1],
                "bbox_w":        r["bbox"][2],
                "bbox_h":        r["bbox"][3],
                "quality_score": r["quality_score"],
                "crop_w":        r["size"][0],
                "crop_h":        r["size"][1],
                "crop_path":     r["crop_path"],
            })

    elapsed = time.time() - t_start

    # CSV report
    if csv_report and all_rows:
        csv_path = output_path / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[run_yolo] Results CSV → {csv_path}")

    print(f"\n{'='*55}")
    print(f"  Images processed : {len(images)}")
    print(f"  Crops saved      : {n_detected}")
    print(f"  Skipped          : {n_skipped}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"  Crops folder     : {crops_dir.resolve()}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO pupil detection pipeline")
    p.add_argument("--model",  default=CONFIG["model_path"])
    p.add_argument("--input",  default=CONFIG["input_dir"])
    p.add_argument("--output", default=CONFIG["output_dir"])
    p.add_argument("--conf",   default=CONFIG["confidence"], type=float,
                   help="Detection confidence threshold (default 0.10)")
    p.add_argument("--size",   default=512, type=int,
                   help="Output crop size in pixels (default 512)")
    p.add_argument("--no-vis", action="store_true",
                   help="Skip annotated visualisation images")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip CSV report")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        model_path    = args.model,
        input_dir     = args.input,
        output_dir    = args.output,
        confidence    = args.conf,
        output_size   = (args.size, args.size),
        save_annotated= not args.no_vis,
        csv_report    = not args.no_csv,
    )