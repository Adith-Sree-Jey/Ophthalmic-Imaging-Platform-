"""
Individual impact analysis: run 4 experiments and report Dice, IoU, Sensitivity, Specificity, Accuracy.
Read-only: only READS outputs/checkpoints/best_model.pth, never writes or changes it.
Does not modify data_loader.py or evaluate.py.

Experiments:
  1. Baseline (Normal DeepVessel): no CLAHE, no morph
  2. CLAHE only: CLAHE + normalization, no morph
  3. CLAHE + normalization: same as 2 (no morph)
  4. CLAHE + normalization + post-processing (Final): CLAHE + norm + morph

Usage: python src/impact_analysis.py
Output: outputs/results_eval/impact_analysis.csv, impact_analysis.txt

Rough time: ~1–4 min (depends on test set size and CPU/GPU). ~20 images: ~1–2 min on GPU.
"""
import os
import sys
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from retina_segmentation.src.model import DeepVesselNet
from retina_segmentation.src.utils import clean_prediction

# ---------- Config (script-based: does not overwrite best_model.pth) ----------
BASE_DIR = Path(__file__).resolve().parents[1]
TEST_IMG_DIR = BASE_DIR / "Data" / "test" / "image"
TEST_MSK_DIR = BASE_DIR / "Data" / "test" / "mask"
CKPT_PATH = BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"
RESULTS_DIR = BASE_DIR / "outputs" / "results_eval"
IMG_SIZE = 512
THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Transforms ----------
def get_transform(use_clahe: bool):
    aug = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if use_clahe:
        aug.append(A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0))
    aug.extend([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return A.Compose(aug)


def apply_morph(mask_u8: np.ndarray) -> np.ndarray:
    return clean_prediction(mask_u8)


# ---------- Load image paths ----------
def get_test_paths():
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    imgs = sorted([p for p in os.listdir(TEST_IMG_DIR) if os.path.splitext(p)[1].lower() in exts])
    img_paths = [os.path.join(TEST_IMG_DIR, n) for n in imgs]
    mask_paths = [os.path.join(TEST_MSK_DIR, n) for n in imgs]
    return img_paths, mask_paths


# ---------- Metrics (single batch) ----------
def metrics_one(pred: torch.Tensor, mask: torch.Tensor, thr: float = THRESH):
    pred_b = (pred > thr).float()
    inter = (pred_b * mask).sum(dim=(2, 3))
    dice = ((2.0 * inter + 1e-6) / (pred_b.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) + 1e-6)).mean().item()
    union = (pred_b + mask).sum(dim=(2, 3)) - inter
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    tp = (pred_b * mask).sum(dim=(2, 3))
    fn = ((1 - pred_b) * mask).sum(dim=(2, 3))
    sens = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
    tn = ((1 - pred_b) * (1 - mask)).sum(dim=(2, 3))
    fp = (pred_b * (1 - mask)).sum(dim=(2, 3))
    spec = ((tn + 1e-6) / (tn + fp + 1e-6)).mean().item()
    total = pred_b.numel()
    acc = ((tp + tn).sum() + 1e-6) / (total + 1e-6)
    return dice, iou, sens, spec, acc.item()


# ---------- Run one experiment ----------
@torch.no_grad()
def run_experiment(name: str, use_clahe: bool, use_morph: bool, model, transform, img_paths, mask_paths):
    dices, ious, sens, specs, accs = [], [], [], [], []
    model.eval()
    for ip, mp in zip(img_paths, mask_paths):
        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_f = (mask > 127).astype("float32")
        data = transform(image=img_rgb, mask=mask_f)
        x = data["image"].unsqueeze(0).to(DEVICE)
        mask_out = data["mask"]
        if isinstance(mask_out, np.ndarray):
            m = torch.from_numpy(mask_out).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        else:
            m = mask_out.float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        prob = model(x)
        if use_morph:
            pred_np = (prob.squeeze().cpu().numpy() > THRESH).astype(np.uint8) * 255
            pred_np = apply_morph(pred_np)
            pred = torch.from_numpy((pred_np > 127).astype("float32")).unsqueeze(0).unsqueeze(0).to(DEVICE)
        else:
            pred = (prob > THRESH).float()
        d, j, se, sp, a = metrics_one(pred, m)
        dices.append(d)
        ious.append(j)
        sens.append(se)
        specs.append(sp)
        accs.append(a)
    return {
        "Experiment": name,
        "Dice": float(np.mean(dices)),
        "IoU": float(np.mean(ious)),
        "Sensitivity": float(np.mean(sens)),
        "Specificity": float(np.mean(specs)),
        "Accuracy": float(np.mean(accs)),
    }


# ---------- Main ----------
def main():
    start = time.perf_counter()
    # Never touch best_model.pth: only read it. Fail fast if missing.
    if not os.path.isfile(CKPT_PATH):
        print(f"Error: Checkpoint not found: {CKPT_PATH}")
        print("Run training first (python src/train.py) or ensure best_model.pth exists.")
        sys.exit(1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    img_paths, mask_paths = get_test_paths()
    if not img_paths:
        print("No test images in", TEST_IMG_DIR)
        return
    print(f"Using checkpoint (read-only): {CKPT_PATH}")
    print(f"Found {len(img_paths)} test images. Running 4 experiments...")
    model = DeepVesselNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    t_no_clahe = get_transform(use_clahe=False)
    t_clahe = get_transform(use_clahe=True)
    experiments = [
        ("1. Baseline (no CLAHE, no morph)", False, False),
        ("2. CLAHE only", True, False),
        ("3. CLAHE + normalization", True, False),
        ("4. CLAHE + normalization + post-processing (Final)", True, True),
    ]
    results = []
    for name, use_clahe, use_morph in experiments:
        transform = t_clahe if use_clahe else t_no_clahe
        row = run_experiment(name, use_clahe, use_morph, model, transform, img_paths, mask_paths)
        results.append(row)
        print(f"  {name}: Dice={row['Dice']:.4f} IoU={row['IoU']:.4f} Sens={row['Sensitivity']:.4f} Spec={row['Specificity']:.4f} Acc={row['Accuracy']:.4f}")
    elapsed = time.perf_counter() - start
    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, "impact_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Experiment", "Dice", "IoU", "Sensitivity", "Specificity", "Accuracy"])
        w.writeheader()
        w.writerows(results)
    txt_path = os.path.join(RESULTS_DIR, "impact_analysis.txt")
    with open(txt_path, "w") as f:
        f.write("Impact analysis – DeepVesselNet\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test images: {len(img_paths)} | threshold={THRESH}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment':<50} {'Dice':>7} {'IoU':>7} {'Sens':>7} {'Spec':>7} {'Acc':>7}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['Experiment']:<50} {r['Dice']:>7.4f} {r['IoU']:>7.4f} {r['Sensitivity']:>7.4f} {r['Specificity']:>7.4f} {r['Accuracy']:>7.4f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Time: {elapsed:.1f} s\n")
    print(f"\nWrote {csv_path}\nWrote {txt_path}\nTime: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
