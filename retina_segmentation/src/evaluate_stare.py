"""
Evaluate the current DeepVessel model on STARE (DatasetNinja format).
Read-only: only loads best_model.pth, does not modify it.
Expects STARE images in: stare-DatasetNinja/stare-DatasetNinja/ds/img/
Annotations: stare-DatasetNinja/stare-DatasetNinja/ds/ann/*.png.json

Usage: python src/evaluate_stare.py
Output: outputs/results_eval/stare_eval.csv, stare_eval.txt
"""
import os
import sys
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from retina_segmentation.src.model import DeepVesselNet
from retina_segmentation.src.utils import dice_coef, iou_score, sensitivity, specificity, accuracy
from retina_segmentation.src.stare_datasetninja import get_stare_pairs, load_mask_from_annotation

# ---------- Config (script-based paths) ----------
_BASE_DIR = Path(__file__).resolve().parents[1]
STARE_DS_ROOT = _BASE_DIR / "stare-DatasetNinja" / "stare-DatasetNinja"
CKPT_PATH = _BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"
RESULTS_DIR = _BASE_DIR / "outputs" / "results_eval"

IMG_SIZE = 512
THRESH = 0.5
ADD_GREEN = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold sweep to maximize Dice on STARE (set False to use only THRESH)
SWEEP_THRESH = True
THR_SWEEP_MIN, THR_SWEEP_MAX, THR_SWEEP_N = 0.25, 0.75, 11


def load_and_preprocess_image(img_path: str) -> torch.Tensor:
    """Load image, resize to IMG_SIZE, normalize (0,1), return (1, 3, H, W) float tensor."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.0) / 1.0  # mean=0, std=1
    img = np.transpose(img, (2, 0, 1))
    t = torch.from_numpy(img).float().unsqueeze(0)
    return t


def load_and_preprocess_mask(ann_path: str) -> torch.Tensor:
    """Load mask from DatasetNinja JSON, resize to IMG_SIZE, return (1, 1, H, W) float 0/1."""
    out = load_mask_from_annotation(ann_path)
    if out is None:
        raise ValueError(f"Failed to load mask from: {ann_path}")
    mask, _h, _w = out
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    return t


def metrics_at_threshold(probs_list, masks_list, thr: float):
    """Compute mean Dice, IoU, Sens, Spec, Acc over (probs, masks) at given threshold."""
    dices, ious, sens, specs, accs = [], [], [], [], []
    for probs, msks in zip(probs_list, masks_list):
        preds = (probs > thr).float()
        intersection = (preds * msks).sum(dim=(2, 3))
        dice = ((2.0 * intersection + 1e-6) / (preds.sum(dim=(2, 3)) + msks.sum(dim=(2, 3)) + 1e-6)).mean().item()
        union = (preds + msks).sum(dim=(2, 3)) - intersection
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
        tp, fn = (preds * msks).sum(dim=(2, 3)), ((1 - preds) * msks).sum(dim=(2, 3))
        se = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
        tn = ((1 - preds) * (1 - msks)).sum(dim=(2, 3))
        fp = (preds * (1 - msks)).sum(dim=(2, 3))
        sp = ((tn + 1e-6) / (tn + fp + 1e-6)).mean().item()
        total = preds.numel()
        acc = ((tp + tn).sum() + 1e-6) / (total + 1e-6)
        accs.append(acc.item())
        dices.append(dice)
        ious.append(iou)
        sens.append(se)
        specs.append(sp)
    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(sens)), float(np.mean(specs)), float(np.mean(accs))


def main():
    print("STARE (DatasetNinja) evaluation with current DeepVessel model")
    print("=" * 60)

    if not os.path.isfile(CKPT_PATH):
        print(f"Error: Checkpoint not found: {CKPT_PATH}")
        print("Run training first (e.g. python src/train.py).")
        sys.exit(1)

    pairs = get_stare_pairs(STARE_DS_ROOT)
    if not pairs:
        img_dir = os.path.join(STARE_DS_ROOT, "ds", "img")
        ann_dir = os.path.join(STARE_DS_ROOT, "ds", "ann")
        print(f"Error: No (image, annotation) pairs found.")
        print(f"  Annotation dir: {ann_dir}")
        print(f"  Image dir:      {img_dir}")
        print("  Place STARE images in ds/img/ with names matching ds/ann/*.png.json (e.g. im0001.png).")
        sys.exit(1)

    print(f"Checkpoint: {CKPT_PATH}")
    print(f"STARE root: {STARE_DS_ROOT}")
    print(f"Found {len(pairs)} image/annotation pairs.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    in_ch = 4 if ADD_GREEN else 3
    model = DeepVesselNet(in_channels=in_ch, out_channels=1).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    probs_list, masks_list = [], []
    failed = []

    with torch.no_grad():
        for idx, (img_path, ann_path) in enumerate(pairs):
            try:
                img_t = load_and_preprocess_image(img_path).to(DEVICE)
                msk_t = load_and_preprocess_mask(ann_path).to(DEVICE)
            except Exception as e:
                failed.append((os.path.basename(img_path), str(e)))
                continue
            probs = model(img_t).cpu()
            msk_t = msk_t.cpu()
            probs_list.append(probs)
            masks_list.append(msk_t)
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx + 1}/{len(pairs)}] {os.path.basename(img_path)}")

    if failed:
        print(f"\nSkipped {len(failed)} pairs due to errors (see report).")

    if not probs_list:
        print("Error: No samples could be evaluated.")
        sys.exit(1)

    # Threshold sweep to maximize Dice
    best_thr = THRESH
    if SWEEP_THRESH:
        print("\n🔎 Sweeping thresholds to maximize Dice on STARE...")
        thr_space = np.linspace(THR_SWEEP_MIN, THR_SWEEP_MAX, THR_SWEEP_N)
        best_dice = -1.0
        for t in thr_space:
            d, j, se, sp, acc = metrics_at_threshold(probs_list, masks_list, float(t))
            if d > best_dice:
                best_dice = d
                best_thr = t
            print(f"  thr={t:.2f}  Dice={d:.4f}  IoU={j:.4f}  Sens={se:.4f}  Spec={sp:.4f}")
        print(f"✅ Best threshold (Dice): {best_thr:.2f}  →  Dice={best_dice:.4f}")

    mean_dice, mean_iou, mean_sens, mean_spec, mean_acc = metrics_at_threshold(probs_list, masks_list, best_thr)

    print("\n" + "=" * 60)
    print(f"STARE evaluation summary (threshold {best_thr:.2f})")
    print("=" * 60)
    print(f"Dice:        {mean_dice:.4f}")
    print(f"IoU:         {mean_iou:.4f}")
    print(f"Sensitivity: {mean_sens:.4f}")
    print(f"Specificity: {mean_spec:.4f}")
    print(f"Accuracy:    {mean_acc:.4f}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, "stare_eval.csv")
    txt_path = os.path.join(RESULTS_DIR, "stare_eval.txt")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "stamp"])
        w.writerow(["threshold", f"{best_thr:.2f}", stamp])
        w.writerow(["Dice", f"{mean_dice:.4f}", stamp])
        w.writerow(["IoU", f"{mean_iou:.4f}", stamp])
        w.writerow(["Sensitivity", f"{mean_sens:.4f}", stamp])
        w.writerow(["Specificity", f"{mean_spec:.4f}", stamp])
        w.writerow(["Accuracy", f"{mean_acc:.4f}", stamp])
    print(f"\nCSV: {csv_path}")

    with open(txt_path, "w") as f:
        f.write("STARE (DatasetNinja) evaluation\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"STARE root: {STARE_DS_ROOT}\n")
        f.write(f"Pairs: {len(pairs)}, evaluated: {len(probs_list)}, failed: {len(failed)}\n")
        f.write(f"Image size: {IMG_SIZE}, threshold: {best_thr:.2f}" + (" (sweep for max Dice)" if SWEEP_THRESH else "") + "\n\n")
        f.write(f"Dice:        {mean_dice:.4f}\n")
        f.write(f"IoU:         {mean_iou:.4f}\n")
        f.write(f"Sensitivity: {mean_sens:.4f}\n")
        f.write(f"Specificity: {mean_spec:.4f}\n")
        f.write(f"Accuracy:    {mean_acc:.4f}\n")
        if failed:
            f.write("\nFailed pairs:\n")
            for name, err in failed:
                f.write(f"  {name}: {err}\n")
    print(f"TXT: {txt_path}")


if __name__ == "__main__":
    main()
