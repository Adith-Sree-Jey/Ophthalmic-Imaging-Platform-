import sys
import os
import json
from pathlib import Path

import shutil
from datetime import datetime


import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from retina_segmentation.src.model import DeepVesselNet
from retina_segmentation.src.data_loader import RetinaDataset
from retina_segmentation.src.utils import (
    dice_coef, iou_score, sensitivity, specificity, accuracy,
    clean_prediction, plot_results
)

# =========================
# CONFIG (edit as needed)
# =========================
# Use script location so best_model.pth and outputs are always this project's
BASE_DIR      = Path(__file__).resolve().parents[1]
TEST_IMG_DIR  = BASE_DIR / "Data" / "test" / "image"
TEST_MSK_DIR  = BASE_DIR / "Data" / "test" / "mask"
CKPT_PATH     = BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"

IMG_SIZE      = 512
ADD_GREEN     = False         # True only if the model was trained with 4-channel (RGB+G)

# Thresholding
_THRESHOLD_RAW = os.getenv("RETINA_EVAL_THRESHOLD")
if _THRESHOLD_RAW is None:
    raise RuntimeError(
        "RETINA_EVAL_THRESHOLD is required. Freeze a threshold selected on "
        "validation data before running held-out test evaluation."
    )
THRESH = float(_THRESHOLD_RAW)
if not 0.0 <= THRESH <= 1.0:
    raise ValueError("RETINA_EVAL_THRESHOLD must be between 0 and 1.")
SWEEP_THRESH  = False         # never select a threshold on the held-out test set
TARGET_SENSITIVITY = 0.99     # historical analysis target; disabled for final test evaluation

# Morphology (post-processing)
MORPH_MODE    = "close"       # "none" | "close" | "openclose" | "utils_clean" (uses src.utils.clean_prediction)
MORPH_KERNEL  = 2             # 2 or 3 recommended; larger may erase tiny vessels

# Visual outputs
SAVE_FIRST_N  = 6             # number of side-by-side visuals to save
SAVE_PROB_MAPS = True         # dump probability heatmaps for the first few samples

# Output folders
OUTPUT_DIR    = BASE_DIR / "outputs"
PRED_DIR      = OUTPUT_DIR / "predictions"
VIS_DIR       = OUTPUT_DIR / "visuals"
RESULTS_DIR   = OUTPUT_DIR / "results_eval"  # per-run evaluation summaries

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# HELPERS
# =========================
def auto_cleanup_outputs():
    """Clean predictions/visuals folders before a new evaluation run."""
    for p in [PRED_DIR, VIS_DIR]:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def binarize_from_output(model_output: torch.Tensor, thr: float) -> np.ndarray:
    """Model probabilities -> binary uint8 mask (0 or 255)."""
    probs = model_output.squeeze().detach().cpu().numpy()
    mask = (probs > thr).astype(np.uint8) * 255
    return mask, probs


def apply_morphology(mask_u8: np.ndarray) -> np.ndarray:
    if MORPH_MODE == "none":
        return mask_u8
    if MORPH_MODE == "utils_clean":
        return clean_prediction(mask_u8)
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    if MORPH_MODE == "close":
        return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if MORPH_MODE == "openclose":
        m = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    # fallback
    return mask_u8


@torch.no_grad()
def evaluate_at_threshold(model, loader, thr: float):
    """Return Dice, IoU, Sens, Spec, Accuracy averaged over loader at a particular threshold."""
    dices, ious, sens, specs, accs = [], [], [], [], []
    for imgs, msks in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        probs = model(imgs)
        preds_thr = (probs > thr).float()

        # inline metrics honoring 'thr'
        intersection = (preds_thr * msks).sum(dim=(2, 3))
        dice = ((2. * intersection + 1e-6) /
                (preds_thr.sum(dim=(2, 3)) + msks.sum(dim=(2, 3)) + 1e-6)).mean().item()
        union = (preds_thr + msks).sum(dim=(2, 3)) - intersection
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
        tp = (preds_thr * msks).sum(dim=(2, 3))
        fn = ((1 - preds_thr) * msks).sum(dim=(2, 3))
        se = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
        tn = ((1 - preds_thr) * (1 - msks)).sum(dim=(2, 3))
        fp = (preds_thr * (1 - msks)).sum(dim=(2, 3))
        sp = ((tn + 1e-6) / (tn + fp + 1e-6)).mean().item()
        total = preds_thr.shape[0] * preds_thr.shape[2] * preds_thr.shape[3]
        acc = ((tp + tn).sum() + 1e-6) / (total + 1e-6)
        accs.append(acc.item())

        dices.append(dice); ious.append(iou); sens.append(se); specs.append(sp)
    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(sens)), float(np.mean(specs)), float(np.mean(accs))


# =========================
# MAIN
# =========================
def main():
    print("➡️  Starting evaluation...")
    auto_cleanup_outputs()

    # sanity: checkpoint path & size
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    print(f"Using checkpoint: {CKPT_PATH} (size: {os.path.getsize(CKPT_PATH)/1e6:.2f} MB)")

    # Held-out test loader only. No train/validation directory is read here.
    test_dataset = RetinaDataset(
        image_dir=TEST_IMG_DIR,
        mask_dir=TEST_MSK_DIR,
        augment=False,
        img_size=IMG_SIZE,
        random_crop=None,
        add_green=ADD_GREEN,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # model
    in_ch = 4 if ADD_GREEN else 3
    model = DeepVesselNet(in_channels=in_ch, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    # optionally sweep threshold to maximize Dice and to reach target sensitivity
    best_thr = THRESH
    best_thr_dice = None
    thr_for_sens = None
    sens_for_099 = None
    if SWEEP_THRESH:
        raise RuntimeError(
            "Threshold sweeping on Data/test is disabled. Select the threshold "
            "using validation data and pass it through RETINA_EVAL_THRESHOLD."
        )
        print("🔎 Sweeping thresholds to maximize Dice...")
        thr_space = np.linspace(0.30, 0.70, 9)
        scores = []
        for t in thr_space:
            d, j, se, sp, acc = evaluate_at_threshold(model, test_loader, t)
            scores.append((t, d, j, se, sp, acc))
            print(f"  thr={t:.2f}  Dice={d:.4f}  IoU={j:.4f}  Sens={se:.4f}  Spec={sp:.4f}  Acc={acc:.4f}")
        best_thr, best_thr_dice = max(scores, key=lambda x: x[1])[0], max(scores, key=lambda x: x[1])[1]
        print(f"✅ Best threshold (Dice): {best_thr:.2f} (Dice={best_thr_dice:.4f})")

        # Find threshold for target sensitivity (lower thr => higher sensitivity)
        print(f"\n🔎 Finding threshold for sensitivity >= {TARGET_SENSITIVITY:.2f}...")
        thr_sens_space = np.linspace(0.15, 0.55, 17)
        all_sens = []
        for t in thr_sens_space:
            d, j, se, sp, acc = evaluate_at_threshold(model, test_loader, t)
            all_sens.append((t, d, j, se, sp, acc))
        candidates = [x for x in all_sens if x[3] >= TARGET_SENSITIVITY]
        if candidates:
            # Pick the one with highest specificity (or Dice) among those meeting sensitivity
            thr_for_sens, d, j, se, sp, acc = max(candidates, key=lambda x: (x[4], x[1]))  # spec, then dice
            sens_for_099 = se
            print(f"  thr={thr_for_sens:.2f}  →  Sens={se:.4f}  Spec={sp:.4f}  Dice={d:.4f}")
            print(f"✅ Threshold for Sens>={TARGET_SENSITIVITY:.2f}: {thr_for_sens:.2f} (Sens={se:.4f}, Spec={sp:.4f})")
        else:
            best = max(all_sens, key=lambda x: x[3])
            thr_for_sens, sens_for_099 = best[0], best[3]
            print(f"  No threshold reached {TARGET_SENSITIVITY:.2f}. Best sensitivity: {sens_for_099:.4f} at thr={thr_for_sens:.2f}")
            print(f"  Tip: Retrain with sensitivity-focused loss (train.py uses beta=0.7), then re-evaluate.")

    # Aggregate metrics at the frozen validation-selected threshold.
    dices, ious, sens, specs, accs = [], [], [], [], []
    saved_vis = 0

    with torch.no_grad():
        for idx, (imgs, msks) in enumerate(test_loader):
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)

            probs = model(imgs)
            # Metric helpers apply the same frozen threshold used for masks.
            dices.append(dice_coef(probs, msks, threshold=THRESH))
            ious.append(iou_score(probs, msks, threshold=THRESH))
            sens.append(sensitivity(probs, msks, threshold=THRESH))
            specs.append(specificity(probs, msks, threshold=THRESH))
            accs.append(accuracy(probs, msks, threshold=THRESH))

            # binarize with chosen threshold for SAVING predictions
            pred_u8, prob_map = binarize_from_output(probs, best_thr)

            # debug: prob stats + optional prob map dump
            print(f"[{idx}] prob min={prob_map.min():.4f} max={prob_map.max():.4f} mean={prob_map.mean():.4f}")
            if SAVE_PROB_MAPS and idx < SAVE_FIRST_N:
                cv2.imwrite(str(PRED_DIR / f"prob_map_{idx}.png"),
                            (prob_map * 255).astype(np.uint8))

            # post-process (gentle by default)
            if MORPH_MODE == "utils_clean":
                pred_clean = clean_prediction(pred_u8)
            else:
                pred_clean = apply_morphology(pred_u8)

            # save mask
            cv2.imwrite(str(PRED_DIR / f"pred_{idx}.png"), pred_clean)

            # save side-by-side visuals for first N images
            if saved_vis < SAVE_FIRST_N:
                rgb = (imgs.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt  = (msks.squeeze().cpu().numpy() * 255).astype(np.uint8)
                plot_results(rgb, gt, pred_clean, save_path=str(VIS_DIR / f"vis_{idx}.png"))
                saved_vis += 1

    # Aggregate held-out test metrics without changing the frozen threshold.
    mean_dice = float(np.mean(dices))
    mean_iou  = float(np.mean(ious))
    mean_se   = float(np.mean(sens))
    mean_sp   = float(np.mean(specs))
    mean_acc  = float(np.mean(accs))

    print(f"\n📊 Evaluation Summary (fixed threshold={THRESH:.2f}):")
    print(f"Dice Coefficient: {mean_dice:.4f}")
    print(f"IoU Score:        {mean_iou:.4f}")
    print(f"Sensitivity:      {mean_se:.4f}")
    print(f"Specificity:      {mean_sp:.4f}")
    print(f"Accuracy:         {mean_acc:.4f}")

    # save a small report
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_txt = RESULTS_DIR / f"eval_{stamp}.txt"
    with open(report_txt, "w") as f:
        f.write("Evaluation Summary\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"Checkpoint size: {os.path.getsize(CKPT_PATH)/1e6:.2f} MB\n")
        f.write(f"Image size: {IMG_SIZE}\n")
        f.write(f"ADD_GREEN:  {ADD_GREEN}\n")
        f.write(f"Morphology: {MORPH_MODE} (k={MORPH_KERNEL})\n")
        f.write(f"Threshold for saving predictions: {best_thr:.2f}\n")
        if best_thr_dice is not None:
            f.write(f"Best-threshold Dice: {best_thr_dice:.4f}\n")
        if thr_for_sens is not None and sens_for_099 is not None:
            f.write(f"Threshold for Sens>={TARGET_SENSITIVITY:.2f}: {thr_for_sens:.2f} (Sens={sens_for_099:.4f})\n")
        f.write(f"\nMetrics (fixed threshold={THRESH:.2f}):\n")
        f.write(f"Dice: {mean_dice:.4f}\n")
        f.write(f"IoU:  {mean_iou:.4f}\n")
        f.write(f"Sens: {mean_se:.4f}\n")
        f.write(f"Spec: {mean_sp:.4f}\n")
        f.write(f"Accuracy: {mean_acc:.4f}\n")

    report_json = RESULTS_DIR / f"eval_{stamp}.json"
    result_payload = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(CKPT_PATH),
        "test_image_dir": str(TEST_IMG_DIR),
        "test_mask_dir": str(TEST_MSK_DIR),
        "threshold": THRESH,
        "metrics": {
            "dice": mean_dice,
            "iou": mean_iou,
            "sensitivity": mean_se,
            "specificity": mean_sp,
            "accuracy": mean_acc,
        },
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print("\n📁 Outputs:")
    print(f"Predictions → {PRED_DIR}")
    print(f"Visuals     → {VIS_DIR}")
    print(f"Report      → {report_txt}")
    print(f"JSON results → {report_json}")
    if SWEEP_THRESH:
        print(f"Best-threshold used to save preds: {best_thr:.2f}")
    if thr_for_sens is not None and sens_for_099 is not None:
        print(f"For ~0.99 sensitivity use threshold: {thr_for_sens:.2f} (Sens={sens_for_099:.4f})")

if __name__ == "__main__":
    main()
