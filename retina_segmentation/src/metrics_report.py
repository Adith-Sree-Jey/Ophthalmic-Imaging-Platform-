import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from torch.utils.data import DataLoader

from retina_segmentation.src.data_loader import RetinaDataset
from retina_segmentation.src.model import DeepVesselNet
from retina_segmentation.src.utils import dice_coef, iou_score, sensitivity, specificity, accuracy

# ------------ Config ------------
BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "Data" / "test" / "image"
MSK_DIR = BASE_DIR / "Data" / "test" / "mask"
CKPT = BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"
OUT_DIR = BASE_DIR / "outputs" / "results"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 1
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ Data ------------
test_ds = RetinaDataset(IMG_DIR, MSK_DIR, augment=False, img_size=IMG_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------ Model ------------
model = DeepVesselNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ------------ Collect per-image + global metrics ------------
per_image_rows = []
all_probs = []
all_truth = []

with torch.no_grad():
    for idx, (img, mask) in enumerate(test_loader):
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        probs = model(img)
        preds = (probs > 0.5).float()

        # Per-image metrics (utils threshold internally, so pass the model probabilities directly)
        d = dice_coef(probs, mask)
        j = iou_score(probs, mask)
        se = sensitivity(probs, mask)
        sp = specificity(probs, mask)
        acc = accuracy(probs, mask)

        per_image_rows.append({
            "index": idx,
            "dice": float(d),
            "iou": float(j),
            "sensitivity": float(se),
            "specificity": float(sp),
            "accuracy": float(acc),
        })

        # For PR/ROC: collect flattened probabilities and labels
        all_probs.append(probs.detach().cpu().numpy().ravel())
        all_truth.append(mask.detach().cpu().numpy().ravel())

# Stack
all_probs = np.concatenate(all_probs, axis=0)
all_truth = np.concatenate(all_truth, axis=0)

# PR / ROC
prec, rec, _ = precision_recall_curve(all_truth, all_probs)
fpr, tpr, _ = roc_curve(all_truth, all_probs)
aupr = auc(rec, prec)
auroc = auc(fpr, tpr)

# ------------ Save CSV + JSON ------------
csv_path = OUT_DIR / "metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "dice", "iou", "sensitivity", "specificity", "accuracy"])
    writer.writeheader()
    for row in per_image_rows:
        writer.writerow(row)

summary = {
    "num_images": len(per_image_rows),
    "mean_dice": float(np.mean([r["dice"] for r in per_image_rows])),
    "mean_iou": float(np.mean([r["iou"] for r in per_image_rows])),
    "mean_sensitivity": float(np.mean([r["sensitivity"] for r in per_image_rows])),
    "mean_specificity": float(np.mean([r["specificity"] for r in per_image_rows])),
    "mean_accuracy": float(np.mean([r["accuracy"] for r in per_image_rows])),
    "aupr": float(aupr),
    "auroc": float(auroc),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary")
for k, v in summary.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Save PR/ROC arrays for plotting (paper_figs.py will read them)
np.save(OUT_DIR / "pr_prec.npy", prec)
np.save(OUT_DIR / "pr_rec.npy", rec)
np.save(OUT_DIR / "roc_fpr.npy", fpr)
np.save(OUT_DIR / "roc_tpr.npy", tpr)

print(f"\nSaved:\n- {csv_path}\n- {OUT_DIR / 'metrics.json'}\n- PR/ROC arrays for plotting")
