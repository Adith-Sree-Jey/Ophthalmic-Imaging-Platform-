import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from retina_segmentation.src.model import DeepVesselNet
from retina_segmentation.src.data_loader import RetinaDataset
from retina_segmentation.src.utils import clean_prediction

# ------------ Config ------------
BASE_DIR = Path(__file__).resolve().parents[1]
TEST_IMG_DIR = BASE_DIR / "Data" / "test" / "image"
TEST_MSK_DIR = BASE_DIR / "Data" / "test" / "mask"
CKPT = BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"
RES_DIR = BASE_DIR / "outputs" / "results"
FIG_SAMPLES_DIR = RES_DIR / "fig_samples"
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_SAMPLES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

# ------------ Load model & data ------------
model = DeepVesselNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

test_ds = RetinaDataset(TEST_IMG_DIR, TEST_MSK_DIR, augment=False, img_size=IMG_SIZE)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# ------------ 1) Save triptychs for first N samples ------------
N = min(8, len(test_ds))  # pick up to 8
with torch.no_grad():
    for idx, (img, mask) in enumerate(test_loader):
        if idx >= N:
            break
        img = img.to(DEVICE); mask = mask.to(DEVICE)
        prob = model(img).squeeze().cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8) * 255
        pred = clean_prediction(pred)

        # prepare visuals
        rgb = (img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt  = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb); axes[0].set_title("Original");    axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Prediction");  axes[2].axis("off")
        plt.tight_layout()
        out_path = FIG_SAMPLES_DIR / f"sample_{idx}.png"
        plt.savefig(out_path, dpi=300); plt.close()

print(f"✅ Saved {N} sample triptychs to: {FIG_SAMPLES_DIR}")

# ------------ 2) Tiled grid of K samples ------------
# Load the just-saved triptychs and tile
def make_grid(image_paths, cols=4):
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]
    h, w, _ = imgs[0].shape
    rows = int(np.ceil(len(imgs) / cols))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w, :] = im
    return grid

sample_paths = [FIG_SAMPLES_DIR / f for f in sorted(os.listdir(FIG_SAMPLES_DIR)) if f.endswith(".png")]
if sample_paths:
    grid = make_grid(sample_paths, cols=4)
    plt.figure(figsize=(12, 8))
    plt.imshow(grid); plt.axis("off"); plt.tight_layout()
    grid_path = RES_DIR / "fig_grid.png"
    plt.savefig(grid_path, dpi=300); plt.close()
    print(f"✅ Saved tiled grid: {grid_path}")

# ------------ 3) PR & ROC curves ------------
pr_prec_path = RES_DIR / "pr_prec.npy"
pr_rec_path  = RES_DIR / "pr_rec.npy"
roc_fpr_path = RES_DIR / "roc_fpr.npy"
roc_tpr_path = RES_DIR / "roc_tpr.npy"

if all(os.path.exists(p) for p in [pr_prec_path, pr_rec_path, roc_fpr_path, roc_tpr_path]):
    prec = np.load(pr_prec_path); rec = np.load(pr_rec_path)
    fpr  = np.load(roc_fpr_path); tpr = np.load(roc_tpr_path)

    # PR
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve")
    plt.grid(True); plt.tight_layout()
    pr_path = RES_DIR / "pr_curve.png"
    plt.savefig(pr_path, dpi=300); plt.close()

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.grid(True); plt.tight_layout()
    roc_path = RES_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=300); plt.close()

    print(f"✅ Saved PR: {pr_path}\n✅ Saved ROC: {roc_path}")
else:
    print("ℹ️ Run `python metrics_report.py` first to generate PR/ROC arrays.")
