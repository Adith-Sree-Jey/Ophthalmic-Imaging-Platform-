"""
Run baseline models (U-Net, ResUNet, R2U-Net) and optionally DeepVesselNet,
then write comparison metrics to a single table for research paper comparison.

Usage:
  python src/compare_baselines.py [--train] [--include-deepvessel]

  --train              Train each baseline if checkpoint missing (otherwise evaluate only).
  --include-deepvessel Also evaluate DeepVesselNet (best_model.pth) and add to table.

Output:
  - outputs/results_eval/baseline_comparison.csv
  - outputs/results_eval/baseline_comparison.txt
"""
import sys
import os
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from retina_segmentation.src.baseline_models import get_baseline_model
from retina_segmentation.src.data_loader import get_dataloaders
from retina_segmentation.src.loss import FocalTverskyLoss, DiceLoss
from retina_segmentation.src.utils import dice_coef, iou_score
from retina_segmentation.src.model import DeepVesselNet

# -----------------------------
# CONFIG (aligned with train.py / evaluate.py)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
TRAIN_IMG_DIR = DATA_DIR / "train" / "image"
TRAIN_MSK_DIR = DATA_DIR / "train" / "mask"
TEST_IMG_DIR = DATA_DIR / "test" / "image"
TEST_MSK_DIR = DATA_DIR / "test" / "mask"
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results_eval"

IMG_SIZE = 512
ADD_GREEN = False
IN_CH = 4 if ADD_GREEN else 3
EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
THRESH = 0.5
GRAD_CLIP = 1.0  # gradient clipping to avoid collapse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_NAMES = ["UNet", "ResUNet", "R2UNet"]


def ckpt_path(name):
    """Checkpoint path for a baseline (e.g. baseline_unet.pth)."""
    return CKPT_DIR / f"baseline_{name.lower()}.pth"


def train_baseline(name, train_loader, val_loader):
    """Train one baseline model; save to baseline_{name}.pth.
    Uses Focal Tversky (alpha=0.6 to penalize FP) + Dice loss to reduce 'all vessel' collapse."""
    if len(train_loader) == 0:
        raise ValueError(f"Train loader is empty; cannot train {name}.")
    if len(val_loader) == 0:
        raise ValueError(f"Val/test loader is empty; cannot validate {name}.")
    model = get_baseline_model(name, in_channels=IN_CH, out_channels=1).to(DEVICE)
    criterion_ft = FocalTverskyLoss(alpha=0.6, beta=0.4, gamma=1.33)  # higher alpha = more FP penalty
    criterion_dice = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_dice = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"{name} train", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss_ft = criterion_ft(logits, masks)
                    loss_dice = criterion_dice(logits, masks)
                    loss = 0.5 * loss_ft + 0.5 * loss_dice
                scaler.scale(loss).backward()
                if GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss_ft = criterion_ft(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = 0.5 * loss_ft + 0.5 * loss_dice
                loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                out = model(imgs)
                dices.append(dice_coef(out, msks))
                ious.append(iou_score(out, msks))
        val_dice = float(np.mean(dices))
        val_iou = float(np.mean(ious))
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_path(name))
            print(f"  [{name}] Epoch {epoch}/{EPOCHS} — loss: {train_loss:.4f} — val Dice: {val_dice:.4f} IoU: {val_iou:.4f} (saved)")
        else:
            print(f"  [{name}] Epoch {epoch}/{EPOCHS} — loss: {train_loss:.4f} — val Dice: {val_dice:.4f} IoU: {val_iou:.4f}")
    print(f"  [{name}] Training done. Best Dice: {best_dice:.4f}")
    return best_dice


@torch.no_grad()
def evaluate_model(model, loader, thr=0.5):
    """Return dict with Dice, IoU, Sensitivity, Specificity, Accuracy (at thr)."""
    model.eval()
    dices, ious, sens, specs, accs = [], [], [], [], []
    for imgs, msks in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        probs = model(imgs)
        preds = (probs > thr).float()

        inter = (preds * msks).sum(dim=(2, 3))
        dice = ((2.0 * inter + 1e-6) / (preds.sum(dim=(2, 3)) + msks.sum(dim=(2, 3)) + 1e-6)).mean().item()
        union = (preds + msks).sum(dim=(2, 3)) - inter
        iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
        tp = (preds * msks).sum(dim=(2, 3))
        fn = ((1 - preds) * msks).sum(dim=(2, 3))
        se = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
        tn = ((1 - preds) * (1 - msks)).sum(dim=(2, 3))
        fp = (preds * (1 - msks)).sum(dim=(2, 3))
        sp = ((tn + 1e-6) / (tn + fp + 1e-6)).mean().item()
        total = preds.shape[0] * preds.shape[2] * preds.shape[3]
        acc = ((tp + tn).sum() + 1e-6) / (total + 1e-6)
        accs.append(acc.item())

        dices.append(dice)
        ious.append(iou)
        sens.append(se)
        specs.append(sp)
    return {
        "Dice": float(np.mean(dices)),
        "IoU": float(np.mean(ious)),
        "Sensitivity": float(np.mean(sens)),
        "Specificity": float(np.mean(specs)),
        "Accuracy": float(np.mean(accs)),
    }


def _load_checkpoint(path):
    """Load state_dict from path; handle both raw state_dict and dict with 'state_dict' key."""
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def run_comparison(do_train=False, include_deepvessel=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    try:
        train_loader, test_loader = get_dataloaders(
            TRAIN_IMG_DIR, TRAIN_MSK_DIR,
            TEST_IMG_DIR, TEST_MSK_DIR,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            random_crop=None,
            add_green=ADD_GREEN,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    except FileNotFoundError as e:
        print(f"Error: Data not found. {e}")
        print("Ensure Data/train/image, Data/train/mask, Data/test/image, Data/test/mask exist with matching images.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    results = {}

    if do_train:
        print("Tip: If you previously had 'all same metrics', delete outputs/checkpoints/baseline_*.pth first for fresh training.\n")

    # Baselines: train only when --train and checkpoint missing; then evaluate
    for name in BASELINE_NAMES:
        path = ckpt_path(name)
        if do_train and not os.path.exists(path):
            if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(TRAIN_MSK_DIR):
                print(f"Data not found; skipping training for {name}. Place data in Data/train, Data/test.")
                continue
            print(f"Training {name}...")
            try:
                train_baseline(name, train_loader, test_loader)
            except Exception as e:
                print(f"  [{name}] Training failed: {e}")
                continue
        if not os.path.exists(path):
            print(f"  No checkpoint for {name}; skipping. (Run with --train to train baselines.)")
            continue
        model = get_baseline_model(name, in_channels=IN_CH, out_channels=1).to(DEVICE)
        try:
            state = _load_checkpoint(path)
            model.load_state_dict(state, strict=True)
        except FileNotFoundError:
            print(f"  [{name}] Checkpoint not found: {path}")
            continue
        except Exception as e:
            print(f"  [{name}] Load failed (wrong checkpoint?): {e}")
            continue
        results[name] = evaluate_model(model, test_loader, thr=THRESH)
        print(f"  {name}: Dice={results[name]['Dice']:.4f} IoU={results[name]['IoU']:.4f} Acc={results[name]['Accuracy']:.4f}")

    # Optional: DeepVesselNet
    if include_deepvessel:
        dv_path = CKPT_DIR / "best_model.pth"
        if os.path.exists(dv_path):
            model = DeepVesselNet(in_channels=IN_CH, out_channels=1).to(DEVICE)
            try:
                state = _load_checkpoint(dv_path)
                model.load_state_dict(state, strict=True)
            except Exception as e:
                print(f"  DeepVesselNet: Load failed: {e}")
            else:
                results["DeepVesselNet"] = evaluate_model(model, test_loader, thr=THRESH)
                print(f"  DeepVesselNet: Dice={results['DeepVesselNet']['Dice']:.4f} IoU={results['DeepVesselNet']['IoU']:.4f} Acc={results['DeepVesselNet']['Accuracy']:.4f}")
        else:
            print("  DeepVesselNet: best_model.pth not found; skip.")

    if not results:
        print("No results to write.")
        return None

    # Warn if all metrics are identical (same checkpoint or all models collapsed to same prediction)
    all_identical = False
    if len(results) > 1:
        first = next(iter(results.values()))
        if all(
            all(m[k] == first[k] for k in first)
            for m in results.values()
        ):
            all_identical = True
            print("\n⚠️  WARNING: All models have identical metrics. This usually means either:")
            print("    (1) The same checkpoint was used for different models (e.g. best_model.pth copied to baseline_*.pth), or")
            print("    (2) All models collapsed to 'predict all vessel' (Sensitivity=1, Specificity=0).")
            print("    Fix: Train each baseline separately with --train, and use separate checkpoint files.")

    # Print comparison table to console
    sep = "-" * 72
    print("\n" + sep)
    print("Baseline comparison (retinal vessel segmentation)")
    print(f"Dataset: Data/train, Data/test | img_size={IMG_SIZE} | threshold={THRESH}")
    print(sep)
    print(f"{'Model':<16} {'Dice':>8} {'IoU':>8} {'Sensitivity':>12} {'Specificity':>12} {'Accuracy':>10}")
    print(sep)
    for model_name, metrics in results.items():
        print(f"{model_name:<16} {metrics['Dice']:>8.4f} {metrics['IoU']:>8.4f} "
              f"{metrics['Sensitivity']:>12.4f} {metrics['Specificity']:>12.4f} {metrics['Accuracy']:>10.4f}")
    print(sep + "\n")

    # Write CSV
    fieldnames = ["Model", "Dice", "IoU", "Sensitivity", "Specificity", "Accuracy"]
    csv_path = RESULTS_DIR / "baseline_comparison.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for model_name, metrics in results.items():
                row = {"Model": model_name, **metrics}
                w.writerow(row)
        print(f"\nWrote {csv_path}")
    except OSError as e:
        print(f"\nError writing CSV: {e}")

    # Write TXT (human-readable table)
    txt_path = RESULTS_DIR / "baseline_comparison.txt"
    try:
        with open(txt_path, "w") as f:
            f.write("Baseline comparison (retinal vessel segmentation)\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: Data/train, Data/test | img_size={IMG_SIZE} | threshold={THRESH}\n")
            f.write("-" * 72 + "\n")
            f.write(f"{'Model':<16} {'Dice':>8} {'IoU':>8} {'Sensitivity':>12} {'Specificity':>12} {'Accuracy':>10}\n")
            f.write("-" * 72 + "\n")
            for model_name, metrics in results.items():
                f.write(f"{model_name:<16} {metrics['Dice']:>8.4f} {metrics['IoU']:>8.4f} "
                        f"{metrics['Sensitivity']:>12.4f} {metrics['Specificity']:>12.4f} {metrics['Accuracy']:>10.4f}\n")
            f.write("-" * 72 + "\n")
            if all_identical:
                f.write("\n⚠️ All metrics identical - check that each model uses its own trained checkpoint.\n")
        print(f"Wrote {txt_path}")
    except OSError as e:
        print(f"Error writing TXT: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare baseline models (U-Net, ResUNet, R2U-Net) and optionally DeepVesselNet.")
    parser.add_argument("--train", action="store_true", help="Train each baseline if checkpoint missing")
    parser.add_argument("--include-deepvessel", action="store_true", help="Include DeepVesselNet (best_model.pth) in comparison")
    args = parser.parse_args()
    try:
        run_comparison(do_train=args.train, include_deepvessel=args.include_deepvessel)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
