import sys
import os
from pathlib import Path
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from retina_segmentation.src.model import DeepVesselNet
from retina_segmentation.src.data_loader import get_train_val_dataloaders
from retina_segmentation.src.loss import FocalTverskyLoss
from retina_segmentation.src.utils import dice_coef, iou_score


# -----------------------------
# CONFIG (edit as needed)
# -----------------------------
BASE_DIR      = Path(__file__).resolve().parents[1]
TRAIN_IMG_DIR = BASE_DIR / "Data" / "train" / "image"
TRAIN_MSK_DIR = BASE_DIR / "Data" / "train" / "mask"
VAL_FRACTION  = 0.20
SPLIT_SEED    = 42

IMG_SIZE     = 512         # resize input (H=W)
RANDOM_CROP  = None        # e.g., 384 for patch training; None to disable
ADD_GREEN    = False       # True => 4-channel input (RGB+G); requires in_channels=4

EPOCHS       = 50
BATCH_SIZE   = 2
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0           # keep 0 on Windows unless you know what you're doing
PIN_MEMORY   = True

OUTPUT_DIR   = BASE_DIR / "outputs"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
LOGS_DIR     = OUTPUT_DIR / "logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# -----------------------------
# UTILS
# -----------------------------
def auto_cleanup_outputs():
    """
    Clean non-checkpoint outputs so runs don't mix results.
    Keeps checkpoints folder intact.
    """
    cleanup_path = BASE_DIR / "outputs"
    assert "retina_segmentation" in str(cleanup_path)

    keep = {"checkpoints"}
    if not cleanup_path.exists():
        cleanup_path.mkdir(parents=True, exist_ok=True)
        return
    for path in cleanup_path.iterdir():
        name = path.name
        if name not in keep:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
    os.makedirs(CKPT_DIR, exist_ok=True)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dices, ious = [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item()

        dices.append(dice_coef(logits, masks))
        ious.append(iou_score(logits, masks))

    avg_loss = running_loss / max(1, len(loader))
    avg_dice = sum(dices) / max(1, len(dices))
    avg_iou  = sum(ious)  / max(1, len(ious))
    return avg_loss, avg_dice, avg_iou


def main():
    # ---------- prep ----------
    auto_cleanup_outputs()
    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_dir = os.path.join(LOGS_DIR, run_tag)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # data
    train_loader, val_loader = get_train_val_dataloaders(
        TRAIN_IMG_DIR, TRAIN_MSK_DIR,
        val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        random_crop=RANDOM_CROP,
        add_green=ADD_GREEN,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    print(
        f"Train/validation samples: {len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)} (seed={SPLIT_SEED}); Data/test is not loaded."
    )

    # model
    in_ch = 4 if ADD_GREEN else 3
    model = DeepVesselNet(in_channels=in_ch, out_channels=1).to(DEVICE)

    # loss/optim/sched
    criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.33)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = amp.GradScaler() if DEVICE.type == "cuda" else None

    # tracking
    best_val_dice = -1.0
    last_ckpt_path = os.path.join(CKPT_DIR, "last_model.pth")
    best_ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")

    # ---------- loop ----------
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        # log
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val",   val_loss,   epoch)
        writer.add_scalar("Metrics/Dice", val_dice, epoch)
        writer.add_scalar("Metrics/IoU",  val_iou,  epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        # save "last"
        torch.save(model.state_dict(), last_ckpt_path)

        # save "best" by Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ Saved best model → {best_ckpt_path} (Dice={best_val_dice :.4f})")

        # optional per-epoch snapshot
        epoch_ckpt = os.path.join(CKPT_DIR, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_ckpt)

    writer.close()
    print("\nTraining complete.")
    print(f"Best Dice: {best_val_dice:.4f}")    
    print(f"TensorBoard logs → {tb_dir}")
    print(f"Checkpoints      → {CKPT_DIR}")


if __name__ == "__main__":
    main()
