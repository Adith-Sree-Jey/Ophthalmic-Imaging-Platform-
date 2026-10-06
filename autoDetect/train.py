"""
train.py

Trains a MobileNetV3-Small image-type classifier.
Transfer learning from ImageNet weights — fast and accurate on small datasets.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR  = Path(__file__).resolve().parent / "dataset"
OUTPUT_DIR   = Path(__file__).resolve().parent / "model"
CLASSES      = ["anterior_segment", "red_glow", "slit_lamp"]

DEFAULT_EPOCHS     = 25
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 1e-3
IMG_SIZE           = 224
# ---------------------------------------------------------------------------


def get_transforms(is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main(epochs: int, batch_size: int, lr: float) -> None:
    print("=" * 60)
    print("  Ophthalmic Imaging - Image Type Classifier Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print(f"  Epochs : {epochs}")
    print(f"  Batch  : {batch_size}")
    print(f"  LR     : {lr}")

    # Datasets
    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", get_transforms(True))
    val_dataset   = datasets.ImageFolder(DATASET_DIR / "val",   get_transforms(False))

    print(f"\n  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}")
    print(f"  Classes       : {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = build_model(num_classes=len(train_dataset.classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters    : {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history = []

    print("\n" + "-" * 60)
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        marker = " ✅" if val_acc > best_val_acc else ""
        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
            f"{val_loss:>8.4f}  {val_acc:>6.1%}  {elapsed:>5.1f}s{marker}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    print("-" * 60)
    print(f"\n  Best Val Accuracy : {best_val_acc:.1%}")
    print(f"  Model saved       → {OUTPUT_DIR / 'best_model.pth'}")

    # Save class map so inference knows the index→label mapping
    class_map = {str(i): cls for i, cls in enumerate(train_dataset.classes)}
    with open(OUTPUT_DIR / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"  Class map saved   → {OUTPUT_DIR / 'class_map.json'}")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr)
