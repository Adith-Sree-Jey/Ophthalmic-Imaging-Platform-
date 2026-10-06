"""
evaluate.py

Evaluates the trained model on the held-out test set.
Shows overall accuracy, per-class accuracy, and confusion matrix.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
MODEL_DIR   = Path(__file__).resolve().parent / "model"
IMG_SIZE    = 224


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    print("=" * 60)
    print("  Ophthalmic Imaging - Image Type Classifier Evaluation")
    print("=" * 60)

    # Load class map
    class_map_path = MODEL_DIR / "class_map.json"
    if not class_map_path.exists():
        print("ERROR: class_map.json not found. Run train.py first.")
        return

    with open(class_map_path) as f:
        class_map: dict[str, str] = json.load(f)
    classes = [class_map[str(i)] for i in range(len(class_map))]
    num_classes = len(classes)
    print(f"\n  Classes : {classes}")

    # Dataset
    test_dataset = datasets.ImageFolder(DATASET_DIR / "test", get_transform())
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"  Test samples : {len(test_dataset)}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(num_classes).to(device)
    model.load_state_dict(
        torch.load(MODEL_DIR / "best_model.pth", map_location=device, weights_only=True)
    )
    model.eval()

    # Evaluate
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Overall accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = correct / len(all_labels)

    print(f"\n  Overall Accuracy : {overall_acc:.1%}  ({correct}/{len(all_labels)})")

    # Per-class accuracy
    print("\n  Per-class Accuracy:")
    print(f"  {'Class':<22} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print("  " + "-" * 48)
    for i, cls in enumerate(classes):
        cls_labels = [l for l in all_labels if l == i]
        cls_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == i and p == i)
        acc = cls_correct / len(cls_labels) if cls_labels else 0.0
        print(f"  {cls:<22} {cls_correct:>7} {len(cls_labels):>7} {acc:>8.1%}")

    # Confusion matrix
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(all_preds, all_labels):
        matrix[label][pred] += 1

    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = "  " + " " * 22 + "".join(f"{cls[:8]:>10}" for cls in classes)
    print(header)
    print("  " + "-" * (22 + 10 * num_classes))
    for i, cls in enumerate(classes):
        row = "  " + f"{cls:<22}" + "".join(f"{matrix[i][j]:>10}" for j in range(num_classes))
        print(row)

    print("\n" + "=" * 60)

    if overall_acc >= 0.97:
        print("  ✅ EXCELLENT — Ready to connect to the web backend.")
    elif overall_acc >= 0.93:
        print("  ✅ GOOD — Acceptable for production use.")
    elif overall_acc >= 0.88:
        print("  ⚠  FAIR — Consider training more epochs or adding data.")
    else:
        print("  ❌ LOW — Review training data and retrain.")

    print("=" * 60)


if __name__ == "__main__":
    main()
