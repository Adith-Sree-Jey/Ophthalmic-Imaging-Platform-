# -*- coding: utf-8 -*-
"""
Glaucoma Classification — MobileNetV3-Small
Fixed version:
  - Stratified train/val split (no class imbalance in val)
  - Script-relative paths (no hardcoded Google Drive paths)
  - plt.show() removed from inference functions (UI handles display)
  - Early stopping with configurable patience
  - Eval runs on held-out val set only (no data leakage)
"""

import os
import copy
import time
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for UI / headless
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve,
)
import seaborn as sns

print("✓ All imports successful!")
print(f"  PyTorch version : {torch.__version__}")
print(f"  GPU available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device          : {torch.cuda.get_device_name(0)}")

# ─── PATHS ────────────────────────────────────────────────────────────────────
# All paths are relative to this script's location.
# Place your dataset in  <script_dir>/data/glaucoma/
# Checkpoints are saved to  <script_dir>/outputs/checkpoints/

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR / "data" / "glaucoma"
OUTPUTS_DIR  = SCRIPT_DIR / "outputs" / "checkpoints"
SAVE_PATH    = OUTPUTS_DIR / "glaucoma_mobilenetv3.pth"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n  Data dir    : {DATA_DIR}")
print(f"  Save path   : {SAVE_PATH}")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2       # 20 % validation — stratified
SEED         = 42

# Early stopping
PATIENCE     = 10        # stop if val acc doesn't improve for this many epochs
MIN_DELTA    = 1e-4      # minimum improvement to count as "better"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Training on : {DEVICE}\n")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─── DATASET WRAPPER ──────────────────────────────────────────────────────────
class TransformSubset(Dataset):
    """Applies a transform to a Subset of an ImageFolder dataset."""
    def __init__(self, subset: Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ─── LOAD DATASET ─────────────────────────────────────────────────────────────
raw_dataset = datasets.ImageFolder(str(DATA_DIR), transform=None)
CLASS_NAMES = raw_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print(f"  Classes found : {CLASS_NAMES}")
print(f"  Total images  : {len(raw_dataset)}")
label_counts = Counter(raw_dataset.targets)
for idx, count in label_counts.items():
    print(f"    {CLASS_NAMES[idx]}: {count} images")

# ─── STRATIFIED SPLIT ─────────────────────────────────────────────────────────
# StratifiedShuffleSplit guarantees the same class ratio in both train and val.
# The old random permutation approach did NOT guarantee this.
all_labels = np.array(raw_dataset.targets)
splitter   = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))

train_set = TransformSubset(Subset(raw_dataset, train_idx), train_transforms)
val_set   = TransformSubset(Subset(raw_dataset, val_idx),   val_transforms)

print(f"\n  Train: {len(train_set)} | Val: {len(val_set)}")

# Verify class balance in val
val_labels  = all_labels[val_idx]
val_counts  = Counter(val_labels)
print("  Val class distribution (stratified):")
for idx, count in val_counts.items():
    print(f"    {CLASS_NAMES[idx]}: {count}")

# ─── WEIGHTED SAMPLER ─────────────────────────────────────────────────────────
train_labels   = all_labels[train_idx]
class_counts   = np.bincount(train_labels)
class_weights  = 1.0 / class_counts
sample_weights = [class_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print("\n✓ DataLoaders ready!")

# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    """MobileNetV3-Small with a custom classification head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[3].in_features   # 1024
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


model = build_model(num_classes=NUM_CLASSES, freeze_backbone=False).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total params    : {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Model size      : ~{total_params * 4 / 1024**2:.1f} MB")

# ─── LOSS / OPTIMIZER / SCHEDULER ────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ─── EARLY STOPPING ───────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when val accuracy hasn't improved by MIN_DELTA
    for PATIENCE consecutive epochs.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = 0.0
        self.counter    = 0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs, patience, min_delta):
    best_val_acc  = 0.0
    best_weights  = None
    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    early_stop    = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(num_epochs):
        t0 = time.time()

        # ── TRAIN ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── VAL ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs     = model(images)
                loss        = criterion(outputs, labels)
                val_loss   += loss.item() * images.size(0)
                preds       = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            marker = "  ← NEW BEST"
        else:
            marker = ""

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch+1:03d}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
            f"({elapsed:.1f}s){marker}"
        )

        # Early stopping check
        if early_stop.step(val_acc):
            print(f"\n⚑ Early stopping triggered — no improvement for {patience} epochs.")
            break

    model.load_state_dict(best_weights)
    print(f"\n✓ Best Val Accuracy: {best_val_acc:.4f}")
    return model, history


model, history = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
)

# ─── SAVE CHECKPOINT ──────────────────────────────────────────────────────────
checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_names"     : CLASS_NAMES,
    "num_classes"     : NUM_CLASSES,
    "img_size"        : IMG_SIZE,
    "architecture"    : "mobilenet_v3_small",
}
torch.save(checkpoint, str(SAVE_PATH))
print(f"✓ Model saved to: {SAVE_PATH}")

# ─── EVALUATION (val set only — no data leakage) ─────────────────────────────
# Runs on the held-out val split only — NOT the full dataset.
# The old code ran on all images including training images, inflating metrics.

print("\n" + "=" * 55)
print("  EVALUATION  (stratified held-out val set)")
print("=" * 55)

model.eval()
all_labels_eval, all_preds_eval, all_probs_eval = [], [], []

with torch.no_grad():
    for images, labels in val_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = probs.argmax(dim=1)

        all_labels_eval.extend(labels.numpy())
        all_preds_eval.extend(preds.cpu().numpy())
        all_probs_eval.extend(probs.cpu().numpy())

all_labels_eval = np.array(all_labels_eval)
all_preds_eval  = np.array(all_preds_eval)
all_probs_eval  = np.array(all_probs_eval)

accuracy    = accuracy_score(all_labels_eval, all_preds_eval)
precision   = precision_score(all_labels_eval, all_preds_eval, average="weighted")
recall      = recall_score(all_labels_eval, all_preds_eval, average="weighted")
f1          = f1_score(all_labels_eval, all_preds_eval, average="weighted")
cm          = confusion_matrix(all_labels_eval, all_preds_eval)

specificity_per_class = []
for i in range(NUM_CLASSES):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
specificity = np.mean(specificity_per_class)

if NUM_CLASSES == 2:
    auc = roc_auc_score(all_labels_eval, all_probs_eval[:, 1])
else:
    auc = roc_auc_score(all_labels_eval, all_probs_eval, multi_class="ovr")

print(f"  Accuracy    : {accuracy*100:.2f}%")
print(f"  Precision   : {precision*100:.2f}%")
print(f"  Recall      : {recall*100:.2f}%")
print(f"  F1 Score    : {f1*100:.2f}%")
print(f"  Specificity : {specificity*100:.2f}%")
print(f"  AUC-ROC     : {auc:.4f}")
print("=" * 55)
print("\nPer-Class Report:")
print(classification_report(all_labels_eval, all_preds_eval, target_names=CLASS_NAMES))

# ─── SAVE EVALUATION PLOTS ────────────────────────────────────────────────────
# plt.show() is NOT called — plots are saved to disk so the UI can display them.
eval_plots_dir = SCRIPT_DIR / "outputs" / "eval_plots"
eval_plots_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

# Metrics bar chart
metrics_names  = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "AUC-ROC"]
metrics_values = [accuracy, precision, recall, f1, specificity, auc]
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
bars = axes[1].bar(metrics_names, [v * 100 for v in metrics_values],
                   color=colors, edgecolor="black", linewidth=0.7)
axes[1].set_ylim(0, 115)
axes[1].set_ylabel("Score (%)")
axes[1].set_title("All Metrics Overview", fontsize=13, fontweight="bold")
axes[1].tick_params(axis="x", rotation=25)
for bar, val in zip(bars, metrics_values):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

# ROC curve
if NUM_CLASSES == 2:
    fpr, tpr, _ = roc_curve(all_labels_eval, all_probs_eval[:, 1])
    axes[2].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.3)

plt.tight_layout()
eval_fig_path = eval_plots_dir / "evaluation_metrics.png"
fig.savefig(str(eval_fig_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n✓ Evaluation plot saved → {eval_fig_path}")

# ─── INFERENCE HELPERS ────────────────────────────────────────────────────────
# These functions are used by the UI. plt.show() is intentionally NOT called.
# The caller (app.py or the UI) is responsible for rendering/displaying plots.

def load_model(checkpoint_path: str = None, device: torch.device = None):
    """
    Load glaucoma model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to .pth file. Defaults to SAVE_PATH (script-relative).
    device : torch.device or None
        Inference device. Defaults to CUDA if available.

    Returns
    -------
    (model, class_names)
    """
    path   = Path(checkpoint_path) if checkpoint_path else SAVE_PATH
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt        = torch.load(str(path), map_location=device)
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]

    mdl = models.mobilenet_v3_small(weights=None)
    in_features = mdl.classifier[3].in_features
    mdl.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device).eval()

    print(f"✓ Model loaded | Classes: {class_names} | Device: {device}")
    return mdl, class_names


def predict_single_image(
    image_path: str,
    model: nn.Module,
    class_names: list,
    device: torch.device = None,
) -> dict:
    """
    Run inference on a single image.

    Returns
    -------
    dict with keys:
        predicted_class : str   — e.g. 'Glaucoma'
        confidence      : float — 0–100
        probabilities   : dict  — {class_name: probability_0_to_100}
        pred_index      : int   — index into class_names
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs      = torch.softmax(model(tensor), dim=1)[0]
        pred_idx   = probs.argmax().item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item() * 100

    # Return structured result — no plt.show() / no figure creation here.
    # The UI renders the confidence bar and image display itself.
    return {
        "predicted_class": pred_class,
        "confidence"     : round(confidence, 2),
        "probabilities"  : {class_names[i]: round(probs[i].item() * 100, 2)
                            for i in range(len(class_names))},
        "pred_index"     : pred_idx,
    }


def predict_and_plot(
    image_path: str,
    model: nn.Module,
    class_names: list,
    device: torch.device = None,
    save_path: str = None,
) -> tuple:
    """
    Run inference and generate a confidence bar chart figure.

    Returns (result_dict, matplotlib_figure).
    Figure is NOT shown — caller decides whether to save or display it.
    Pass save_path to write the figure to disk automatically.
    """
    result = predict_single_image(image_path, model, class_names, device)

    img    = Image.open(image_path).convert("RGB")
    probs  = [result["probabilities"][c] for c in class_names]
    color  = "#e74c3c" if result["pred_index"] == 0 else "#2ecc71"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    bars = axes[1].bar(class_names, probs,
                       color=["#e74c3c", "#2ecc71"], edgecolor="black")
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Confidence (%)")
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, f"{p:.1f}%",
                     ha="center", fontweight="bold")
    fig.suptitle(
        f"→ {result['predicted_class']}  ({result['confidence']:.1f}%)",
        fontsize=14, fontweight="bold", color=color,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Plot saved → {save_path}")

    # plt.show() intentionally omitted — caller handles display
    return result, fig


def predict_batch(
    image_paths: list,
    model: nn.Module,
    class_names: list,
    device: torch.device = None,
    save_grid_path: str = None,
) -> list:
    """
    Run inference on a list of image paths.

    Returns list of result dicts (same format as predict_single_image).
    Optionally saves a summary grid image to save_grid_path.
    plt.show() is NOT called.
    """
    device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    rows    = len(image_paths)

    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    if rows == 1:
        axes = [axes]

    for i, path in enumerate(image_paths):
        result = predict_single_image(path, model, class_names, device)
        results.append(result)

        img   = Image.open(path).convert("RGB")
        probs = [result["probabilities"][c] for c in class_names]
        color = "#e74c3c" if result["pred_index"] == 0 else "#2ecc71"

        axes[i][0].imshow(img)
        axes[i][0].axis("off")
        axes[i][0].set_title(Path(path).name, fontsize=10)

        bars = axes[i][1].bar(class_names, probs,
                               color=["#e74c3c", "#2ecc71"],
                               edgecolor="black", linewidth=0.8)
        axes[i][1].set_ylim(0, 120)
        axes[i][1].set_ylabel("Confidence (%)")
        for bar, p in zip(bars, probs):
            axes[i][1].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 1, f"{p:.1f}%",
                             ha="center", fontweight="bold", fontsize=11)
        axes[i][1].set_title(
            f"→ {result['predicted_class']}  ({result['confidence']:.1f}%)",
            fontsize=12, fontweight="bold", color=color,
        )

    plt.tight_layout()

    if save_grid_path:
        fig.savefig(save_grid_path, dpi=150, bbox_inches="tight")
        print(f"✓ Batch grid saved → {save_grid_path}")

    plt.close(fig)

    # Print summary table
    print(f"\n{'─'*55}")
    print(f"{'Image':<30} {'Prediction':<18} {'Confidence':>10}")
    print(f"{'─'*55}")
    for r, path in zip(results, image_paths):
        print(f"{Path(path).name:<30} {r['predicted_class']:<18} {r['confidence']:>9.2f}%")
    print(f"{'─'*55}")

    return results


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: single image prediction after training
    # Uncomment and set IMAGE_PATH to test inference standalone.
    #
    # IMAGE_PATH = SCRIPT_DIR / "test_images" / "sample.jpg"
    # mdl, cls   = load_model()
    # result     = predict_single_image(str(IMAGE_PATH), mdl, cls)
    # print(result)
    pass
