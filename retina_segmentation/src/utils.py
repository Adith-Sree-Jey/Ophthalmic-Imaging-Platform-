import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# ---------- METRIC FUNCTIONS ----------
def dice_coef(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """Dice Coefficient"""
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    return ((2. * intersection + smooth) / 
            (y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + smooth)).mean().item()


def iou_score(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """Intersection over Union (IoU)"""
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = (y_pred + y_true).sum(dim=(2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


def sensitivity(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """Sensitivity (Recall)"""
    y_pred = (y_pred > threshold).float()
    tp = (y_pred * y_true).sum(dim=(2, 3))
    fn = ((1 - y_pred) * y_true).sum(dim=(2, 3))
    return ((tp + smooth) / (tp + fn + smooth)).mean().item()


def specificity(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """Specificity"""
    y_pred = (y_pred > threshold).float()
    tn = ((1 - y_pred) * (1 - y_true)).sum(dim=(2, 3))
    fp = (y_pred * (1 - y_true)).sum(dim=(2, 3))
    return ((tn + smooth) / (tn + fp + smooth)).mean().item()


def accuracy(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """Pixel accuracy: (TP + TN) / total pixels"""
    y_pred = (y_pred > threshold).float()
    tp = (y_pred * y_true).sum(dim=(2, 3))
    tn = ((1 - y_pred) * (1 - y_true)).sum(dim=(2, 3))
    total = y_pred.numel() // y_pred.shape[0]  # pixels per sample
    correct = (tp + tn).sum(dim=1)
    return ((correct + smooth) / (total + smooth)).mean().item()


# ---------- POST-PROCESSING ----------
def clean_prediction(pred_mask):
    """
    Apply morphological operations to clean segmentation mask.
    Removes noise & fills gaps.
    """
    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    return pred_mask


# ---------- VISUALIZATION ----------
def plot_results(image, ground_truth, prediction, save_path=None):
    """
    Plot original image, ground truth, and prediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Prediction (Cleaned)")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
