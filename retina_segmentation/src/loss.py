import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation
    alpha: controls penalty for false positives
    beta: controls penalty for false negatives
    smooth: avoids division by zero
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Ensure same shape and type
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()

        # True positives, false positives & false negatives
        tp = (y_pred * y_true).sum(dim=(2, 3))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky_index.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: focuses on hard-to-segment pixels
    gamma > 1 increases focus on hard examples
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.33, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        tversky_loss = self.tversky(y_pred, y_true)
        return torch.pow(tversky_loss, self.gamma)


class DiceLoss(nn.Module):
    """Soft Dice loss (differentiable). Helps prevent 'predict all positive' collapse."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
