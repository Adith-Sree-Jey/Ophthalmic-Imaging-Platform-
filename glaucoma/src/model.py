from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class EfficientGlaucomaNet(nn.Module):
    """
    EfficientNet-B4 backbone with a lightweight classification head.
    Input: 224x224 RGB optic disc crop
    Output: 2-class logits [glaucoma, normal]
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_p: float = 0.4,
    ):
        super().__init__()

        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b4(weights=weights)
        self.features = base.features
        self.pool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(1792),
            nn.Dropout(p=dropout_p),
            nn.Linear(1792, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_p / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# Keep the legacy name so existing imports in the backend continue to work.
HybridGlaucomaNet = EfficientGlaucomaNet


def build_hybrid_model(
    num_classes: int = 2,
    pretrained: bool = True,
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> EfficientGlaucomaNet:
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientGlaucomaNet(num_classes=num_classes, pretrained=pretrained)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        state = checkpoint.get("model_state_dict", checkpoint)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            print(f"[WARNING] Checkpoint mismatch: {exc}")
            model.load_state_dict(state, strict=False)
        print(f"[EfficientGlaucomaNet] Loaded checkpoint: {checkpoint_path}")

    return model.to(resolved_device).eval()
