"""
image_classifier.py  (autoDetect version)

Drop-in replacement for the backend image_classifier.py.
Uses the locally trained MobileNetV3-Small model instead of any external API.

No internet required. No API keys. Runs in milliseconds.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR      = Path(__file__).resolve().parent / "model"
MODEL_WEIGHTS  = MODEL_DIR / "best_model.pth"
CLASS_MAP_FILE = MODEL_DIR / "class_map.json"
IMG_SIZE       = 224

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, reused for every request
# ---------------------------------------------------------------------------
_model:     nn.Module | None       = None
_class_map: dict[str, str] | None  = None
_device:    torch.device | None    = None
_transform: transforms.Compose | None = None
_load_error: str | None            = None


def _ensure_loaded() -> None:
    global _model, _class_map, _device, _transform, _load_error

    if _model is not None or _load_error is not None:
        return

    try:
        if not CLASS_MAP_FILE.exists():
            _load_error = f"class_map.json not found at {CLASS_MAP_FILE}. Run train.py first."
            return
        if not MODEL_WEIGHTS.exists():
            _load_error = f"best_model.pth not found at {MODEL_WEIGHTS}. Run train.py first."
            return

        with open(CLASS_MAP_FILE) as f:
            _class_map = json.load(f)

        num_classes = len(_class_map)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        m = models.mobilenet_v3_small(weights=None)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
        m.load_state_dict(
            torch.load(MODEL_WEIGHTS, map_location=_device, weights_only=True)
        )
        m.eval()
        _model = m.to(_device)

        _transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    except Exception as exc:
        _load_error = f"Failed to load image type classifier: {exc}"


def classify_image_type(image_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Classify an eye image as anterior_segment, red_glow, or slit_lamp.

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes (JPG or PNG).
    filename : str
        Original filename (used for mime type hint and result reporting).

    Returns
    -------
    dict with keys: filename, detected_type, confidence, reasoning
    """
    _ensure_loaded()

    if _load_error:
        return _fallback(filename, _load_error)

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _transform(image).unsqueeze(0).to(_device)  # type: ignore[arg-type]

        with torch.no_grad():
            logits = _model(tensor)                          # type: ignore[misc]
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu()

        pred_idx   = int(probs.argmax())
        pred_prob  = float(probs[pred_idx])
        pred_label = _class_map[str(pred_idx)]               # type: ignore[index]

        if pred_prob >= 0.90:
            confidence = "high"
        elif pred_prob >= 0.70:
            confidence = "medium"
        else:
            confidence = "low"

        reasoning = (
            f"Local classifier: {pred_label} "
            f"({pred_prob:.1%} confidence)"
        )

        return {
            "filename":      filename,
            "detected_type": pred_label,
            "confidence":    confidence,
            "reasoning":     reasoning,
        }

    except Exception as exc:
        return _fallback(filename, f"Classification failed: {exc}")


def _fallback(filename: str, reasoning: str) -> dict[str, Any]:
    inferred_type = _infer_type_from_filename(filename)
    if inferred_type != "unknown":
        reasoning = f"{reasoning} Falling back to filename pattern."
    return {
        "filename":      filename,
        "detected_type": inferred_type,
        "confidence":    "medium" if inferred_type != "unknown" else "low",
        "reasoning":     reasoning,
    }


def _infer_type_from_filename(filename: str) -> str:
    text = (filename or "").lower()

    if (
        "anterior" in text
        or "ant seg" in text
        or "anterior_segment" in text
        or "anterior segment" in text
    ):
        return "anterior_segment"

    if (
        "red_glow" in text
        or "red glow" in text
        or "retroillumination" in text
        or "retro illumination" in text
    ):
        return "red_glow"

    if "slit_lamp" in text or "slit lamp" in text or "slitlamp" in text:
        return "slit_lamp"

    return "unknown"
