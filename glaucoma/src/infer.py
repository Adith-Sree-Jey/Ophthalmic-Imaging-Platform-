from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from glaucoma.src.model import EfficientGlaucomaNet, HybridGlaucomaNet, build_hybrid_model
    from glaucoma.src.vessel_features import VesselFeatures, extract_vessel_features
except ImportError:
    from .model import EfficientGlaucomaNet, HybridGlaucomaNet, build_hybrid_model
    from .vessel_features import VesselFeatures, extract_vessel_features


CLASS_NAMES = ["glaucoma", "normal"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = MODULE_DIR / "outputs" / "checkpoints" / "glaucoma_mobilenetv3.pth"
DEFAULT_HYBRID_CHECKPOINT_PATH = MODULE_DIR / "outputs" / "checkpoints" / "hybrid_glaucoma.pth"
INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _format_prediction_label(label: str) -> str:
    normalized = (label or "").strip().replace("_", " ").lower()
    if normalized == "non glaucoma":
        normalized = "normal"
    return normalized.title() if normalized else "Unknown"


def _build_legacy_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _compute_disc_crop_box(image: Image.Image, crop_size: int = 400) -> tuple[int, int, int, int]:
    img = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    green = bgr[:, :, 1]
    blurred = cv2.GaussianBlur(green, (51, 51), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx, cy = max_loc

    h, w = img.shape[:2]
    mx, my = int(w * 0.20), int(h * 0.20)
    if not (mx < cx < w - mx and my < cy < h - my):
        cx, cy = w // 2, h // 2

    half = crop_size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    return x1, y1, x2, y2


def _apply_disc_crop(image: Image.Image, crop_size: int = 400) -> Image.Image:
    """
    Crop around optic disc (brightest region in green channel).
    This is required for EfficientGlaucomaNet-B4 - model was trained on crops.
    Falls back to centre crop if disc not found in central 60% of image.
    """
    img = np.array(image.convert("RGB"))
    x1, y1, x2, y2 = _compute_disc_crop_box(image, crop_size=crop_size)
    cropped = img[y1:y2, x1:x2]
    cropped_image = Image.fromarray(cropped)
    cropped_image.info["disc_crop_box"] = (x1, y1, x2, y2)
    return cropped_image


def _crop_vessel_mask_to_disc(vessel_mask, crop_box, target_shape):
    """
    Crop vessel mask to disc region. Handles empty, None, or mismatched masks gracefully.
    """
    try:
        if vessel_mask is None:
            return None

        mask_np = np.array(vessel_mask)

        # Guard: empty or zero-dimension mask
        if mask_np.size == 0 or 0 in mask_np.shape:
            return None

        # Guard: mask too small to crop meaningfully
        if mask_np.shape[0] < 4 or mask_np.shape[1] < 4:
            return None

        target_h, target_w = target_shape
        if target_h <= 0 or target_w <= 0:
            return None

        x1, y1, x2, y2 = crop_box

        # Clamp crop box to mask dimensions
        x1 = max(0, min(x1, mask_np.shape[1] - 1))
        y1 = max(0, min(y1, mask_np.shape[0] - 1))
        x2 = max(x1 + 1, min(x2, mask_np.shape[1]))
        y2 = max(y1 + 1, min(y2, mask_np.shape[0]))

        cropped = mask_np[y1:y2, x1:x2]

        # Guard: cropped result is empty
        if cropped.size == 0 or 0 in cropped.shape:
            return None

        resized = cv2.resize(
            cropped.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
        return resized

    except Exception as e:
        print(f"[Glaucoma] _crop_vessel_mask_to_disc failed gracefully: {e}")
        return None


def load_model(checkpoint_path: str = None, device=None):
    resolved_device = device or DEVICE
    resolved_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH

    try:
        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

        checkpoint = torch.load(str(resolved_path), map_location=resolved_device)
        class_names = checkpoint.get("class_names")
        num_classes = checkpoint.get("num_classes")

        if not isinstance(class_names, list) or not class_names:
            raise ValueError("Checkpoint is missing a valid 'class_names' list.")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("Checkpoint is missing a valid 'num_classes' value.")

        model = _build_legacy_model(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device).eval()
        return model, class_names
    except Exception as exc:
        raise RuntimeError(f"Failed to load glaucoma model from '{resolved_path}': {exc}") from exc


def load_hybrid_model(checkpoint_path: str = None, device=None) -> EfficientGlaucomaNet:
    """
    Load EfficientGlaucomaNet-B4 from checkpoint.
    Called by glaucoma_service.py when hybrid_glaucoma.pth exists.
    """
    resolved_device = device or DEVICE
    resolved_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_HYBRID_CHECKPOINT_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Hybrid checkpoint not found: {resolved_path}")

    checkpoint = torch.load(str(resolved_path), map_location=resolved_device)
    num_classes = int(checkpoint.get("num_classes", len(CLASS_NAMES)) or len(CLASS_NAMES))
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    test_metrics = checkpoint.get("test_metrics", "not stored")

    model = EfficientGlaucomaNet(
        num_classes=num_classes,
        pretrained=False,
        dropout_p=0.4,
    ).to(resolved_device)

    state = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state, strict=True)
        print("[Glaucoma] EfficientGlaucomaNet-B4 loaded (strict=True)")
    except RuntimeError as exc:
        print(f"[Glaucoma] strict=True failed: {exc}\nRetrying strict=False")
        model.load_state_dict(state, strict=False)

    model.eval()
    print(f"[Glaucoma] Hybrid model loaded. Classes: {class_names}")
    print(f"[Glaucoma] Test metrics: {test_metrics}")
    return model


def predict_single_image(image_path: str, model, class_names: list, device=None) -> dict:
    resolved_device = device or DEVICE

    try:
        image = Image.open(image_path).convert("RGB")
        tensor = INFER_TRANSFORM(image).unsqueeze(0).to(resolved_device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            pred_index = int(probabilities.argmax().item())

        return {
            "predicted_class": _format_prediction_label(class_names[pred_index]),
            "confidence": round(float(probabilities[pred_index].item() * 100), 2),
            "probabilities": {
                str(class_name).lower().replace("non_glaucoma", "normal"): round(
                    float(probabilities[index].item() * 100),
                    2,
                )
                for index, class_name in enumerate(class_names)
            },
            "pred_index": pred_index,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to run glaucoma inference for '{image_path}': {exc}") from exc


def predict_and_plot(image_path: str, model, class_names, device=None, save_path=None) -> tuple:
    try:
        result = predict_single_image(image_path, model, class_names, device=device)
        image = Image.open(image_path).convert("RGB")
        probabilities = [
            result["probabilities"].get(str(class_name).lower().replace("non_glaucoma", "normal"), 0.0)
            for class_name in class_names
        ]
        bar_colors = ["#B91C1C" if str(name).lower() == "glaucoma" else "#15803D" for name in class_names]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title("Input Image")

        bars = axes[1].bar(class_names, probabilities, color=bar_colors, edgecolor="black")
        axes[1].set_ylim(0, 115)
        axes[1].set_ylabel("Confidence (%)")

        for bar, probability in zip(bars, probabilities):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{probability:.1f}%",
                ha="center",
                fontweight="bold",
            )

        fig.suptitle(
            f"{result['predicted_class']} ({result['confidence']:.1f}%)",
            fontsize=14,
            fontweight="bold",
            color=bar_colors[result["pred_index"]] if result["pred_index"] < len(bar_colors) else "#172033",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return result, fig
    except Exception as exc:
        raise RuntimeError(f"Failed to create glaucoma plot for '{image_path}': {exc}") from exc


def run_glaucoma_inference(
    image: Image.Image,
    vessel_mask: np.ndarray = None,
    model: HybridGlaucomaNet = None,
    checkpoint_path: str = None,
) -> dict:
    print(f"[InferDebug] vessel_mask present: {vessel_mask is not None}")
    image = _apply_disc_crop(image, crop_size=400)

    if model is None:
        model = load_hybrid_model(checkpoint_path=checkpoint_path, device=DEVICE)

    model.eval()
    img_rgb = image.convert("RGB")
    img_np = np.array(img_rgb)
    tensor = INFER_TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)

    gradcam_map, logits = _run_with_gradcam(model, tensor)
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())
    prediction = _format_prediction_label(CLASS_NAMES[pred_idx])
    probabilities = {
        CLASS_NAMES[i]: round(float(probs[i].item() * 100), 2)
        for i in range(len(CLASS_NAMES))
    }
    confidence = round(float(probs[pred_idx].item() * 100), 2)

    gradcam_b64, gradcam_focus = _process_gradcam(gradcam_map, img_np)

    vessel_source = "segmentation"
    crop_box = image.info.get("disc_crop_box")
    effective_mask = _crop_vessel_mask_to_disc(vessel_mask, crop_box, img_np.shape[:2])
    if effective_mask is None:
        effective_mask = _make_fallback_mask(np.array(image))
        vessel_source = "estimated"
    if effective_mask is not None and (effective_mask.size == 0 or 0 in effective_mask.shape):
        effective_mask = None
    vessel_features: VesselFeatures = extract_vessel_features(
        vessel_mask=effective_mask,
        original_image=img_np,
    )
    overlay_b64 = _make_vessel_overlay(img_np, effective_mask)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "gradcam_b64": gradcam_b64,
        "gradcam_focus": gradcam_focus,
        "vessel_features": vessel_features.to_dict(),
        "risk_level": vessel_features.risk_level,
        "overlay_b64": overlay_b64,
        "vessel_source": vessel_source,
    }


def _run_with_gradcam(model: HybridGlaucomaNet, tensor: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    gradients = []
    activations = []

    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    tensor.requires_grad_(True)
    logits = model(tensor)
    pred_idx = int(logits.argmax(dim=1).item())
    logits[0, pred_idx].backward()

    forward_handle.remove()
    backward_handle.remove()

    if not gradients or not activations:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32), logits.detach()

    grads = gradients[0]
    activations_tensor = activations[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations_tensor).sum(dim=1).squeeze(0)
    cam = F.relu(cam).cpu().numpy()

    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam.astype(np.float32), logits.detach()


def _process_gradcam(cam: np.ndarray, img_np: np.ndarray) -> tuple[str, str]:
    height, width = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.55, heatmap_rgb, 0.45, 0)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    return encoded, _describe_cam_region(cam_resized, height, width)


def _describe_cam_region(cam: np.ndarray, height: int, width: int) -> str:
    peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)

    vertical = "superior" if peak_y < height * 0.4 else "inferior" if peak_y > height * 0.6 else "central"
    horizontal = "nasal" if peak_x < width * 0.4 else "temporal" if peak_x > width * 0.6 else ""
    region = f"{vertical} {horizontal}".strip() + " optic disc region"

    threshold = cam.max() * 0.7
    focus_ratio = float((cam > threshold).sum()) / float(cam.size or 1)
    spread = "diffuse activation across" if focus_ratio > 0.15 else "localized activation at"
    return f"{spread} {region}"


def _make_vessel_overlay(img_np: np.ndarray, vessel_mask: np.ndarray) -> str:
    if vessel_mask is None:
        return ""

    height, width = img_np.shape[:2]
    mask_resized = cv2.resize((vessel_mask > 0).astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    overlay = img_np.copy()
    overlay[mask_resized > 127, 1] = np.clip(overlay[mask_resized > 127, 1].astype(int) + 80, 0, 255)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _make_fallback_mask(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, mask = cv2.threshold(enhanced, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)
