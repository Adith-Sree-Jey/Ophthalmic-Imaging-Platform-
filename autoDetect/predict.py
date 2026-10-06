"""
predict.py

Test the trained model on a single image or a folder of images.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --folder path/to/folder/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path(__file__).resolve().parent / "model"
IMG_SIZE  = 224

CONFIDENCE_LABELS = {
    "high":   lambda p: p >= 0.90,
    "medium": lambda p: 0.70 <= p < 0.90,
    "low":    lambda p: p < 0.70,
}


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


def load_model() -> tuple[nn.Module, dict[str, str], torch.device]:
    class_map_path = MODEL_DIR / "class_map.json"
    if not class_map_path.exists():
        raise FileNotFoundError("class_map.json not found. Run train.py first.")

    with open(class_map_path) as f:
        class_map: dict[str, str] = json.load(f)

    classes     = [class_map[str(i)] for i in range(len(class_map))]
    num_classes = len(classes)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = build_model(num_classes).to(device)
    model.load_state_dict(
        torch.load(MODEL_DIR / "best_model.pth", map_location=device, weights_only=True)
    )
    model.eval()
    return model, class_map, device


def predict_image(
    image_path: Path,
    model: nn.Module,
    class_map: dict[str, str],
    device: torch.device,
) -> dict[str, str]:
    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx    = int(probs.argmax())
    pred_prob   = float(probs[pred_idx])
    pred_label  = class_map[str(pred_idx)]

    if pred_prob >= 0.90:
        confidence = "high"
    elif pred_prob >= 0.70:
        confidence = "medium"
    else:
        confidence = "low"

    all_probs = {class_map[str(i)]: f"{float(probs[i]):.1%}" for i in range(len(class_map))}

    return {
        "filename":      image_path.name,
        "detected_type": pred_label,
        "confidence":    confidence,
        "probability":   f"{pred_prob:.1%}",
        "all_probs":     all_probs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Provide --image or --folder")
        return

    print("Loading model...")
    model, class_map, device = load_model()
    print(f"Model loaded. Device: {device}\n")

    extensions = {".jpg", ".jpeg", ".png"}

    if args.image:
        paths = [Path(args.image)]
    else:
        paths = [p for p in Path(args.folder).iterdir()
                 if p.is_file() and p.suffix.lower() in extensions]
        paths.sort()

    print(f"{'Filename':<50} {'Type':<20} {'Confidence':<12} {'Probability'}")
    print("-" * 100)

    for path in paths:
        result = predict_image(path, model, class_map, device)
        print(
            f"{result['filename']:<50} "
            f"{result['detected_type']:<20} "
            f"{result['confidence']:<12} "
            f"{result['probability']}"
        )
        if args.image:
            print(f"\n  All probabilities:")
            for cls, prob in result["all_probs"].items():
                print(f"    {cls:<22} {prob}")


if __name__ == "__main__":
    main()