"""
Load STARE dataset in DatasetNinja format: decode annotation JSONs (bitmap masks) and pair with images.
Used by evaluate_stare.py.
"""
import os
import base64
import zlib
import json
import numpy as np
import cv2


def decode_bitmap_to_patch(data_b64: str) -> np.ndarray | None:
    """
    Decode DatasetNinja/Supervisely bitmap.data (base64) into a grayscale mask patch (0/255).
    Tries: base64 -> zlib decompress -> PNG decode, or base64 -> PNG decode.
    Returns (H, W) uint8 array or None on failure.
    """
    try:
        raw = base64.b64decode(data_b64)
    except Exception:
        return None
    # Often base64(zlib(png_bytes))
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    # Binarize: any non-zero -> 255
    return (arr > 0).astype(np.uint8) * 255


def load_mask_from_annotation(ann_path: str) -> tuple[np.ndarray, int, int] | None:
    """
    Load full binary mask from a DatasetNinja annotation JSON (STARE vessels).
    Returns (mask_uint8, height, width) with values 0 or 255, or None if file invalid.
    """
    if not os.path.isfile(ann_path):
        return None
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    size = data.get("size", {})
    h, w = int(size.get("height", 0)), int(size.get("width", 0))
    if h <= 0 or w <= 0:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "bitmap" or obj.get("classTitle") != "vessels":
            continue
        bm = obj.get("bitmap", {})
        b64 = bm.get("data")
        origin = bm.get("origin", [0, 0])
        if not b64:
            continue
        patch = decode_bitmap_to_patch(b64)
        if patch is None:
            continue
        ph, pw = patch.shape
        x, y = int(origin[0]), int(origin[1])
        # Clip to canvas
        x2 = min(x + pw, w)
        y2 = min(y + ph, h)
        px2 = x2 - x
        py2 = y2 - y
        if px2 <= 0 or py2 <= 0:
            continue
        mask[y:y2, x:x2] = np.maximum(mask[y:y2, x:x2], patch[:py2, :px2])
    return mask, h, w


def get_stare_pairs(stare_ds_root: str) -> list[tuple[str, str]]:
    """
    From STARE DatasetNinja root (path to folder containing ds/ann and ds/img),
    return list of (image_path, annotation_path) for every annotation file that has
    a matching image. Image name is derived from annotation: im0001.png.json -> im0001.png.
    """
    ann_dir = os.path.join(stare_ds_root, "ds", "ann")
    img_dir = os.path.join(stare_ds_root, "ds", "img")
    if not os.path.isdir(ann_dir):
        return []
    pairs = []
    for name in sorted(os.listdir(ann_dir)):
        if not name.endswith(".json"):
            continue
        # im0001.png.json -> im0001.png
        base = name[:-5]  # strip .json
        if not base.endswith(".png"):
            base = base + ".png"
        ann_path = os.path.join(ann_dir, name)
        img_path = os.path.join(img_dir, base)
        if os.path.isfile(img_path):
            pairs.append((img_path, ann_path))
    return pairs
