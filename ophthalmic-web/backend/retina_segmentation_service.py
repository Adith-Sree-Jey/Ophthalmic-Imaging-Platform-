from __future__ import annotations

import base64
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from retina_segmentation.src.infer import load_model, predict_single_image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RETINA_MODULE_DIR = WORKSPACE_ROOT / "retina_segmentation"
RETINA_TEMP_DIR = RETINA_MODULE_DIR / "temp"
RETINA_CHECKPOINT = RETINA_MODULE_DIR / "outputs" / "checkpoints" / "best_model.pth"

if not RETINA_MODULE_DIR.exists():
  raise RuntimeError(f"Retina module not found: {RETINA_MODULE_DIR}")


@lru_cache
def get_retina_model():
  if not RETINA_CHECKPOINT.exists():
    raise FileNotFoundError(f"Retina checkpoint not found: {RETINA_CHECKPOINT}")
  return load_model(RETINA_CHECKPOINT)


def _encode_png_base64(image: np.ndarray) -> str:
  ok, buffer = cv2.imencode(".png", image)
  if not ok:
    raise RuntimeError("Failed to encode retina segmentation image.")
  return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _build_overlay(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
  overlay = image_bgr.copy()
  vessel_mask = mask_u8 > 0
  green = np.zeros_like(overlay)
  green[..., 1] = 255
  overlay[vessel_mask] = cv2.addWeighted(
    overlay[vessel_mask],
    0.5,
    green[vessel_mask],
    0.5,
    0.0,
  )
  return overlay


def segment_retina_image(image_bytes: bytes, filename: str, threshold: float) -> dict[str, Any]:
  RETINA_TEMP_DIR.mkdir(parents=True, exist_ok=True)

  suffix = Path(filename or "fundus.png").suffix or ".png"
  file_id = uuid.uuid4().hex
  input_path = RETINA_TEMP_DIR / f"{file_id}_input{suffix}"
  mask_path = RETINA_TEMP_DIR / f"{file_id}_mask.png"

  input_path.write_bytes(image_bytes)

  model = get_retina_model()
  mask = predict_single_image(str(input_path), model, threshold=threshold)
  mask_u8 = mask.astype(np.uint8)
  cv2.imwrite(str(mask_path), mask_u8)

  image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
  original_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
  if original_bgr is None:
    raise RuntimeError("Uploaded fundus image could not be decoded.")

  resized_bgr = cv2.resize(
    original_bgr,
    (mask_u8.shape[1], mask_u8.shape[0]),
    interpolation=cv2.INTER_AREA,
  )
  overlay_bgr = _build_overlay(resized_bgr, mask_u8)

  return {
    "original_image_base64": _encode_png_base64(resized_bgr),
    "mask_base64": _encode_png_base64(mask_u8),
    "overlay_image_base64": _encode_png_base64(overlay_bgr),
    "mask_filename": f"{Path(filename or 'fundus').stem}_vessel_mask.png",
    "temp_input_path": str(input_path),
    "temp_mask_path": str(mask_path),
  }


def get_vessel_mask(image_bytes: bytes, threshold: float = 0.5) -> np.ndarray:
  """
  Expose binary vessel mask for consumption by the glaucoma pipeline.
  Returns an H x W uint8 array with values 0 or 1.
  """
  RETINA_TEMP_DIR.mkdir(parents=True, exist_ok=True)

  file_id = uuid.uuid4().hex
  input_path = RETINA_TEMP_DIR / f"{file_id}_glaucoma_input.png"

  try:
    input_path.write_bytes(image_bytes)
    model = get_retina_model()
    mask = predict_single_image(str(input_path), model, threshold=threshold)
    return (mask > 0).astype(np.uint8)
  finally:
    if input_path.exists():
      try:
        input_path.unlink()
      except OSError:
        pass
