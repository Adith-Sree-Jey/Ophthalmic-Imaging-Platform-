from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any

from PIL import Image
from glaucoma.src.report_service import generate_glaucoma_report as build_glaucoma_report


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GLAUCOMA_MODULE_DIR = WORKSPACE_ROOT / "glaucoma"
GLAUCOMA_TEMP_DIR = GLAUCOMA_MODULE_DIR / "temp"
GLAUCOMA_LEGACY_CHECKPOINT = GLAUCOMA_MODULE_DIR / "outputs" / "checkpoints" / "glaucoma_mobilenetv3.pth"
GLAUCOMA_HYBRID_CHECKPOINT = GLAUCOMA_MODULE_DIR / "outputs" / "checkpoints" / "hybrid_glaucoma.pth"


_GLAUCOMA_INFER_MODULE = None
_GLAUCOMA_MODEL = None
_GLAUCOMA_CLASS_NAMES = None
_GLAUCOMA_MODE = None
_GLAUCOMA_STATUS = {
    "available": False,
    "status": "Model not found",
    "reason": f"No glaucoma checkpoint found under {GLAUCOMA_MODULE_DIR / 'outputs' / 'checkpoints'}",
    "checkpoint_path": None,
    "mode": None,
}


def _load_infer_module():
    return importlib.import_module("glaucoma.src.infer")


def _load_vessel_mask(image_path: str):
    import numpy as np
    try:
        from retina_segmentation_service import get_vessel_mask
        with open(image_path, "rb") as image_file:
            mask = get_vessel_mask(image_file.read())

        # Validate mask before returning
        if mask is None:
            return None
        mask_np = np.array(mask)
        if mask_np.size == 0 or 0 in mask_np.shape:
            print(f"[Glaucoma] Vessel mask empty for {image_path} — using fallback")
            return None
        return mask_np

    except Exception as exc:
        print(f"[Glaucoma] Vessel mask unavailable: {exc}")
        return None


def _normalize_inference_result(result: dict[str, Any], model_mode: str) -> dict[str, Any]:
    prediction = result.get("prediction") or result.get("predicted_class", "Unknown")
    vessel_features = result.get("vessel_features") or {}
    risk_level = result.get("risk_level") or vessel_features.get("risk_level")

    return {
        "prediction": prediction,
        "predicted_class": prediction,
        "confidence": round(float(result.get("confidence") or 0.0), 2),
        "probabilities": result.get("probabilities", {}) or {},
        "risk_level": risk_level,
        "gradcam_b64": result.get("gradcam_b64"),
        "gradcam_focus": result.get("gradcam_focus"),
        "overlay_b64": result.get("overlay_b64"),
        "vessel_features": vessel_features or None,
        "vessel_source": result.get("vessel_source"),
        "model_mode": model_mode,
    }


def preload_glaucoma_model() -> None:
    global _GLAUCOMA_INFER_MODULE, _GLAUCOMA_MODEL, _GLAUCOMA_CLASS_NAMES, _GLAUCOMA_MODE, _GLAUCOMA_STATUS

    try:
        _GLAUCOMA_INFER_MODULE = _load_infer_module()
    except Exception as exc:
        _GLAUCOMA_MODEL = None
        _GLAUCOMA_CLASS_NAMES = None
        _GLAUCOMA_MODE = None
        _GLAUCOMA_STATUS = {
            "available": False,
            "status": "Model unavailable",
            "reason": str(exc),
            "checkpoint_path": None,
            "mode": None,
        }
        print(f"[Glaucoma] Warning: {exc}")
        return

    if GLAUCOMA_HYBRID_CHECKPOINT.exists():
        try:
            _GLAUCOMA_MODEL = _GLAUCOMA_INFER_MODULE.load_hybrid_model(str(GLAUCOMA_HYBRID_CHECKPOINT))
            _GLAUCOMA_CLASS_NAMES = list(getattr(_GLAUCOMA_INFER_MODULE, "CLASS_NAMES", ["glaucoma", "normal"]))
            _GLAUCOMA_MODE = "hybrid"
            _GLAUCOMA_STATUS = {
                "available": True,
                "status": "Active workspace",
                "reason": None,
                "checkpoint_path": str(GLAUCOMA_HYBRID_CHECKPOINT),
                "mode": _GLAUCOMA_MODE,
            }
            print("[Glaucoma] Hybrid model loaded at startup.")
            return
        except Exception as exc:
            print(f"[Glaucoma] Hybrid checkpoint load failed: {exc}")

    if GLAUCOMA_LEGACY_CHECKPOINT.exists():
        try:
            _GLAUCOMA_MODEL, _GLAUCOMA_CLASS_NAMES = _GLAUCOMA_INFER_MODULE.load_model(str(GLAUCOMA_LEGACY_CHECKPOINT))
            _GLAUCOMA_MODE = "legacy"
            _GLAUCOMA_STATUS = {
                "available": True,
                "status": "Active workspace",
                "reason": None,
                "checkpoint_path": str(GLAUCOMA_LEGACY_CHECKPOINT),
                "mode": _GLAUCOMA_MODE,
            }
            print("[Glaucoma] Legacy model loaded at startup.")
            return
        except Exception as exc:
            _GLAUCOMA_MODEL = None
            _GLAUCOMA_CLASS_NAMES = None
            _GLAUCOMA_MODE = None
            _GLAUCOMA_STATUS = {
                "available": False,
                "status": "Model unavailable",
                "reason": str(exc),
                "checkpoint_path": str(GLAUCOMA_LEGACY_CHECKPOINT),
                "mode": None,
            }
            print(f"[Glaucoma] Warning: {exc}")
            return

    _GLAUCOMA_MODEL = None
    _GLAUCOMA_CLASS_NAMES = None
    _GLAUCOMA_MODE = None
    _GLAUCOMA_STATUS = {
        "available": False,
        "status": "Model not found",
        "reason": f"Neither hybrid nor legacy checkpoint was found in {GLAUCOMA_MODULE_DIR / 'outputs' / 'checkpoints'}",
        "checkpoint_path": None,
        "mode": None,
    }
    print(f"[Glaucoma] Warning: {_GLAUCOMA_STATUS['reason']}")


def ensure_glaucoma_model_loaded(force_retry: bool = False) -> None:
    has_any_checkpoint = GLAUCOMA_HYBRID_CHECKPOINT.exists() or GLAUCOMA_LEGACY_CHECKPOINT.exists()
    should_retry = force_retry or (not _GLAUCOMA_STATUS.get("available") and has_any_checkpoint)
    if should_retry:
        preload_glaucoma_model()


def get_glaucoma_module_status() -> dict[str, Any]:
    ensure_glaucoma_model_loaded()
    return dict(_GLAUCOMA_STATUS)


def predict_glaucoma_file(image_path: str) -> dict[str, Any]:
    ensure_glaucoma_model_loaded()
    if not _GLAUCOMA_STATUS.get("available") or _GLAUCOMA_MODEL is None or _GLAUCOMA_INFER_MODULE is None:
        raise RuntimeError(_GLAUCOMA_STATUS.get("reason") or "Glaucoma module is unavailable.")

    try:
        if _GLAUCOMA_MODE == "hybrid":
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
            vessel_mask = _load_vessel_mask(image_path)
            result = _GLAUCOMA_INFER_MODULE.run_glaucoma_inference(
                image=image,
                vessel_mask=vessel_mask,
                model=_GLAUCOMA_MODEL,
            )
            return _normalize_inference_result(result, "hybrid")

        legacy_result = _GLAUCOMA_INFER_MODULE.predict_single_image(
            image_path,
            _GLAUCOMA_MODEL,
            _GLAUCOMA_CLASS_NAMES,
        )
        return _normalize_inference_result(legacy_result, "legacy")
    except Exception as exc:
        raise RuntimeError(f"Glaucoma prediction failed: {exc}") from exc


def predict_glaucoma_bytes(image_bytes: bytes) -> dict[str, Any]:
    ensure_glaucoma_model_loaded()
    if not _GLAUCOMA_STATUS.get("available") or _GLAUCOMA_MODEL is None or _GLAUCOMA_INFER_MODULE is None:
        raise RuntimeError(_GLAUCOMA_STATUS.get("reason") or "Glaucoma module is unavailable.")

    if _GLAUCOMA_MODE != "hybrid":
        raise RuntimeError("Byte-stream prediction is only available when the hybrid glaucoma model is active.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    try:
        from retina_segmentation_service import get_vessel_mask

        vessel_mask = get_vessel_mask(image_bytes)
    except Exception as exc:
        print(f"[Glaucoma] Vessel mask unavailable: {exc}")
        vessel_mask = None

    result = _GLAUCOMA_INFER_MODULE.run_glaucoma_inference(
        image=image,
        vessel_mask=vessel_mask,
        model=_GLAUCOMA_MODEL,
    )
    return _normalize_inference_result(result, "hybrid")


def predict_glaucoma(image_bytes: bytes) -> dict[str, Any]:
    return predict_glaucoma_bytes(image_bytes)


def generate_report(
    *,
    image_bytes: bytes,
    patient_info: dict[str, Any] | None = None,
    backend: str = "local",
) -> dict[str, Any]:
    inference_result = predict_glaucoma(image_bytes)
    vf = inference_result.get("vessel_features") or {}
    print(f"[ReportDebug] vessel_features: {vf}")
    print(
        f"[ReportDebug] prediction: {inference_result.get('prediction')} "
        f"confidence: {inference_result.get('confidence')}"
    )
    print(f"[ReportDebug] model_mode: {inference_result.get('model_mode')}")
    report = build_glaucoma_report(inference_result, backend=backend)
    patient_info = patient_info or {}

    return {
        "prediction": inference_result.get("prediction", "Unknown"),
        "confidence": inference_result.get("confidence", 0.0),
        "risk_level": inference_result.get("risk_level"),
        "probabilities": inference_result.get("probabilities", {}),
        "vessel_features": inference_result.get("vessel_features", {}),
        "vessel_source": inference_result.get("vessel_source", "estimated"),
        "report": report.to_dict(),
        "patient_info": {
            "patient_id": patient_info.get("patient_id", ""),
            "eye_side": patient_info.get("eye_side", ""),
        },
        "gradcam_b64": inference_result.get("gradcam_b64"),
        "overlay_b64": inference_result.get("overlay_b64"),
        "gradcam_focus": inference_result.get("gradcam_focus"),
        "model_mode": inference_result.get("model_mode"),
    }


def generate_report_from_file(
    *,
    image_path: str,
    patient_info: dict[str, Any] | None = None,
    backend: str = "local",
) -> dict[str, Any]:
    """Generate a report through the file path supported by hybrid and legacy models."""
    inference_result = predict_glaucoma_file(image_path)
    vf = inference_result.get("vessel_features") or {}
    print(f"[ReportDebug] vessel_features: {vf}")
    print(
        f"[ReportDebug] prediction: {inference_result.get('prediction')} "
        f"confidence: {inference_result.get('confidence')}"
    )
    print(f"[ReportDebug] model_mode: {inference_result.get('model_mode')}")
    report = build_glaucoma_report(inference_result, backend=backend)
    patient_info = patient_info or {}

    return {
        "prediction": inference_result.get("prediction", "Unknown"),
        "confidence": inference_result.get("confidence", 0.0),
        "risk_level": inference_result.get("risk_level"),
        "probabilities": inference_result.get("probabilities", {}),
        "vessel_features": inference_result.get("vessel_features", {}),
        "vessel_source": inference_result.get("vessel_source", "estimated"),
        "report": report.to_dict(),
        "patient_info": {
            "patient_id": patient_info.get("patient_id", ""),
            "eye_side": patient_info.get("eye_side", ""),
        },
        "gradcam_b64": inference_result.get("gradcam_b64"),
        "overlay_b64": inference_result.get("overlay_b64"),
        "gradcam_focus": inference_result.get("gradcam_focus"),
        "model_mode": inference_result.get("model_mode"),
    }
