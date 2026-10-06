from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from report_text_engine import V7Result, generate_report_texts

# ---------------------------------------------------------------------------
# Resolve project root for checkpoints and runtime assets
# ---------------------------------------------------------------------------

BACKEND_DIR  = Path(__file__).resolve().parent
WEB_DIR      = BACKEND_DIR.parent
PROJECT_ROOT = WEB_DIR
LEGACY_PROJECT_ROOT = WEB_DIR.parent

try:
    from model.inference_multimodal import CataractInferenceEngineV7, InferenceConfig
    _V7_AVAILABLE = True
except ImportError as _v7_err:
    _V7_AVAILABLE = False
    _V7_IMPORT_ERROR = str(_v7_err)

try:
    from data_pipeline.crop_roi_v3 import RedGlowCropper
    _ROI_CROPPERS_AVAILABLE = True
except ImportError as _roi_err:
    _ROI_CROPPERS_AVAILABLE = False
    _ROI_CROPPERS_IMPORT_ERROR = str(_roi_err)

# ---------------------------------------------------------------------------
# Grad-CAM — optional import (non-fatal if unavailable)
# ---------------------------------------------------------------------------

try:
    from gradcam_engine import GradCAMAnalyser
    _GRADCAM_AVAILABLE = True
except ImportError as _gc_err:
    _GRADCAM_AVAILABLE = False
    _GRADCAM_IMPORT_ERROR = str(_gc_err)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECOMMENDATION_MAP: dict[str, str] = {
    "NS1": "Early cataract. Monitor annually.",
    "NS2": "Mild cataract. Review in 6 months.",
    "NS3": "Moderate cataract. Consider surgical referral.",
    "NS4": "Severe cataract. Urgent surgical referral recommended.",
}

SEVERITY_MAP: dict[str, str] = {
    "NS1": "Mild",
    "NS2": "Mild",
    "NS3": "Moderate",
    "NS4": "Severe",
}


@dataclass
class ClassificationArtifacts:
    result: dict[str, Any]
    analyzed_at: datetime


class ClassifierService:
    """
    Wraps the V7 multimodal inference engine.

    The engine is loaded once at startup (lazy, on first classify call) so the
    FastAPI worker starts quickly even if the GPU is not yet warmed up.

    Grad-CAM is initialised alongside the model and runs automatically on
    every classify_exam() call. If Grad-CAM fails for any reason the
    classification result is still returned — Grad-CAM is non-fatal.
    """

    def __init__(self) -> None:
        self._engine: CataractInferenceEngineV7 | None = None
        self._gradcam_ant: GradCAMAnalyser | None = None
        self._gradcam_rg: GradCAMAnalyser | None = None
        self._gradcam_sl: GradCAMAnalyser | None = None
        self._red_glow_cropper = RedGlowCropper() if _ROI_CROPPERS_AVAILABLE else None
        self._checkpoint_lookup_error: str | None = None
        self._checkpoint = self._resolve_checkpoint()
        self._yolo_weights = self._resolve_yolo_weights()
        self._startup_error = self._build_startup_error()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def availability_error(self) -> str | None:
        """Explain why cataract inference is unavailable without loading a model."""
        return self._startup_error

    def classify_exam(
        self,
        anterior_segment_bytes: bytes,
        red_glow_bytes: bytes | None = None,
        slit_lamp_bytes: bytes | None = None,
        case_id: str = "UNKNOWN",
        filename: str | None = None,
    ) -> ClassificationArtifacts:
        self._ensure_ready()

        tmp_dir = tempfile.mkdtemp(prefix="ophthalmic_imaging_")
        try:
            ant_path = self._save_bytes(tmp_dir, "anterior.jpg", anterior_segment_bytes)
            rg_path  = self._save_bytes(tmp_dir, "red_glow.jpg", red_glow_bytes)  if red_glow_bytes  else None
            sl_path  = self._save_bytes(tmp_dir, "slit_lamp.jpg", slit_lamp_bytes) if slit_lamp_bytes else None

            rg_cropped_path = self._crop_optional_modality(
                modality_path=rg_path,
                cropper=self._red_glow_cropper,
                output_name="red_glow_cropped.png",
                label="Red glow",
            )
            # Fall back to anterior image when optional modalities are missing
            effective_rg = rg_cropped_path or rg_path or ant_path
            effective_sl = sl_path or ant_path

            v7_result = self._engine.predict_case(
                ant_path=str(ant_path),
                rg_path=str(effective_rg),
                sl_path=str(effective_sl),
                case_id=case_id or "UNKNOWN",
            )

            analyzed_at = datetime.now()

            # Build base64 image strings for the API response
            anterior_b64 = self._file_to_base64(ant_path)
            rg_b64       = self._file_to_base64(rg_path)  if rg_path  else None
            sl_b64       = self._file_to_base64(sl_path)  if sl_path  else None
            rg_crop_b64  = self._file_to_base64(rg_cropped_path) if rg_cropped_path else None
            sl_crop_b64  = None

            # Cropped pupil image saved by V7 engine
            crop_b64: str | None = None
            saved = v7_result.get("saved_files", {})
            ant_crop_path = saved.get("anterior_crop")
            if ant_crop_path and Path(ant_crop_path).exists():
                crop_b64 = self._file_to_base64(Path(ant_crop_path))

            grade: str            = v7_result.get("grade", "Unknown")
            confidence: float     = float(v7_result.get("confidence", 0.0))
            probs: dict           = v7_result.get("probabilities", {})
            attention: dict       = v7_result.get("attention", {})
            needs_review: bool    = bool(v7_result.get("needs_review", False))
            review_reason: str | None = v7_result.get("review_reason")
            bbox                  = v7_result.get("bbox_xywh")

            recommendation = RECOMMENDATION_MAP.get(
                grade, "Grade unavailable. Please review clinically."
            )

            payload = {
                "case_id":                  case_id or "UNKNOWN",
                "detected":                 True,
                "grade":                    grade,
                "confidence":               round(confidence, 4),
                "confidence_pct":           f"{round(confidence * 100)}%",
                "probabilities":            probs,
                "needs_review":             needs_review,
                "review_reason":            review_reason,
                "attention":                attention,
                "bbox":                     list(bbox) if bbox else None,
                "quality_score":            None,
                "crop_image_base64":        crop_b64,
                "red_glow_crop_base64":     rg_crop_b64,
                "slit_lamp_crop_base64":    sl_crop_b64,
                "pupil_crop_applied":       bool(v7_result.get("pupil_crop_applied", False)),
                "anterior_segment_base64":  anterior_b64,
                "red_glow_base64":          rg_b64,
                "slit_lamp_base64":         sl_b64,
                "recommendation":           recommendation,
                "severity":                 SEVERITY_MAP.get(grade, "Unknown"),
                "model_info":               "V7_best | YOLOv8 pupil crop",
                # Grad-CAM fields — populated below, default None
                "gradcam_heatmap_base64":   None,
                "red_glow_gradcam_base64":  None,
                "slit_lamp_gradcam_base64": None,
                "gradcam_region_info":      {},
                "report_texts":             {},
            }
            if filename:
                payload["filename"] = filename

            # ── Grad-CAM ──────────────────────────────────────────────
            self._run_gradcam(
                payload   = payload,
                ant_path  = ant_crop_path if ant_crop_path and Path(ant_crop_path).exists()
                            else str(ant_path),
                rg_path   = str(effective_rg),
                sl_path   = str(effective_sl),
                pred_idx  = v7_result.get("grade_index", 0),
            )
            # NOTE: MedGemma report text is NOT generated here.
            # It is generated on-demand when /report or /report-batch is called,
            # so V7 finishes instantly and the UI shows results without waiting.

            return ClassificationArtifacts(result=payload, analyzed_at=analyzed_at)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def classify_image(
        self,
        image_bytes: bytes,
        filename: str | None = None,
    ) -> ClassificationArtifacts:
        """Single-image convenience wrapper (anterior only)."""
        return self.classify_exam(
            anterior_segment_bytes=image_bytes,
            red_glow_bytes=None,
            slit_lamp_bytes=None,
            case_id="UNKNOWN",
            filename=filename,
        )

    def classify_batch(
        self, files: list[tuple[str, bytes]]
    ) -> list[ClassificationArtifacts]:
        artifacts: list[ClassificationArtifacts] = []
        for filename, image_bytes in files:
            try:
                artifacts.append(self.classify_image(image_bytes, filename=filename))
            except Exception as exc:
                artifacts.append(
                    ClassificationArtifacts(
                        result=self._error_payload(filename, str(exc)),
                        analyzed_at=datetime.now(),
                    )
                )
        return artifacts

    # ------------------------------------------------------------------
    # Grad-CAM runner (non-fatal)
    # ------------------------------------------------------------------

    def _run_gradcam(
        self,
        payload:  dict,
        ant_path: str,
        rg_path:  str,
        sl_path:  str,
        pred_idx: int,
    ) -> None:
        """
        Run Grad-CAM on the anterior image and attach results to payload.

        Populates:
            payload["gradcam_heatmap_base64"]  — PNG overlay as base64 string
            payload["gradcam_region_info"]     — dict of region activation stats
                                                 (used by report_text_engine.py
                                                  to build the MedGemma prompt)

        Any exception is caught and logged — Grad-CAM failure never breaks
        the classification response.
        """
        if self._gradcam_ant is None or self._gradcam_rg is None or self._gradcam_sl is None:
            return

        try:
            import torch
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            device = self._gradcam_ant.device

            def _load_tensor(path: str, size: int) -> torch.Tensor:
                img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
                tfm = A.Compose([A.Resize(size, size), A.Normalize(**NORM), ToTensorV2()])
                return tfm(image=img)["image"].unsqueeze(0).to(device)

            # Match the exact sizes used during V7 inference
            ant_t = _load_tensor(ant_path, 260)
            rg_t  = _load_tensor(rg_path,  224)
            sl_t  = _load_tensor(sl_path,  260)

            ant_rgb = np.array(Image.open(ant_path).convert("RGB"), dtype=np.uint8)
            rg_rgb = np.array(Image.open(rg_path).convert("RGB"), dtype=np.uint8)
            sl_rgb = np.array(Image.open(sl_path).convert("RGB"), dtype=np.uint8)

            ant_overlay_rgb, region_info = self._gradcam_ant.run(
                ant_tensor=ant_t,
                rg_tensor=rg_t,
                sl_tensor=sl_t,
                image_rgb=ant_rgb,
                pred_class_idx=pred_idx,
            )
            rg_overlay_rgb, _ = self._gradcam_rg.run(
                ant_tensor=ant_t,
                rg_tensor=rg_t,
                sl_tensor=sl_t,
                image_rgb=rg_rgb,
                pred_class_idx=pred_idx,
            )
            sl_overlay_rgb, _ = self._gradcam_sl.run(
                ant_tensor=ant_t,
                rg_tensor=rg_t,
                sl_tensor=sl_t,
                image_rgb=sl_rgb,
                pred_class_idx=pred_idx,
            )

            buf = io.BytesIO()
            Image.fromarray(ant_overlay_rgb).save(buf, format="PNG")
            payload["gradcam_heatmap_base64"] = base64.b64encode(buf.getvalue()).decode()
            payload["gradcam_region_info"] = region_info
            buf = io.BytesIO()
            Image.fromarray(rg_overlay_rgb).save(buf, format="PNG")
            payload["red_glow_gradcam_base64"] = base64.b64encode(buf.getvalue()).decode()
            buf = io.BytesIO()
            Image.fromarray(sl_overlay_rgb).save(buf, format="PNG")
            payload["slit_lamp_gradcam_base64"] = base64.b64encode(buf.getvalue()).decode()

            top = region_info.get("top_regions", [])
            print(f"[Classifier] Grad-CAM OK — top regions: {top}, "
                  f"spread: {region_info.get('activation_spread')}, "
                  f"intensity: {region_info.get('activation_intensity')}")

        except Exception as exc:
            # Non-fatal — classification result is unchanged
            print(f"[Classifier] Grad-CAM failed (non-fatal): {exc}")
            payload["gradcam_heatmap_base64"] = None
            payload["red_glow_gradcam_base64"] = None
            payload["slit_lamp_gradcam_base64"] = None
            payload["gradcam_region_info"]    = {}

    def _generate_report_texts(self, payload: dict[str, Any]) -> None:
        """Generate report narrative during classification so UI/DB/PDF share one text set."""
        grade = payload.get("grade")
        if grade not in {"NS1", "NS2", "NS3", "NS4"}:
            payload["report_texts"] = {}
            return

        try:
            attention = payload.get("attention") or {}
            report_input = V7Result(
                grade=grade,
                confidence=float(payload.get("confidence") or 0.0),
                probabilities={k: float(v) for k, v in (payload.get("probabilities") or {}).items()},
                attention={
                    "anterior": float(attention.get("anterior", 0.0)),
                    "red_glow": float(attention.get("red_glow", 0.0)),
                    "slit_lamp": float(attention.get("slit_lamp", 0.0)),
                },
                case_id=payload.get("case_id"),
                clinician=payload.get("graded_by"),
                gradcam_region_info=payload.get("gradcam_region_info") or {},
                raw_result=payload,
            )
            payload["report_texts"] = generate_report_texts(report_input)
        except Exception as exc:
            print(f"[Classifier] Report text generation failed (non-fatal): {exc}")
            payload["report_texts"] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._startup_error:
            raise RuntimeError(self._startup_error)

        if self._engine is None:
            import torch
            self._engine = CataractInferenceEngineV7(
                InferenceConfig(
                    checkpoint   = str(self._checkpoint),
                    yolo_weights = str(self._yolo_weights) if self._yolo_weights else None,
                    device       = "cuda" if _cuda_available() else "cpu",
                    use_tta      = True,
                    temperature  = 1.1,
                    ce_weight    = 0.65,
                    binary_boost = 0.05,
                    output_dir   = str(PROJECT_ROOT / "output"),
                )
            )

            # Initialise Grad-CAM right after the model is loaded
            if _GRADCAM_AVAILABLE:
                try:
                    device = torch.device("cuda" if _cuda_available() else "cpu")
                    self._gradcam_ant = GradCAMAnalyser(
                        model=self._engine.model,
                        device=device,
                        target_layer_name="backbone_ant.features",
                        modality="anterior",
                    )
                    self._gradcam_rg = GradCAMAnalyser(
                        model=self._engine.model,
                        device=device,
                        target_layer_name="backbone_rg.stages",
                        modality="red_glow",
                    )
                    self._gradcam_sl = GradCAMAnalyser(
                        model=self._engine.model,
                        device=device,
                        target_layer_name="backbone_sl.features",
                        modality="slit_lamp",
                    )
                    print("[Classifier] Grad-CAM analyser ready.")
                except Exception as exc:
                    print(f"[Classifier] Grad-CAM init failed (non-fatal): {exc}")
                    self._gradcam_ant = None
                    self._gradcam_rg = None
                    self._gradcam_sl = None
            else:
                print(f"[Classifier] Grad-CAM not available: {_GRADCAM_IMPORT_ERROR}")

    def _build_startup_error(self) -> str | None:
        if not _V7_AVAILABLE:
            return (
                f"V7 inference engine could not be imported: {_V7_IMPORT_ERROR}. "
                "Ensure all model dependencies are installed."
            )
        if self._checkpoint is None:
            return self._checkpoint_lookup_error or (
                "Cataract classification is unavailable because no trained V7 "
                "checkpoint was found. Provide a real checkpoint by setting "
                "CATARACT_CHECKPOINT_PATH, or place best_acc_model.pth or "
                "best_qwk_model.pth under ophthalmic-web/checkpoints/. Placeholder "
                "or randomly initialized weights are not accepted."
            )
        return None

    def _resolve_checkpoint(self) -> Path | None:
        configured_path = os.getenv("CATARACT_CHECKPOINT_PATH", "").strip()
        if configured_path:
            configured = Path(configured_path).expanduser()
            if not configured.is_absolute():
                configured = PROJECT_ROOT / configured
            if configured.is_file():
                return configured
            self._checkpoint_lookup_error = (
                "Cataract classification is unavailable: CATARACT_CHECKPOINT_PATH "
                f"does not point to a file ({configured}). Supply the trained "
                "best_acc_model.pth or best_qwk_model.pth asset; do not use fake weights."
            )
            return None

        candidates = [
            PROJECT_ROOT / "checkpoints" / "best_acc_model.pth",
            PROJECT_ROOT / "checkpoints" / "best_qwk_model.pth",
            PROJECT_ROOT / "checkpoints_v7" / "best_qwk_model.pth",
            LEGACY_PROJECT_ROOT / "checkpoints" / "best_acc_model.pth",
            LEGACY_PROJECT_ROOT / "checkpoints" / "best_qwk_model.pth",
            LEGACY_PROJECT_ROOT / "checkpoints_v7" / "best_qwk_model.pth",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _resolve_yolo_weights(self) -> Path | None:
        candidates = [
            PROJECT_ROOT / "pupil_crop" / "Yolo" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "detect" / "pupil_detector" / "weights" / "best.pt",
            LEGACY_PROJECT_ROOT / "pupil_crop" / "Yolo" / "weights" / "best.pt",
            LEGACY_PROJECT_ROOT / "runs" / "detect" / "pupil_detector" / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    @staticmethod
    def _save_bytes(directory: str, name: str, data: bytes) -> Path:
        path = Path(directory) / name
        path.write_bytes(data)
        return path

    @staticmethod
    def _file_to_base64(path: Path | str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        raw = p.read_bytes()
        if p.suffix.lower() == ".png":
            return base64.b64encode(raw).decode()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return base64.b64encode(raw).decode()

    @staticmethod
    def _crop_optional_modality(
        modality_path: Path | None,
        cropper: Any,
        output_name: str,
        label: str,
    ) -> Path | None:
        if modality_path is None or cropper is None:
            return None

        output_path = modality_path.parent / output_name
        try:
            result = cropper.crop(str(modality_path), str(output_path))
            if result.success and result.output_path and Path(result.output_path).exists():
                print(f"  {label} crop: OK (score={result.confidence:.2f})")
                return Path(result.output_path)
            print(f"  {label} crop: failed - {result.error_msg or result.method_used}")
        except Exception as exc:
            print(f"  {label} crop: exception - {exc}")
        return None

    @staticmethod
    def _error_payload(filename: str, error: str) -> dict[str, Any]:
        return {
            "case_id":                  "UNKNOWN",
            "filename":                 filename,
            "detected":                 False,
            "grade":                    None,
            "confidence":               None,
            "confidence_pct":           "N/A",
            "probabilities":            {},
            "needs_review":             False,
            "review_reason":            None,
            "attention":                {},
            "bbox":                     None,
            "quality_score":            None,
            "crop_image_base64":        None,
            "red_glow_crop_base64":     None,
            "slit_lamp_crop_base64":    None,
            "anterior_segment_base64":  None,
            "red_glow_base64":          None,
            "slit_lamp_base64":         None,
            "recommendation":           "Classification failed. Please review and retry.",
            "severity":                 "Unknown",
            "model_info":               "V7_best | YOLOv8 pupil crop",
            "gradcam_heatmap_base64":   None,
            "red_glow_gradcam_base64":  None,
            "slit_lamp_gradcam_base64": None,
            "gradcam_region_info":      {},
            "error":                    error,
        }


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
