from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)

try:
    import medgemma_engine  # type: ignore[import-not-found]
    from medgemma_engine import generate as medgemma_generate  # type: ignore[import-not-found]
    _MEDGEMMA_IMPORT_ERROR = None
except ImportError as exc:
    medgemma_engine = None
    medgemma_generate = None
    _MEDGEMMA_IMPORT_ERROR = str(exc)


FALLBACK_REPORT_GENERIC = (
    "1. Findings\n"
    "Narrative generation was unavailable. Review the extracted optic disc and vessel biomarkers shown in the analysis panel.\n\n"
    "2. Interpretation\n"
    "A free-text interpretation could not be generated, so no additional clinical conclusion is being added beyond the measured features.\n\n"
    "3. Recommendation\n"
    "Correlate the extracted features with optic disc examination, intraocular pressure, OCT RNFL assessment, and visual field testing.\n\n"
    "4. Disclaimer\n"
    "This is an AI-assisted screening support output and must not replace ophthalmologist judgment."
)

_MEDGEMMA_WARM_LOCK = threading.Lock()
_MEDGEMMA_WARM = False


@dataclass
class GlaucomaReport:
    findings: str
    interpretation: str
    recommendation: str
    disclaimer: str
    grounding_score: str
    generation_time: float
    raw_text: str

    def to_dict(self) -> dict:
        return {
            "findings": self.findings,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
            "disclaimer": self.disclaimer,
            "grounding_score": self.grounding_score,
            "generation_time": self.generation_time,
            "raw_text": self.raw_text,
        }


def _ensure_medgemma_loaded() -> None:
    global _MEDGEMMA_WARM
    if medgemma_engine is None:
        raise RuntimeError(
            f"MedGemma engine unavailable: {_MEDGEMMA_IMPORT_ERROR or 'not installed'}"
        )
    if _MEDGEMMA_WARM or getattr(medgemma_engine, "_model", None) is not None:
        _MEDGEMMA_WARM = True
        return

    with _MEDGEMMA_WARM_LOCK:
        if _MEDGEMMA_WARM:
            return
        LOGGER.info("Warming MedGemma model for Glaucoma report generation")
        medgemma_engine._load_model()
        _MEDGEMMA_WARM = True
        LOGGER.info("MedGemma model warm and cached for Glaucoma reports")


def _feature_float(vessel_features: dict, key: str, default: float = 0.0) -> float:
    return float(vessel_features.get(key, default) or default)


def _clean_markdown_artifacts(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""

    cleaned = normalized
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"(?m)^\s*\d+\)\s+", "", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _build_feature_based_fallback_text(
    *,
    vessel_features: dict,
    vessel_source: str | None,
    predicted_class: str,
    confidence: float,
) -> str:
    cdr = _feature_float(vessel_features, "cdr")
    vessel_density = _feature_float(vessel_features, "vessel_density")
    fractal_dimension = _feature_float(vessel_features, "fractal_dimension")
    mean_tortuosity = _feature_float(vessel_features, "mean_tortuosity")
    thin_vessel_ratio = _feature_float(vessel_features, "thin_vessel_ratio")
    disc_region_density = _feature_float(vessel_features, "disc_region_density")
    cdr_interpretation = vessel_features.get("cdr_interpretation", "not available")
    risk_level = vessel_features.get("risk_level", "unknown")
    source_note = "segmentation-derived" if vessel_source == "segmentation" else "estimated from the image"

    return (
        "1. Findings\n"
        f"Quantitative optic disc and vascular analysis was completed using {source_note} vessel features. "
        f"The extracted cup-to-disc ratio is {cdr:.2f}, interpreted as {cdr_interpretation}. "
        f"Vessel density is {vessel_density:.2f}, disc region density is {disc_region_density:.2f}, "
        f"fractal dimension is {fractal_dimension:.2f}, mean tortuosity is {mean_tortuosity:.4f}, "
        f"and thin vessel ratio is {thin_vessel_ratio:.2f}.\n\n"
        "2. Interpretation\n"
        f"The narrative fallback is being kept strictly grounded in the extracted biomarkers. "
        f"The feature pattern suggests a {risk_level} structural/vascular risk profile that should be interpreted alongside the disc appearance and full clinical examination. "
        f"Supporting classifier output: {predicted_class} at {confidence:.1f}% confidence.\n\n"
        "3. Recommendation\n"
        "Correlate these extracted features with slit-lamp fundus examination, intraocular pressure, OCT RNFL analysis, and visual field testing before making management decisions.\n\n"
        "4. Disclaimer\n"
        "This is an AI-assisted screening support output derived from image features and is not a standalone diagnosis."
    )


def _build_repair_prompt(
    *,
    original_text: str,
    predicted_class: str,
    confidence: float,
    vessel_features: dict,
    vessel_source: str | None,
    gradcam_focus: str | None,
    risk_level: str | None,
) -> str:
    cdr = _feature_float(vessel_features, "cdr")
    vessel_density = _feature_float(vessel_features, "vessel_density")
    disc_region_density = _feature_float(vessel_features, "disc_region_density")
    fractal_dimension = _feature_float(vessel_features, "fractal_dimension")
    mean_tortuosity = _feature_float(vessel_features, "mean_tortuosity")
    thin_vessel_ratio = _feature_float(vessel_features, "thin_vessel_ratio")
    cdr_interpretation = vessel_features.get("cdr_interpretation", "not available")

    return (
        "Rewrite the following glaucoma screening narrative into a stronger clinician-facing report.\n"
        "Write in plain sentences only.\n"
        "Do NOT use markdown, asterisks, bold, bullet points, or numbered sub-lists inside the section bodies.\n"
        "The rewrite must be grounded in extracted image features first, with classifier output only as secondary support.\n"
        "Do not use the headings Summary, Classification Detail, or Review Note.\n"
        "Do not say 'based on the provided information'.\n"
        "Do not start by naming the predicted class.\n"
        "Do not make absolute claims such as 'no signs of glaucoma' unless strongly justified.\n"
        "Write exactly four numbered sections:\n"
        "1. Findings\n"
        "2. Interpretation\n"
        "3. Recommendation\n"
        "4. Disclaimer\n\n"
        "Required grounded values to mention explicitly:\n"
        f"- CDR: {cdr:.3f}\n"
        f"- CDR interpretation: {cdr_interpretation}\n"
        f"- Vessel density: {vessel_density:.3f}\n"
        f"- Disc region density: {disc_region_density:.3f}\n"
        f"- Fractal dimension: {fractal_dimension:.3f}\n"
        f"- Mean tortuosity: {mean_tortuosity:.4f}\n"
        f"- Thin vessel ratio: {thin_vessel_ratio:.3f}\n"
        f"- Feature-derived risk level: {risk_level or 'unknown'}\n"
        f"- Vessel feature source: {vessel_source or 'unknown'}\n"
        f"- Supporting classifier output: {predicted_class} ({confidence:.2f}%)\n"
        f"- Grad-CAM focus context: {gradcam_focus or 'not available'}\n\n"
        "Original low-quality report to rewrite:\n"
        f"{original_text.strip()}"
    )


def build_grounded_prompt(
    *,
    predicted_class: str,
    confidence: float,
    vessel_features: dict,
    vessel_source: str | None,
    gradcam_focus: str | None,
    risk_level: str | None,
    needs_review: bool,
    review_reason: str | None,
) -> str:
    cdr = _feature_float(vessel_features, "cdr")
    cdr_interpretation = vessel_features.get("cdr_interpretation", "not available")
    vessel_density = _feature_float(vessel_features, "vessel_density")
    disc_region_density = _feature_float(vessel_features, "disc_region_density")
    fractal_dimension = _feature_float(vessel_features, "fractal_dimension")
    mean_tortuosity = _feature_float(vessel_features, "mean_tortuosity")
    thin_vessel_ratio = _feature_float(vessel_features, "thin_vessel_ratio")
    vessel_area_ratio = _feature_float(vessel_features, "vessel_area_ratio")

    return (
        "You are a clinical ophthalmology AI assistant.\n"
        "Write in plain sentences only. Do NOT use markdown, asterisks, bold, bullet points, or numbered sub-lists in the section bodies.\n"
        "Use plain paragraph text only after each heading.\n\n"
        "Base the report primarily on extracted optic disc and vascular features, not on the binary classifier label.\n"
        "Treat the classifier output only as secondary supporting context.\n"
        "Use only the grounded numeric facts below and do not invent findings.\n"
        "Avoid generic filler phrases such as 'based on the provided information' or overly certain claims such as 'no signs of glaucoma' unless the feature values directly justify cautious wording.\n"
        "Write exactly four numbered sections titled:\n"
        "1. Findings\n"
        "2. Interpretation\n"
        "3. Recommendation\n"
        "4. Disclaimer\n\n"
        "Grounded feature extraction summary:\n"
        f"- Cup-to-disc ratio (CDR): {cdr:.3f}\n"
        "- CDR reference bands: normal < 0.5, suspect 0.5-0.6, concerning > 0.6\n"
        f"- CDR interpretation: {cdr_interpretation}\n"
        f"- Vessel density: {vessel_density:.3f}\n"
        f"- Disc region density: {disc_region_density:.3f}\n"
        f"- Fractal dimension: {fractal_dimension:.3f}\n"
        f"- Mean tortuosity: {mean_tortuosity:.4f}\n"
        f"- Thin vessel ratio: {thin_vessel_ratio:.3f}\n"
        f"- Vessel area ratio: {vessel_area_ratio:.3f}\n"
        f"- Feature-derived risk level: {risk_level or 'unknown'}\n"
        f"- Vessel feature source: {vessel_source or 'unknown'}\n"
        f"- Grad-CAM focus context: {gradcam_focus or 'not available'}\n"
        "Supporting classifier output (secondary only):\n"
        f"- Predicted class: {predicted_class}\n"
        f"- Confidence: {confidence:.2f}%\n"
        f"- Needs review: {'Yes' if needs_review else 'No'}\n"
        f"- Review reason: {review_reason or 'None'}\n\n"
        "Requirements:\n"
        "- In Findings, explicitly mention the numeric values for CDR, vessel density, fractal dimension, mean tortuosity, and thin vessel ratio.\n"
        "- In Interpretation, explain the clinical meaning of the feature pattern using cautious language.\n"
        "- The first sentence of the report must not mention the predicted class.\n"
        "- Mention the classifier output at most once, and clearly label it as supporting context.\n"
        "- If vessel source is 'estimated', explicitly say the vessel features are approximate.\n"
        "- Keep the disclaimer explicit that this is AI-assisted screening support and not a diagnosis.\n"
    )


def _is_low_quality_report(report_text: str, vessel_features: dict | None) -> bool:
    if not report_text:
        return True

    text = _clean_markdown_artifacts(report_text)
    lowered = text.lower()
    banned_phrases = [
        "here's a structured clinical report based on the provided information",
        "classification detail",
        "review note",
        "summary",
        "the image shows evidence of glaucoma",
        "the eye shows no signs of glaucoma",
        "the model has high confidence",
    ]
    if any(phrase in lowered for phrase in banned_phrases):
        return True

    if vessel_features:
        required_values = [
            str(round(_feature_float(vessel_features, "cdr"), 2)),
            str(round(_feature_float(vessel_features, "vessel_density"), 2)),
            str(round(_feature_float(vessel_features, "fractal_dimension"), 2)),
        ]
        mentioned_values = sum(value in text for value in required_values)
        if mentioned_values < 2:
            return True

        metric_terms = ["cup-to-disc", "cdr", "vessel density", "fractal", "tortuosity", "thin vessel"]
        if sum(term in lowered for term in metric_terms) < 2:
            return True

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if "glaucoma" in first_line or "normal" in first_line:
            return True

    return False


def validate_grounding(report_text: str, *, predicted_class: str, vessel_features: dict | None) -> str:
    if not vessel_features:
        return "0/0 (legacy prompt)"

    cleaned_report = _clean_markdown_artifacts(report_text)
    text = cleaned_report.lower()
    checks = {
        "cdr": str(round(_feature_float(vessel_features, "cdr"), 2)) in cleaned_report,
        "vascular_metric": any(
            str(round(_feature_float(vessel_features, metric), 2)) in cleaned_report
            for metric in ["vessel_density", "fractal_dimension", "thin_vessel_ratio", "disc_region_density"]
        ),
        "feature_language": any(keyword in text for keyword in ["cup-to-disc", "cdr", "vessel density", "fractal", "tortuosity", "thin vessel"]),
        "recommendation": any(keyword in text for keyword in ["refer", "review", "monitor", "oct", "visual field", "iop", "correlate"]),
        "disclaimer": any(keyword in text for keyword in ["ophthalmologist", "ai-assisted", "screening support", "not a diagnosis", "clinical judgment"]),
    }
    passed = sum(checks.values())
    total = len(checks)
    return f"{passed}/{total} ({int((passed / total) * 100)}%)"


def parse_report_sections(raw_text: str) -> dict:
    sections = {
        "findings": "",
        "interpretation": "",
        "recommendation": "",
        "disclaimer": "",
    }

    cleaned_text = _clean_markdown_artifacts(raw_text)
    if not cleaned_text:
        return sections

    heading_aliases = {
        "findings": "findings",
        "finding": "findings",
        "summary": "findings",
        "interpretation": "interpretation",
        "interpret": "interpretation",
        "assessment": "interpretation",
        "classification": "interpretation",
        "recommendation": "recommendation",
        "recommend": "recommendation",
        "plan": "recommendation",
        "disclaimer": "disclaimer",
        "caution": "disclaimer",
        "limitation": "disclaimer",
    }

    current_key = None
    saw_heading = False

    for raw_line in cleaned_text.splitlines():
      line = raw_line.strip()
      if not line:
          continue

      normalized_heading = re.sub(r"^[\W_]*\d*\.?\s*", "", line).strip().rstrip(":").lower()
      if normalized_heading in heading_aliases:
          current_key = heading_aliases[normalized_heading]
          saw_heading = True
          continue

      if current_key:
          sections[current_key] = f"{sections[current_key]} {line}".strip()

    if saw_heading:
        return sections

    blocks = [segment.strip() for segment in re.split(r"\n\s*\n", cleaned_text) if segment.strip()]
    if blocks:
        if len(blocks) > 0:
            sections["findings"] = blocks[0]
        if len(blocks) > 1:
            sections["interpretation"] = blocks[1]
        if len(blocks) > 2:
            sections["recommendation"] = blocks[2]
        if len(blocks) > 3:
            sections["disclaimer"] = blocks[3]

    if not any(sections.values()):
        sections["findings"] = cleaned_text
        if len(blocks) > 1:
            sections["interpretation"] = blocks[1]

    return sections


def generate_glaucoma_report_with_meta(
    patient_name: str,
    mri_number: str,
    uid: str,
    eye_side: str,
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    needs_review: bool,
    review_reason: str | None,
    vessel_features: dict | None = None,
    vessel_source: str | None = None,
    gradcam_focus: str | None = None,
    risk_level: str | None = None,
) -> tuple[str, str | None]:
    del patient_name, mri_number, uid, eye_side, probabilities

    if not vessel_features:
        return FALLBACK_REPORT_GENERIC, "feature_data_unavailable"

    try:
        _ensure_medgemma_loaded()
    except Exception as exc:
        LOGGER.info("Glaucoma report fallback triggered before generation: %s", exc)
        return (
            _build_feature_based_fallback_text(
                vessel_features=vessel_features,
                vessel_source=vessel_source,
                predicted_class=predicted_class,
                confidence=confidence,
            ),
            "report_unavailable",
        )

    prompt = build_grounded_prompt(
        predicted_class=predicted_class,
        confidence=confidence,
        vessel_features=vessel_features,
        vessel_source=vessel_source,
        gradcam_focus=gradcam_focus,
        risk_level=risk_level,
        needs_review=needs_review,
        review_reason=review_reason,
    )

    LOGGER.info("Glaucoma report prompt length=%s", len(prompt))

    try:
        report_text = _clean_markdown_artifacts(medgemma_generate(prompt).strip())
        if _is_low_quality_report(report_text, vessel_features):
            LOGGER.info("Glaucoma report failed quality gate; requesting rewrite.")
            repair_prompt = _build_repair_prompt(
                original_text=report_text,
                predicted_class=predicted_class,
                confidence=confidence,
                vessel_features=vessel_features,
                vessel_source=vessel_source,
                gradcam_focus=gradcam_focus,
                risk_level=risk_level,
            )
            report_text = _clean_markdown_artifacts(medgemma_generate(repair_prompt).strip())

        if _is_low_quality_report(report_text, vessel_features):
            LOGGER.info("Glaucoma report rewrite still low quality; using feature-based fallback.")
            return (
                _build_feature_based_fallback_text(
                    vessel_features=vessel_features,
                    vessel_source=vessel_source,
                    predicted_class=predicted_class,
                    confidence=confidence,
                ),
                "low_quality_replaced",
            )

        LOGGER.info("Glaucoma report response length=%s", len(report_text))
        return report_text, None
    except Exception as exc:
        LOGGER.info("Glaucoma report fallback triggered due to error: %s", exc)
        if vessel_features:
            return (
                _build_feature_based_fallback_text(
                    vessel_features=vessel_features,
                    vessel_source=vessel_source,
                    predicted_class=predicted_class,
                    confidence=confidence,
                ),
                "report_unavailable",
            )
        return FALLBACK_REPORT_GENERIC, "report_unavailable"


def generate_glaucoma_report(
    inference_result: dict,
    backend: str = "local",
    api_key: str = None,
    model_id: str = "google/medgemma-4b-it",
) -> GlaucomaReport:
    del backend, api_key, model_id

    started_at = time.time()
    raw_text, _warning = generate_glaucoma_report_with_meta(
        patient_name="",
        mri_number="",
        uid="",
        eye_side="UNKNOWN",
        predicted_class=inference_result.get("prediction") or inference_result.get("predicted_class", "Unknown"),
        confidence=float(inference_result.get("confidence", 0.0) or 0.0),
        probabilities=inference_result.get("probabilities", {}),
        needs_review=bool(inference_result.get("needs_review", False)),
        review_reason=inference_result.get("review_reason"),
        vessel_features=inference_result.get("vessel_features"),
        vessel_source=inference_result.get("vessel_source"),
        gradcam_focus=inference_result.get("gradcam_focus"),
        risk_level=inference_result.get("risk_level"),
    )

    sections = parse_report_sections(raw_text)
    grounding_score = validate_grounding(
        raw_text,
        predicted_class=inference_result.get("prediction") or inference_result.get("predicted_class", "Unknown"),
        vessel_features=inference_result.get("vessel_features"),
    )

    return GlaucomaReport(
        findings=sections["findings"],
        interpretation=sections["interpretation"],
        recommendation=sections["recommendation"],
        disclaimer=sections["disclaimer"],
        grounding_score=grounding_score,
        generation_time=round(time.time() - started_at, 2),
        raw_text=raw_text,
    )
