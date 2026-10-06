#!/usr/bin/env python3
"""
Helpers for generating clinician-facing cataract reports with V7 + MedGemma.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from inference_multimodal import CataractInferenceEngineV7


V7_MODEL_NAME = "V7_best_qwk"
MEDGEMMA_MODEL_NAME = "MedGemma_4B_multimodal"
REQUIRED_FINDING_KEYS = ("anterior_segment", "red_glow", "slit_lamp", "conclusion")

GRADE_SEVERITY_LABELS = {
    "NS1": "Mild",
    "NS2": "Moderate",
    "NS3": "Moderately Severe",
    "NS4": "Severe",
}

FALLBACK_FINDING_LIBRARY = {
    "NS1": {
        "anterior_segment": (
            "The lens nucleus shows mild yellowing with early nuclear opacity. "
            "Changes remain limited and are compatible with NS1 nuclear sclerosis."
        ),
        "red_glow": (
            "The red reflex remains largely preserved with only mild reduction in glow. "
            "Light transmission appears only minimally affected."
        ),
        "slit_lamp": (
            "The slit beam shows mild scattering through the nuclear zone without marked density. "
            "This appearance is consistent with early nuclear sclerosis."
        ),
        "conclusion": (
            "The combined appearance supports mild nuclear sclerosis, and NS1 grading is appropriate."
        ),
    },
    "NS2": {
        "anterior_segment": (
            "The lens nucleus shows clear yellow discoloration with moderate nuclear opacity visible "
            "through the pupil. The appearance is compatible with intermediate brunescence."
        ),
        "red_glow": (
            "The red reflex is reduced compared with a clear lens but remains appreciable. "
            "This suggests moderate nuclear density."
        ),
        "slit_lamp": (
            "The slit beam demonstrates broader scattering through the nuclear zone with moderate "
            "backscatter intensity. These findings fit NS2 nuclear sclerosis."
        ),
        "conclusion": (
            "The combination of moderate brunescence, partial red reflex reduction, and moderate slit beam "
            "scatter supports NS2 nuclear sclerosis."
        ),
    },
    "NS3": {
        "anterior_segment": (
            "The lens nucleus shows distinct amber-yellow discoloration consistent with moderate-severe "
            "brunescence. The nuclear opacity is clearly visible through the undilated pupil."
        ),
        "red_glow": (
            "The red reflex is significantly diminished compared with normal. Reduced fundal glow indicates "
            "increased nuclear density blocking light transmission."
        ),
        "slit_lamp": (
            "The slit beam shows dense, bright scattering as it passes through the nuclear zone. "
            "The increased backscatter width is consistent with NS3 grading."
        ),
        "conclusion": (
            "The combination of nuclear brunescence, reduced red reflex, and dense slit beam scattering "
            "supports NS3 nuclear sclerosis."
        ),
    },
    "NS4": {
        "anterior_segment": (
            "The lens nucleus shows pronounced brunescence with dense dark yellow-brown opacity. "
            "The nucleus appears markedly sclerotic."
        ),
        "red_glow": (
            "The red reflex is severely reduced or nearly extinguished, indicating dense nuclear opacity "
            "with marked obstruction of light transmission."
        ),
        "slit_lamp": (
            "The slit beam demonstrates very dense scattering through the nucleus with intense backscatter "
            "and poor transparency. This appearance is consistent with advanced nuclear sclerosis."
        ),
        "conclusion": (
            "Marked brunescence, severe red reflex loss, and very dense slit beam scattering support NS4 "
            "nuclear sclerosis."
        ),
    },
}


@dataclass
class ReportPaths:
    output_dir: Path
    pdf_path: Path
    payload_path: Path


def ensure_image_paths_exist(paths: Dict[str, str]) -> None:
    missing = [(key, path) for key, path in paths.items() if not Path(path).exists()]
    if missing:
        joined = "; ".join(f"{key}={path}" for key, path in missing)
        raise FileNotFoundError(f"Missing required image paths: {joined}")


def infer_case_id(case_id: Optional[str], input_paths: Dict[str, str]) -> str:
    if case_id:
        return case_id
    anterior_stem = Path(input_paths["anterior_path"]).stem
    return anterior_stem.replace(" ", "_")


def rebase_dataset_path(raw_path: str, project_root: Optional[str | Path] = None) -> str:
    """
    Rebase Windows absolute dataset paths from metadata CSVs onto the current runtime.

    Example:
      C:\\...\\cataract_ns_grading_v2\\Data\\NS1\\...jpg
      -> <project_root>/Data/NS1/...jpg
    """
    candidate = Path(raw_path)
    if candidate.exists() or project_root is None:
        return str(candidate)

    project_root = Path(project_root)
    normalized = raw_path.replace("\\", "/")
    for marker in ("Data_Cropped/", "Data/"):
        idx = normalized.find(marker)
        if idx != -1:
            relative_parts = normalized[idx:].split("/")
            return str(project_root.joinpath(*relative_parts))

    return str(candidate)


def severity_label_for_grade(grade: str) -> str:
    return GRADE_SEVERITY_LABELS.get(grade, "Unknown Severity")


def build_medgemma_prompt(
    *,
    grade: str,
    severity_label: str,
    confidence: float,
    needs_review: bool,
    review_reason: Optional[str],
    strict_retry: bool = False,
) -> str:
    review_line = (
        f"Review flag: yes. Reason: {review_reason}."
        if needs_review and review_reason
        else "Review flag: no."
    )
    format_line = (
        "Return only valid JSON with exactly these keys: "
        '["anterior_segment","red_glow","slit_lamp","conclusion"]. '
        "Each value must be a single paragraph string."
    )
    if strict_retry:
        format_line += " Do not include markdown fences, comments, or any extra keys."

    return (
        "You are an ophthalmology report assistant. "
        "You will receive three images from one cataract case in this order: "
        "1) anterior segment, 2) red glow, 3) slit lamp. "
        f"The CNN grade is fixed and must not be changed: {grade} ({severity_label}). "
        f"CNN confidence: {confidence:.2%}. {review_line} "
        "Describe only visible findings from each image. Align your wording to the fixed CNN grade. "
        "Do not invent patient history, symptoms, surgical decisions, or findings that are not visible. "
        "If the images are not clearly supportive, say that the findings are limited or only partially supportive, "
        "but still do not override the CNN grade. "
        "For anterior segment, describe lens color and nuclear opacity. "
        "For red glow, describe reflex brightness or darkness. "
        "For slit lamp, describe beam scatter density and backscatter. "
        + format_line
    )


def build_medgemma_messages(
    prompt: str,
    anterior_image: Image.Image,
    red_glow_image: Image.Image,
    slit_lamp_image: Image.Image,
) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You write concise ophthalmology image findings and follow the requested JSON schema exactly.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Image 1: anterior segment"},
                {"type": "image", "image": anterior_image},
                {"type": "text", "text": "Image 2: red glow"},
                {"type": "image", "image": red_glow_image},
                {"type": "text", "text": "Image 3: slit lamp"},
                {"type": "image", "image": slit_lamp_image},
            ],
        },
    ]


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1].strip()

    raise ValueError("Could not locate JSON object in MedGemma response.")


def parse_medgemma_response(text: str) -> Dict[str, str]:
    blob = _extract_json_blob(text)
    data = json.loads(blob)
    if sorted(data.keys()) != sorted(REQUIRED_FINDING_KEYS):
        raise ValueError(f"Unexpected keys in MedGemma response: {sorted(data.keys())}")

    normalized = {}
    for key in REQUIRED_FINDING_KEYS:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty paragraph for key: {key}")
        normalized[key] = " ".join(value.strip().split())
    return normalized


def build_fallback_findings(grade: str, needs_review: bool, review_reason: Optional[str]) -> Dict[str, str]:
    fallback = dict(FALLBACK_FINDING_LIBRARY[grade])
    if needs_review:
        note = " The CNN marked this case for clinician review before relying on the report."
        fallback["conclusion"] += note
        if review_reason:
            fallback["conclusion"] += f" Review reason: {review_reason}."
    return fallback


class MedGemmaExplainer:
    """
    Thin wrapper around the Hugging Face MedGemma multimodal checkpoint.

    The model is loaded lazily so local development can still import this module
    without downloading or authenticating against Hugging Face.
    """

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 400,
        do_sample: bool = False,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _generate_once(self, messages: Sequence[Dict]) -> str:
        import torch

        self._load()
        assert self.processor is not None
        assert self.model is not None

        inputs = self.processor.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            dtype = torch.float16
            inputs = inputs.to(self.model.device, dtype=dtype)
        else:
            inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
            )
        generated_tokens = generation[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def generate_findings(
        self,
        *,
        cnn_result: Dict,
        anterior_path: str,
        red_glow_path: str,
        slit_lamp_path: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        images = {
            "anterior": Image.open(anterior_path).convert("RGB"),
            "red_glow": Image.open(red_glow_path).convert("RGB"),
            "slit_lamp": Image.open(slit_lamp_path).convert("RGB"),
        }

        attempts = []
        for strict_retry in (False, True):
            prompt = build_medgemma_prompt(
                grade=cnn_result["grade"],
                severity_label=severity_label_for_grade(cnn_result["grade"]),
                confidence=cnn_result["confidence"],
                needs_review=cnn_result["needs_review"],
                review_reason=cnn_result["review_reason"],
                strict_retry=strict_retry,
            )
            messages = build_medgemma_messages(
                prompt=prompt,
                anterior_image=images["anterior"],
                red_glow_image=images["red_glow"],
                slit_lamp_image=images["slit_lamp"],
            )
            raw_text = self._generate_once(messages)
            attempts.append({"strict_retry": strict_retry, "raw_text": raw_text})
            try:
                parsed = parse_medgemma_response(raw_text)
                return parsed, {"raw_text": raw_text, "fallback_used": False}
            except Exception:
                continue

        fallback = build_fallback_findings(
            grade=cnn_result["grade"],
            needs_review=cnn_result["needs_review"],
            review_reason=cnn_result["review_reason"],
        )
        return fallback, {"raw_text": json.dumps(attempts), "fallback_used": True}


def build_report_payload(
    *,
    case_id: str,
    cnn_result: Dict,
    findings: Dict[str, str],
    input_paths: Dict[str, str],
    explainer_debug: Optional[Dict[str, str]] = None,
) -> Dict:
    payload = {
        "case_id": case_id,
        "models": {"cnn": V7_MODEL_NAME, "explainer": MEDGEMMA_MODEL_NAME},
        "grading": {
            "grade": cnn_result["grade"],
            "severity_label": severity_label_for_grade(cnn_result["grade"]),
            "confidence": cnn_result["confidence"],
            "needs_review": cnn_result["needs_review"],
            "review_reason": cnn_result["review_reason"],
        },
        "probabilities": cnn_result["probabilities"],
        "attention": cnn_result["attention"],
        "findings": {key: findings[key] for key in REQUIRED_FINDING_KEYS},
        "input_paths": input_paths,
    }
    if explainer_debug:
        payload["explainer_debug"] = explainer_debug
    return payload


def validate_report_payload(payload: Dict) -> None:
    missing_top = [key for key in ("case_id", "models", "grading", "probabilities", "attention", "findings", "input_paths") if key not in payload]
    if missing_top:
        raise ValueError(f"Missing required payload keys: {missing_top}")

    findings = payload["findings"]
    missing_findings = [key for key in REQUIRED_FINDING_KEYS if key not in findings or not findings[key]]
    if missing_findings:
        raise ValueError(f"Missing required findings: {missing_findings}")

    if payload["grading"]["grade"] not in GRADE_SEVERITY_LABELS:
        raise ValueError(f"Unexpected grade: {payload['grading']['grade']}")


def compute_report_paths(output_root: str | Path, case_id: str, report_date: Optional[date] = None) -> ReportPaths:
    report_date = report_date or date.today()
    output_dir = Path(output_root) / report_date.isoformat() / case_id
    return ReportPaths(
        output_dir=output_dir,
        pdf_path=output_dir / "report.pdf",
        payload_path=output_dir / "report_payload.json",
    )


def _scaled_reportlab_image(image_path: str, max_width: float, max_height: float):
    from reportlab.platypus import Image as RLImage

    img = RLImage(image_path)
    scale = min(max_width / img.drawWidth, max_height / img.drawHeight)
    scale = min(scale, 1.0)
    img.drawWidth *= scale
    img.drawHeight *= scale
    return img


def render_pdf_report(payload: Dict, pdf_path: str | Path) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "reportlab is required for PDF generation. Install it with `pip install reportlab`."
        ) from exc

    validate_report_payload(payload)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, title=f"Cataract Report - {payload['case_id']}")
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "Header",
        parent=styles["Heading1"],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#12263A"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#1F3C5B"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#455A64"),
    )

    grading = payload["grading"]
    header_prefix = "PREDICTED GRADING" if grading["needs_review"] else "GRADING"
    header_line = (
        f"{header_prefix}: {grading['grade']} - {grading['severity_label']} "
        f"(Confidence: {grading['confidence']:.0%})"
    )

    story = [
        Paragraph(header_line, header_style),
        Paragraph(f"Case ID: {payload['case_id']}", meta_style),
        Paragraph(
            f"CNN model: {payload['models']['cnn']} | Explainer model: {payload['models']['explainer']}",
            meta_style,
        ),
        Spacer(1, 0.18 * inch),
    ]

    modality_specs = [
        ("ANTERIOR SEGMENT", "anterior_path", "anterior_segment"),
        ("RED GLOW", "red_glow_path", "red_glow"),
        ("SLIT LAMP", "slit_lamp_path", "slit_lamp"),
    ]
    for title, path_key, finding_key in modality_specs:
        image_block = _scaled_reportlab_image(
            payload["input_paths"][path_key],
            max_width=6.2 * inch,
            max_height=2.6 * inch,
        )
        block = [
            Paragraph(title, section_style),
            image_block,
            Spacer(1, 0.08 * inch),
            Paragraph(payload["findings"][finding_key], body_style),
        ]
        story.append(KeepTogether(block))

    story.extend(
        [
            Paragraph("CONCLUSION", section_style),
            Paragraph(payload["findings"]["conclusion"], body_style),
        ]
    )
    if grading["needs_review"]:
        review_text = "Clinician review is required before relying on this predicted grading."
        if grading["review_reason"]:
            review_text += f" Review reason: {grading['review_reason']}."
        story.append(Paragraph(review_text, body_style))

    doc.build(story)
    return pdf_path


def save_report_payload(payload: Dict, payload_path: str | Path) -> Path:
    validate_report_payload(payload)
    payload_path = Path(payload_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload_path


def load_case_from_metadata(
    metadata_csv: str | Path,
    case_id: str,
    project_root: Optional[str | Path] = None,
) -> Dict[str, str]:
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    row = df[df["group_id"] == case_id]
    if row.empty:
        raise KeyError(f"Could not find case_id '{case_id}' in {metadata_csv}")
    record = row.iloc[0]
    return {
        "case_id": record["group_id"],
        "anterior_path": rebase_dataset_path(record["anterior_path"], project_root=project_root),
        "red_glow_path": rebase_dataset_path(record["red_glow_path"], project_root=project_root),
        "slit_lamp_path": rebase_dataset_path(record["slit_lamp_path"], project_root=project_root),
        "label": record.get("label"),
    }


def select_smoke_cases(
    metadata_csv: str | Path,
    labels: Iterable[str] = ("NS1", "NS2", "NS3", "NS4"),
    project_root: Optional[str | Path] = None,
) -> List[Dict[str, str]]:
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    cases = []
    for label in labels:
        subset = df[df["label"] == label]
        if subset.empty:
            continue
        row = subset.iloc[0]
        cases.append(
            {
                "case_id": row["group_id"],
                "anterior_path": rebase_dataset_path(row["anterior_path"], project_root=project_root),
                "red_glow_path": rebase_dataset_path(row["red_glow_path"], project_root=project_root),
                "slit_lamp_path": rebase_dataset_path(row["slit_lamp_path"], project_root=project_root),
                "label": label,
            }
        )
    return cases


def run_case_report(
    *,
    anterior_path: str,
    red_glow_path: str,
    slit_lamp_path: str,
    case_id: Optional[str],
    output_root: str | Path,
    inference_engine: CataractInferenceEngineV7,
    explainer: Optional[MedGemmaExplainer] = None,
) -> Tuple[Dict, ReportPaths]:
    input_paths = {
        "anterior_path": anterior_path,
        "red_glow_path": red_glow_path,
        "slit_lamp_path": slit_lamp_path,
    }
    ensure_image_paths_exist(input_paths)
    resolved_case_id = infer_case_id(case_id, input_paths)

    cnn_result = inference_engine.predict_case(
        ant_path=anterior_path,
        rg_path=red_glow_path,
        sl_path=slit_lamp_path,
        case_id=resolved_case_id,
    )

    if explainer is None:
        findings = build_fallback_findings(
            grade=cnn_result["grade"],
            needs_review=cnn_result["needs_review"],
            review_reason=cnn_result["review_reason"],
        )
        explainer_debug = {"raw_text": "", "fallback_used": True}
    else:
        findings, explainer_debug = explainer.generate_findings(
            cnn_result=cnn_result,
            anterior_path=anterior_path,
            red_glow_path=red_glow_path,
            slit_lamp_path=slit_lamp_path,
        )

    if cnn_result["needs_review"]:
        findings = dict(findings)
        findings["conclusion"] = (
            findings["conclusion"].rstrip(".")
            + ". Clinician review is required before relying on this predicted grading."
        )
        if cnn_result["review_reason"]:
            findings["conclusion"] += f" Review reason: {cnn_result['review_reason']}."

    payload = build_report_payload(
        case_id=resolved_case_id,
        cnn_result=cnn_result,
        findings=findings,
        input_paths=input_paths,
        explainer_debug=explainer_debug,
    )
    report_paths = compute_report_paths(output_root=output_root, case_id=resolved_case_id)
    save_report_payload(payload, report_paths.payload_path)
    render_pdf_report(payload, report_paths.pdf_path)
    return payload, report_paths
