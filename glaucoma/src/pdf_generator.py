"""
Cataract MedGemma report pipeline map traced before this Glaucoma extension:
- Classification starts in `ophthalmic-web/backend/classifier.py` via `ClassifierService.classify_exam()`, which builds the Cataract result payload and stores `report_texts` as empty because report text is generated later.
- The backend report trigger lives in `ophthalmic-web/backend/main.py` at `GET /report` and `POST /report-batch`; both call `_generate_medgemma_texts_for_record()` before PDF generation.
- `_generate_medgemma_texts_for_record()` in `ophthalmic-web/backend/main.py` builds a `report_text_engine.V7Result` object and calls `generate_report_texts(...)`.
- The MedGemma client/service lives in `ophthalmic-web/backend/medgemma_engine.py`; it lazily loads `google/medgemma-4b-it`, accepts a single prompt string via `generate(prompt, max_new_tokens=...)`, and returns generated text.
- The Cataract prompt templates live in `ophthalmic-web/backend/report_text_engine.py` inside `_build_verbose_modality_prompt()` and `_build_verbose_conclusion_prompt()`, where clinical fields are interpolated into text prompts before calling MedGemma.
- Cataract PDF generation lives in `ophthalmic-web/backend/pdf_report.py` and uses ReportLab (`BaseDocTemplate`, `Table`, `Paragraph`, custom `Flowable`s) to render the final downloadable PDF.
- The frontend trigger for Cataract PDF download is `downloadBatchReport()` in `ophthalmic-web/frontend/src/api.js`, called from the button in `ophthalmic-web/frontend/src/components/UploadPanel.jsx`.
- Cataract report text is not rendered inline in the current frontend; the user flow is classification first, then PDF download.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
MARGIN_Y = 16 * mm
NAVY = colors.HexColor("#0A2342")
TEXT = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5A6478")
BORDER = colors.HexColor("#D0D7E2")
REVIEW_BG = colors.HexColor("#FFF8E1")
REVIEW_BORDER = colors.HexColor("#F9A825")
REVIEW_TEXT = colors.HexColor("#7A5000")
RESULT_RED = colors.HexColor("#B91C1C")
RESULT_GREEN = colors.HexColor("#15803D")
REPORTS_DIR = Path(__file__).resolve().parents[1] / "temp" / "reports"
LOGO_PATH = Path(__file__).resolve().parents[2] / "image.png"


def _draw_logo(canvas: Canvas, x: float, y: float, width: float, height: float) -> None:
    if LOGO_PATH.exists():
        try:
            image = ImageReader(str(LOGO_PATH))
            img_w, img_h = image.getSize()
            scale = min(width / img_w, height / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale
            canvas.drawImage(
                image,
                x,
                y + (height - draw_h) / 2,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                mask="auto",
            )
            return
        except Exception:
            pass

    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(2)
    canvas.arc(x + 4, y + 8, x + width - 4, y + height - 8, startAng=0, extent=180)
    canvas.arc(x + 10, y + 3, x + width - 10, y + height - 20, startAng=200, extent=140)
    canvas.circle(x + width / 2, y + height / 2, 10, stroke=1, fill=0)
    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=TEXT,
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=NAVY,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
    }


def _draw_wrapped_paragraph(canvas: Canvas, text: str, style: ParagraphStyle, x: float, y: float, width: float) -> float:
    para = Paragraph(text.replace("\n", "<br/>"), style)
    w, h = para.wrap(width, PAGE_H)
    para.drawOn(canvas, x, y - h)
    return y - h


def _draw_probability_row(
    canvas: Canvas,
    label: str,
    value: float,
    y: float,
    color: colors.Color,
) -> None:
    canvas.setFillColor(TEXT)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(MARGIN_X + 8, y, label)

    bar_x = MARGIN_X + 95
    bar_y = y - 3
    bar_w = 170
    bar_h = 9
    pct = max(0.0, min(100.0, float(value or 0.0)))

    canvas.setFillColor(colors.HexColor("#EEEEEE"))
    canvas.roundRect(bar_x, bar_y, bar_w, bar_h, 4, fill=1, stroke=0)
    canvas.setFillColor(color)
    canvas.roundRect(bar_x, bar_y, bar_w * (pct / 100.0), bar_h, 4, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(MARGIN_X + 300, y, f"{pct:.2f}%")


def _normalize_probability_keys(probabilities: dict | None) -> dict[str, float]:
    normalized: dict[str, float] = {}

    for raw_key, raw_value in (probabilities or {}).items():
        key = str(raw_key or "").strip().lower().replace("-", "_").replace(" ", "_")
        if key == "normal":
            key = "non_glaucoma"
        elif key == "nonglaucoma":
            key = "non_glaucoma"

        try:
            normalized[key] = float(raw_value or 0.0)
        except (TypeError, ValueError):
            normalized[key] = 0.0

    return normalized


def generate_glaucoma_pdf(
    report_text: str,
    patient_name: str,
    mri_number: str,
    uid: str,
    eye_side: str,
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    needs_review: bool,
    output_path: str,
) -> str:
    """
    Generates a PDF report and saves it to output_path.
    Returns the absolute path of the saved PDF.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_output = Path(output_path)
    if not resolved_output.is_absolute():
        resolved_output = (Path.cwd() / resolved_output).resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    normalized_probabilities = _normalize_probability_keys(probabilities)
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=A4)
    today = datetime.now().strftime("%d-%b-%Y")
    content_w = PAGE_W - (MARGIN_X * 2)

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(1)

    header_top = PAGE_H - MARGIN_Y
    _draw_logo(canvas, MARGIN_X, header_top - 34, 40, 28)
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(MARGIN_X + 48, header_top - 10, "Ophthalmic Imaging - Clinical AI Workbench")
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(MARGIN_X + 48, header_top - 27, "Glaucoma Classification Report")
    canvas.line(MARGIN_X, header_top - 40, PAGE_W - MARGIN_X, header_top - 40)

    info_top = header_top - 52
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(TEXT)
    canvas.drawString(MARGIN_X, info_top, f"Patient Name : {patient_name or '-'}")
    canvas.drawString(PAGE_W / 2, info_top, f"Eye Side : {eye_side or 'UNKNOWN'}")
    canvas.drawString(MARGIN_X, info_top - 14, f"MRI Number : {mri_number or '-'}")
    canvas.drawString(PAGE_W / 2, info_top - 14, f"Date : {today}")
    canvas.drawString(MARGIN_X, info_top - 28, f"UID : {uid or '-'}")
    canvas.drawString(PAGE_W / 2, info_top - 28, "Module : Glaucoma")
    canvas.line(MARGIN_X, info_top - 38, PAGE_W - MARGIN_X, info_top - 38)

    section_top = info_top - 56
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(NAVY)
    canvas.drawString(MARGIN_X, section_top, "RESULT")

    table_top = section_top - 12
    canvas.setStrokeColor(BORDER)
    canvas.rect(MARGIN_X, table_top - 34, content_w, 34, stroke=1, fill=0)
    canvas.line(MARGIN_X + content_w / 2, table_top - 34, MARGIN_X + content_w / 2, table_top)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_X + 10, table_top - 12, "Finding")
    canvas.drawString(MARGIN_X + content_w / 2 + 10, table_top - 12, "Confidence")
    canvas.setFont("Helvetica-Bold", 12)
    canvas.setFillColor(RESULT_RED if predicted_class.lower() == "glaucoma" else RESULT_GREEN)
    canvas.drawString(MARGIN_X + 10, table_top - 28, predicted_class or "Unknown")
    canvas.drawString(MARGIN_X + content_w / 2 + 10, table_top - 28, f"{float(confidence or 0.0):.2f}%")

    breakdown_top = table_top - 54
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(TEXT)
    canvas.drawString(MARGIN_X, breakdown_top, "Confidence Breakdown:")
    _draw_probability_row(
        canvas,
        "Glaucoma",
        normalized_probabilities.get("glaucoma", 0.0),
        breakdown_top - 18,
        RESULT_RED,
    )
    _draw_probability_row(
        canvas,
        "Non-Glaucoma",
        normalized_probabilities.get("non_glaucoma", 0.0),
        breakdown_top - 36,
        RESULT_GREEN,
    )

    current_y = breakdown_top - 54
    if needs_review:
        canvas.setFillColor(REVIEW_BG)
        canvas.roundRect(MARGIN_X, current_y - 20, content_w, 20, 4, fill=1, stroke=0)
        canvas.setFillColor(REVIEW_TEXT)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(MARGIN_X + 8, current_y - 13, "[REVIEW RECOMMENDED] Confidence threshold triggered.")
        current_y -= 30

    canvas.setStrokeColor(BORDER)
    canvas.line(MARGIN_X, current_y, PAGE_W - MARGIN_X, current_y)
    current_y -= 18
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(MARGIN_X, current_y, "CLINICAL REPORT (MedGemma generated)")
    current_y -= 8

    report_box_top = current_y
    report_box_height = 250
    canvas.setStrokeColor(BORDER)
    canvas.roundRect(MARGIN_X, report_box_top - report_box_height, content_w, report_box_height, 6, fill=0, stroke=1)
    text_y = report_box_top - 14
    text_x = MARGIN_X + 10
    text_w = content_w - 20

    for block in [segment.strip() for segment in report_text.split("\n\n") if segment.strip()]:
        text_y = _draw_wrapped_paragraph(canvas, block, styles["body"], text_x, text_y, text_w) - 8
        if text_y < report_box_top - report_box_height + 16:
            break

    footer_y = MARGIN_Y + 26
    canvas.line(MARGIN_X, footer_y + 10, PAGE_W - MARGIN_X, footer_y + 10)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN_X, footer_y, "Generated by Ophthalmic Imaging Clinical AI")
    canvas.drawString(MARGIN_X, footer_y - 10, "This report is AI-assisted and not a substitute for clinical judgment.")

    canvas.save()
    resolved_output.write_bytes(buffer.getvalue())
    return str(resolved_output)
