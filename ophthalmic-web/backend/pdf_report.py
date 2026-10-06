"""
pdf_report.py - Ophthalmic Imaging Medical Diagnostic Report Generator

Produces hospital-grade A4 PDF reports for cataract grading results from the
V7 Multimodal CNN (EfficientNet-B2 + ConvNeXt-Tiny).

Two public functions (signatures unchanged):
    generate_pdf_report(username, session_record) -> bytes
    generate_batch_pdf_report(username, results, generated_at) -> bytes
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Import dynamic text engine (same backend/ directory)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PDF_LOGO_PATH = PROJECT_ROOT / "image.png"

from report_text_engine import V7Result, generate_report_texts  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4  # 595.27, 841.89
MARGIN_LEFT = 18 * mm
MARGIN_RIGHT = 18 * mm
MARGIN_TOP = 15 * mm
MARGIN_BOTTOM = 15 * mm

NAVY = colors.HexColor("#0A2342")
LIGHT_BORDER = colors.HexColor("#B0BEC5")
TEXT_DARK = colors.HexColor("#1A1A1A")
TEXT_MUTED = colors.HexColor("#455A64")
ROW_ALT = colors.HexColor("#F4F6F9")
FOOTER_BG = colors.HexColor("#F4F6F9")
FOOTER_TEXT = colors.HexColor("#78909C")
CONCLUSION_BG = colors.HexColor("#EBF5FB")
CONCLUSION_BORDER = colors.HexColor("#1A5276")
WARNING_BG = colors.HexColor("#FFF8E1")
WARNING_BORDER = colors.HexColor("#F9A825")
WARNING_TEXT = colors.HexColor("#7A5000")
BRAND_BLUE = colors.HexColor("#2563EB")

GRADE_COLOURS = {
    "NS1": colors.HexColor("#1B7A3E"),
    "NS2": colors.HexColor("#7A6000"),
    "NS3": colors.HexColor("#B84B00"),
    "NS4": colors.HexColor("#8B0000"),
    "Unknown": colors.HexColor("#37474F"),
}

SEVERITY_LABELS = {"NS1": "Minimal", "NS2": "Mild", "NS3": "Advanced", "NS4": "Severe"}

_UNKNOWN_TEXTS = {
    "anterior":               "Grade unavailable. Please review the anterior segment image clinically.",
    "red_glow":               "Grade unavailable. Please review the red glow image clinically.",
    "slit_lamp":              "Grade unavailable. Please review the slit lamp image clinically.",
    "conclusion":             "Grade unavailable. Please review clinically.",
    "attention_narrative":    "",
    "probability_commentary": "",
}

CONTENT_WIDTH = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT


# ---------------------------------------------------------------------------
# Logo helper
# ---------------------------------------------------------------------------
def _draw_pdf_logo(canvas: Canvas, x: float, y: float, width: float, height: float):
    """Draw a vector logo in the PDF header."""
    canvas.saveState()
    canvas.setStrokeColor(BRAND_BLUE)
    canvas.setFillColor(BRAND_BLUE)
    canvas.setLineCap(1)
    canvas.setLineJoin(1)

    cx = x + width / 2
    cy = y + height / 2 - 4

    # Top rainbow arcs
    canvas.setLineWidth(2.8)
    top_y = y + height - 6
    for inset in (3, 8, 13, 18, 23):
        canvas.arc(x + inset, cy - 3, x + width - inset, top_y, startAng=0, extent=180)

    # Lower eye sweep
    canvas.setLineWidth(3.0)
    canvas.arc(x + 12, y + 6, x + width - 12, y + height * 0.60, startAng=198, extent=144)

    # Center emblem: blue disc with white droplet inside, plus blue ring
    outer_r = min(width, height) * 0.19
    disc_r = outer_r * 0.74
    canvas.circle(cx, cy, outer_r, stroke=1, fill=0)
    canvas.circle(cx, cy, disc_r, stroke=0, fill=1)

    white_drop = canvas.beginPath()
    white_drop.moveTo(cx - disc_r * 0.05, cy + disc_r * 0.95)
    white_drop.curveTo(cx + disc_r * 0.22, cy + disc_r * 0.25, cx + disc_r * 0.28, cy - disc_r * 0.10, cx + disc_r * 0.03, cy - disc_r * 0.82)
    white_drop.curveTo(cx - disc_r * 0.23, cy - disc_r * 0.58, cx - disc_r * 0.18, cy - disc_r * 0.18, cx - disc_r * 0.04, cy + disc_r * 0.10)
    white_drop.curveTo(cx + disc_r * 0.02, cy + disc_r * 0.30, cx + disc_r * 0.00, cy + disc_r * 0.60, cx - disc_r * 0.05, cy + disc_r * 0.95)
    canvas.setFillColor(colors.white)
    canvas.drawPath(white_drop, stroke=0, fill=1)

    # Small blue base drop to mimic the source mark
    canvas.setFillColor(BRAND_BLUE)
    inner_drop = canvas.beginPath()
    inner_drop.moveTo(cx - disc_r * 0.12, cy - disc_r * 0.50)
    inner_drop.curveTo(cx + disc_r * 0.04, cy - disc_r * 0.34, cx + disc_r * 0.06, cy - disc_r * 0.14, cx - disc_r * 0.03, cy + disc_r * 0.04)
    inner_drop.curveTo(cx - disc_r * 0.18, cy - disc_r * 0.06, cx - disc_r * 0.22, cy - disc_r * 0.30, cx - disc_r * 0.12, cy - disc_r * 0.50)
    canvas.drawPath(inner_drop, stroke=0, fill=1)
    canvas.restoreState()


def _draw_pdf_logo_image(canvas: Canvas, x: float, y: float, width: float, height: float):
    """Draw the configured logo image, falling back to the vector mark if unavailable."""
    if not PDF_LOGO_PATH.exists():
        _draw_pdf_logo(canvas, x, y, width, height)
        return

    try:
        image = ImageReader(str(PDF_LOGO_PATH))
        img_w, img_h = image.getSize()
        scale = min(width / img_w, height / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        draw_x = x + (width - draw_w) / 2
        draw_y = y + (height - draw_h) / 2
        canvas.drawImage(image, draw_x, draw_y, width=draw_w, height=draw_h, mask="auto", preserveAspectRatio=True)
    except Exception:
        _draw_pdf_logo(canvas, x, y, width, height)


# ---------------------------------------------------------------------------
# Report text helper (KEEP VERBATIM)
# ---------------------------------------------------------------------------
def _get_report_texts(result: dict) -> dict:
    """
    Build all dynamic report text blocks from the V7 inference result dict.

    The result dict is expected to have:
        grade            : str   â€” "NS1" | "NS2" | "NS3" | "NS4" | "Unknown"
        confidence       : float â€” 0.0 to 1.0
        probabilities    : dict  â€” {"NS1": float, "NS2": float, ...}
        attention        : dict  â€” {"anterior": float, "red_glow": float, "slit_lamp": float}

    Returns a dict with keys:
        anterior, red_glow, slit_lamp, conclusion,
        attention_narrative, probability_commentary
    """
    try:
        cached_texts = result.get("report_texts")
        expected_keys = {
            "anterior",
            "red_glow",
            "slit_lamp",
            "conclusion",
            "attention_narrative",
            "probability_commentary",
        }
        if isinstance(cached_texts, dict) and expected_keys.issubset(cached_texts.keys()):
            return {key: cached_texts[key] for key in expected_keys}

        grade = result.get("grade") or "Unknown"
        if grade not in ("NS1", "NS2", "NS3", "NS4"):
            return _UNKNOWN_TEXTS.copy()

        confidence    = result.get("confidence") or 0.0
        probabilities = result.get("probabilities") or {}
        attention     = result.get("attention") or {}

        # Ensure all three attention keys are present â€” avoid max() on empty dict
        attention_full = {
            "anterior":  float(attention.get("anterior",  0.0)),
            "red_glow":  float(attention.get("red_glow",  0.0)),
            "slit_lamp": float(attention.get("slit_lamp", 0.0)),
        }

        v7 = V7Result(
            grade=grade,
            confidence=float(confidence),
            probabilities={k: float(v) for k, v in probabilities.items()},
            attention=attention_full,
            case_id=result.get("case_id"),
            clinician=result.get("clinician") or result.get("graded_by"),
            gradcam_region_info=result.get("gradcam_region_info") or {},
            raw_result=result,
        )
        return generate_report_texts(v7)

    except Exception as exc:
        print(f"[pdf_report] _get_report_texts failed: {exc}")
        return _UNKNOWN_TEXTS.copy()


# ---------------------------------------------------------------------------
# NumberedCanvas â€” for Page X of Y
# ---------------------------------------------------------------------------
class NumberedCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for idx, state in enumerate(self._saved_page_states, 1):
            self.__dict__.update(state)
            self._draw_page_number(idx, num_pages)
            super().showPage()
        super().save()

    def _draw_page_number(self, page_num: int, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 6.5)
        self.setFillColor(FOOTER_TEXT)
        self.drawRightString(
            PAGE_W - MARGIN_RIGHT,
            MARGIN_BOTTOM - 2 * mm,
            f"Page {page_num} of {page_count}",
        )
        self.restoreState()


# ---------------------------------------------------------------------------
# Custom Flowable: Probability Bars
# ---------------------------------------------------------------------------
class ProbabilityBarsFlowable(Flowable):
    """Draws NS1â€“NS4 probability bars using canvas rect() calls."""

    BAR_MAX_W = 220
    ROW_H = 10
    ROW_GAP = 4
    LABEL_W = 25
    PCT_W = 30
    PADDING_TOP = 4
    PADDING_BOTTOM = 4

    def __init__(self, probabilities: dict[str, float], predicted_grade: str):
        super().__init__()
        self.probabilities = probabilities
        self.predicted_grade = predicted_grade
        self.width = CONTENT_WIDTH
        self.height = self.PADDING_TOP + 4 * (self.ROW_H + self.ROW_GAP) - self.ROW_GAP + self.PADDING_BOTTOM

    def draw(self):
        c = self.canv
        y = self.height - self.PADDING_TOP - self.ROW_H
        for grade_label in ["NS1", "NS2", "NS3", "NS4"]:
            prob = self.probabilities.get(grade_label, 0.0)
            # Grade label
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(NAVY)
            c.drawString(0, y + 1, grade_label)

            # Background bar
            bar_x = self.LABEL_W + 4
            c.setFillColor(colors.HexColor("#EEEEEE"))
            c.rect(bar_x, y, self.BAR_MAX_W, self.ROW_H, fill=1, stroke=0)

            # Filled bar
            fill_w = max(prob * self.BAR_MAX_W, 0)
            if grade_label == self.predicted_grade:
                bar_colour = GRADE_COLOURS.get(grade_label, LIGHT_BORDER)
            else:
                bar_colour = LIGHT_BORDER
            c.setFillColor(bar_colour)
            c.rect(bar_x, y, fill_w, self.ROW_H, fill=1, stroke=0)

            # Percentage label
            c.setFont("Helvetica", 8)
            c.setFillColor(TEXT_MUTED)
            c.drawString(bar_x + self.BAR_MAX_W + 6, y + 1, f"{round(prob * 100, 1)}%")

            y -= (self.ROW_H + self.ROW_GAP)


# ---------------------------------------------------------------------------
# Custom Flowable: Diagnosis Banner
# ---------------------------------------------------------------------------
class DiagnosisBannerFlowable(Flowable):
    """Full-width coloured diagnosis banner with grade badge."""

    def __init__(self, grade: str, severity: str, confidence_pct: str,
                 model_info: str, needs_review: bool, review_reason: str | None):
        super().__init__()
        self.grade = grade
        self.severity = severity
        self.confidence_pct = confidence_pct
        self.model_info = model_info
        self.needs_review = needs_review
        self.review_reason = review_reason
        self.width = CONTENT_WIDTH
        self.height = 62
        if needs_review:
            self.height += 22

    def draw(self):
        c = self.canv
        bg = GRADE_COLOURS.get(self.grade, GRADE_COLOURS["Unknown"])
        banner_h = 62
        y_base = self.height - banner_h if self.needs_review else 0

        # Main background
        main_w = self.width * 0.75
        c.setFillColor(bg)
        c.rect(0, y_base, main_w, banner_h, fill=1, stroke=0)

        # Badge area (slightly lighter)
        badge_w = self.width * 0.25
        r_val = bg.red * 0.85
        g_val = bg.green * 0.85
        b_val = bg.blue * 0.85
        c.setFillColor(colors.Color(r_val, g_val, b_val))
        c.rect(main_w, y_base, badge_w, banner_h, fill=1, stroke=0)

        # Text on main area
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(10, y_base + banner_h - 16, "NUCLEAR  SCLEROSIS  CATARACT")

        c.setFont("Helvetica-Bold", 14)
        c.drawString(10, y_base + banner_h - 34,
                     f"Grade: {self.grade} \u2014 {self.severity}")

        c.setFont("Helvetica", 10)
        c.drawString(10, y_base + banner_h - 48,
                     f"Confidence: {self.confidence_pct}")

        c.setFont("Helvetica", 8)
        c.drawString(10, y_base + 6,
                     f"AI Model: {self.model_info}")

        # Badge text
        c.setFont("Helvetica-Bold", 22)
        badge_x = main_w + badge_w / 2
        badge_y = y_base + banner_h / 2 - 6
        c.drawCentredString(badge_x, badge_y, self.grade)

        # Warning row if needed
        if self.needs_review:
            wy = 0
            wh = 20
            c.setFillColor(WARNING_BG)
            c.rect(0, wy, self.width, wh, fill=1, stroke=0)
            # Left accent border
            c.setFillColor(WARNING_BORDER)
            c.rect(0, wy, 3, wh, fill=1, stroke=0)
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(WARNING_TEXT)
            reason_text = self.review_reason or ""
            c.drawString(8, wy + 6,
                         f"\u26A0  Clinical review recommended \u2014 {reason_text}".strip())


# ---------------------------------------------------------------------------
# Page callbacks: letterhead + footer
# ---------------------------------------------------------------------------
def _make_page_callback(report_id: str, report_date: str, report_time: str):
    """Returns a callback that draws the letterhead and footer on every page."""

    def _on_page(canvas: Canvas, doc):
        canvas.saveState()

        # â”€â”€ LETTERHEAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lh_top = PAGE_H - MARGIN_TOP
        logo_x = MARGIN_LEFT
        logo_y = lh_top - 86
        logo_w = 96
        logo_h = 86

        # Logo placeholder â€” dashed box
        _draw_pdf_logo_image(canvas, logo_x, logo_y, logo_w, logo_h)

        # Right-side text
        text_x = logo_x + logo_w + 18
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(text_x, lh_top - 16, "Ophthalmic Imaging")

        canvas.setFillColor(TEXT_MUTED)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(text_x, lh_top - 30, "Ophthalmology AI Diagnostic Centre")

        canvas.setFont("Helvetica", 8)
        canvas.drawString(text_x, lh_top - 42, "Department of Cataract & Refractive Surgery")

        # Thin rule below subtitle lines
        rule_y = lh_top - 48
        canvas.setStrokeColor(LIGHT_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(text_x, rule_y, PAGE_W - MARGIN_RIGHT, rule_y)

        # Report No / Date / Time
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(text_x, rule_y - 12, f"Report No: {report_id}")

        canvas.setFillColor(TEXT_MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(text_x + 180, rule_y - 12,
                          f"Date: {report_date}   Time: {report_time}")

        # Full-width rule below letterhead
        sep_y = logo_y - 6
        canvas.setStrokeColor(NAVY)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN_LEFT, sep_y, PAGE_W - MARGIN_RIGHT, sep_y)

        # â”€â”€ FOOTER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        footer_top = MARGIN_BOTTOM + 8 * mm
        footer_h = 8 * mm + 4

        # Footer background
        canvas.setFillColor(FOOTER_BG)
        canvas.rect(MARGIN_LEFT, MARGIN_BOTTOM - 4 * mm,
                    CONTENT_WIDTH, footer_h + 4, fill=1, stroke=0)

        # Top rule
        canvas.setStrokeColor(LIGHT_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_LEFT, footer_top, PAGE_W - MARGIN_RIGHT, footer_top)

        # Disclaimer
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(FOOTER_TEXT)
        disclaimer = (
            "This report is generated by an AI-assisted system and is intended to support "
            "\u2014 not replace \u2014 clinical judgement. "
            "A qualified ophthalmologist must review all findings before clinical decisions."
        )
        canvas.drawCentredString(PAGE_W / 2, footer_top - 10, disclaimer)

        # Bottom line: org name left, page number right (page # handled by NumberedCanvas)
        canvas.drawString(MARGIN_LEFT, MARGIN_BOTTOM - 2 * mm,
                          "Ophthalmic Imaging | Ophthalmology AI Diagnostic Centre")

        canvas.restoreState()

    return _on_page


# ---------------------------------------------------------------------------
# Shared styles
# ---------------------------------------------------------------------------
def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "section_heading": ParagraphStyle(
            "SectionHeading", parent=base["BodyText"],
            fontName="Helvetica-Bold", fontSize=9,
            textColor=NAVY, spaceAfter=2, spaceBefore=6,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["BodyText"],
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=TEXT_DARK,
        ),
        "body_large": ParagraphStyle(
            "ReportBodyLarge", parent=base["BodyText"],
            fontName="Helvetica", fontSize=9.5, leading=15,
            textColor=TEXT_DARK,
        ),
        "body_italic": ParagraphStyle(
            "ReportBodyItalic", parent=base["BodyText"],
            fontName="Helvetica-Oblique", fontSize=9, leading=14,
            textColor=TEXT_MUTED,
        ),
        "label_bold": ParagraphStyle(
            "LabelBold", parent=base["BodyText"],
            fontName="Helvetica-Bold", fontSize=8,
            textColor=NAVY,
        ),
        "value": ParagraphStyle(
            "Value", parent=base["BodyText"],
            fontName="Helvetica", fontSize=9,
            textColor=TEXT_DARK,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"],
            fontName="Helvetica-Bold", fontSize=8,
            textColor=NAVY, alignment=TA_CENTER,
        ),
        "caption_sm": ParagraphStyle(
            "CaptionSm", parent=base["BodyText"],
            fontName="Helvetica", fontSize=7.5,
            textColor=TEXT_MUTED, alignment=TA_CENTER,
        ),
        "badge": ParagraphStyle(
            "Badge", parent=base["BodyText"],
            fontName="Helvetica-Bold", fontSize=7,
            textColor=NAVY, alignment=TA_CENTER,
        ),
        "missing": ParagraphStyle(
            "Missing", parent=base["BodyText"],
            fontName="Helvetica-Oblique", fontSize=9,
            textColor=colors.HexColor("#9E9E9E"), alignment=TA_CENTER,
        ),
        "warning": ParagraphStyle(
            "Warning", parent=base["BodyText"],
            fontName="Helvetica-Bold", fontSize=9,
            textColor=WARNING_TEXT,
        ),
    }


# ---------------------------------------------------------------------------
# Decode base64 image helper
# ---------------------------------------------------------------------------
def _decode_image_el(image_b64: str, max_w: int = 150, max_h: int = 120) -> Image:
    raw = base64.b64decode(image_b64)
    with PILImage.open(io.BytesIO(raw)) as pil:
        w, h = pil.size
    scale = min(max_w / w, max_h / h, 1.0)
    img = Image(io.BytesIO(raw), width=w * scale, height=h * scale)
    img.hAlign = "CENTER"
    return img


# ---------------------------------------------------------------------------
# Image placeholder (dashed rect)
# ---------------------------------------------------------------------------
class ImagePlaceholderFlowable(Flowable):
    def __init__(self, w: float = 150, h: float = 120):
        super().__init__()
        self.width = w
        self.height = h

    def draw(self):
        c = self.canv
        c.setStrokeColor(LIGHT_BORDER)
        c.setLineWidth(0.8)
        c.setDash(3, 3)
        c.rect(0, 0, self.width, self.height, fill=0, stroke=1)
        c.setDash()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#9E9E9E"))
        c.drawCentredString(self.width / 2, self.height / 2 - 3, "Not provided")


# ---------------------------------------------------------------------------
# Section rule helper
# ---------------------------------------------------------------------------
class SectionRule(Flowable):
    """A thin horizontal rule below a section heading."""
    def __init__(self, colour=NAVY, thickness=0.5, width=CONTENT_WIDTH):
        super().__init__()
        self.colour = colour
        self.thickness = thickness
        self.width = width
        self.height = self.thickness + 2

    def draw(self):
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


# ---------------------------------------------------------------------------
# Conclusion box (rounded rect background)
# ---------------------------------------------------------------------------
class ConclusionBoxFlowable(Flowable):
    """Wraps text in a light blue-grey rounded box."""

    def __init__(self, text: str, styles: dict):
        super().__init__()
        self.text = text
        self.styles = styles
        self.padding = 8
        # Pre-calculate height
        style = ParagraphStyle(
            "ConclusionInner", parent=styles["body_large"],
        )
        p = Paragraph(text, style)
        w_avail = CONTENT_WIDTH - 2 * self.padding
        pw, ph = p.wrap(w_avail, 1000)
        self.width = CONTENT_WIDTH
        self.height = ph + 2 * self.padding
        self._para = p
        self._para_h = ph

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(CONCLUSION_BG)
        c.setStrokeColor(CONCLUSION_BORDER)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, self.width, self.height, 2, fill=1, stroke=1)
        # Text
        self._para.drawOn(c, self.padding, self.height - self.padding - self._para_h)


# ---------------------------------------------------------------------------
# Shared layout helpers for multi-eye reports
# ---------------------------------------------------------------------------
def _build_patient_info_table(
    styles: dict[str, ParagraphStyle],
    patient_name: str,
    mri_number: str,
    username: str,
    model_info: str,
) -> Table:
    info_data = [
        [Paragraph("<b>Patient Name:</b>", styles["label_bold"]),
         Paragraph(patient_name or "\u2014", styles["value"]),
         Paragraph("<b>MRI Number:</b>", styles["label_bold"]),
         Paragraph(mri_number or "\u2014", styles["value"])],
        [Paragraph("<b>Clinician:</b>", styles["label_bold"]),
         Paragraph(username or "\u2014", styles["value"]),
         Paragraph("<b>Model:</b>", styles["label_bold"]),
         Paragraph(model_info or "\u2014", styles["value"])],
    ]

    col_w = CONTENT_WIDTH / 4
    info_table = Table(info_data, colWidths=[col_w] * 4)
    info_style_cmds = [
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.5, LIGHT_BORDER),
    ]
    for i in range(len(info_data)):
        bg = ROW_ALT if i % 2 == 0 else colors.white
        info_style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

    info_table.setStyle(TableStyle(info_style_cmds))
    return info_table


def _render_story_pdf(
    story: list[Any],
    report_id: str,
    report_date_str: str,
    report_time_str: str,
) -> bytes:
    if not story:
        raise ValueError("PDF story is empty Ã¢â‚¬â€ no content was generated")

    buffer = io.BytesIO()

    frame_top = PAGE_H - MARGIN_TOP - 74
    frame_bottom = MARGIN_BOTTOM + 12 * mm
    frame_height = frame_top - frame_bottom

    frame = Frame(
        MARGIN_LEFT, frame_bottom,
        CONTENT_WIDTH, frame_height,
        leftPadding=0, rightPadding=0,
        topPadding=4, bottomPadding=4,
        id="main",
    )

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
    )
    doc.addPageTemplates([
        PageTemplate(
            id="report",
            frames=[frame],
            onPage=_make_page_callback(report_id, report_date_str, report_time_str),
        ),
    ])
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


def _append_eye_report_sections(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    *,
    eye_side: str,
    case_id: str,
    grade: str,
    severity: str,
    confidence_pct: str,
    probabilities: dict[str, float],
    attention: dict[str, float],
    needs_review: bool,
    review_reason: str | None,
    texts: dict[str, str],
    ant_b64: str | None,
    rg_b64: str | None,
    sl_b64: str | None,
    model_info: str,
) -> None:
    eye_label = eye_side or "UNKNOWN"
    eye_meta = Table(
        [[
            Paragraph("<b>Eye Side:</b>", styles["label_bold"]),
            Paragraph(eye_label, styles["value"]),
            Paragraph("<b>Case ID:</b>", styles["label_bold"]),
            Paragraph(case_id or "\u2014", styles["value"]),
        ]],
        colWidths=[CONTENT_WIDTH / 4] * 4,
    )
    eye_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(Paragraph(f"{eye_label} EYE REPORT", styles["section_heading"]))
    story.append(SectionRule(NAVY, 0.8))
    story.append(Spacer(1, 4))
    story.append(eye_meta)
    story.append(Spacer(1, 8))

    banner = DiagnosisBannerFlowable(
        grade=grade,
        severity=severity,
        confidence_pct=confidence_pct,
        model_info=model_info,
        needs_review=needs_review,
        review_reason=review_reason,
    )
    story.append(banner)
    story.append(Spacer(1, 8))

    dominant_key = max(attention, key=attention.get) if attention else None

    image_cells: list[list[Any]] = [[], [], []]
    modalities = [
        ("Anterior Segment", ant_b64, "anterior"),
        ("Red Glow (Retroillumination)", rg_b64, "red_glow"),
        ("Slit Lamp", sl_b64, "slit_lamp"),
    ]

    for label, b64, key in modalities:
        image_cells[0].append(Paragraph(label, styles["caption"]))
        if b64:
            image_cells[1].append(_decode_image_el(b64, max_w=150, max_h=120))
        else:
            image_cells[1].append(ImagePlaceholderFlowable(150, 120))

        attn_pct = round(attention.get(key, 0) * 100, 1)
        attn_text = f"Attention: {attn_pct}%"
        if key == dominant_key:
            attn_text += "<br/><b>\u2605 Primary Driver</b>"
        image_cells[2].append(Paragraph(attn_text, styles["caption_sm"]))

    img_col_w = CONTENT_WIDTH / 3
    img_table = Table(image_cells, colWidths=[img_col_w] * 3)
    img_table.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 1, LIGHT_BORDER),
        ("LINEBEFORE", (1, 0), (1, -1), 0.5, LIGHT_BORDER),
        ("LINEBEFORE", (2, 0), (2, -1), 0.5, LIGHT_BORDER),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("IMAGING FINDINGS", styles["section_heading"]))
    story.append(SectionRule(NAVY, 0.5))
    story.append(Spacer(1, 4))

    findings_data = []
    for mod_label, key in [
        ("Anterior Segment", "anterior"),
        ("Red Glow", "red_glow"),
        ("Slit Lamp", "slit_lamp"),
    ]:
        label_str = mod_label
        if key == dominant_key:
            label_str += " \u2605"
        findings_data.append([
            Paragraph(f"<b>{label_str}</b>", styles["label_bold"]),
            Paragraph(texts[key], styles["body"]),
        ])

    findings_table = Table(findings_data, colWidths=[90, CONTENT_WIDTH - 90])
    findings_style_cmds = [
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]
    for i in range(len(findings_data)):
        bg = ROW_ALT if i % 2 == 0 else colors.white
        findings_style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        if i < len(findings_data) - 1:
            findings_style_cmds.append(
                ("LINEBELOW", (0, i), (-1, i), 0.3, colors.HexColor("#D0D7E2"))
            )
    findings_table.setStyle(TableStyle(findings_style_cmds))
    story.append(findings_table)
    story.append(Spacer(1, 10))

    if probabilities:
        story.append(Paragraph("CLASSIFICATION PROBABILITIES", styles["section_heading"]))
        story.append(SectionRule(NAVY, 0.5))
        story.append(Spacer(1, 4))
        story.append(ProbabilityBarsFlowable(probabilities, grade))
        story.append(Spacer(1, 10))

    story.append(Paragraph("CONCLUSION &amp; RECOMMENDATION", styles["section_heading"]))
    story.append(Spacer(1, 4))
    story.append(ConclusionBoxFlowable(texts["conclusion"], styles))
    story.append(Spacer(1, 10))

    if texts["attention_narrative"] or texts["probability_commentary"]:
        story.append(Paragraph("MODEL ANALYSIS", styles["section_heading"]))
        story.append(SectionRule(NAVY, 0.5))
        story.append(Spacer(1, 4))
        if texts["attention_narrative"]:
            story.append(Paragraph(texts["attention_narrative"], styles["body"]))
            story.append(Spacer(1, 6))
        if texts["probability_commentary"]:
            story.append(Paragraph(texts["probability_commentary"], styles["body_italic"]))


def _build_combined_batch_report(
    username: str,
    valid_results: list[dict[str, Any]],
    generated_at: datetime,
) -> bytes:
    styles = _build_styles()
    first_result = valid_results[0]
    patient_name = next((r.get("patient_name") for r in valid_results if r.get("patient_name")), "")
    mri_number = next((r.get("mri_number") for r in valid_results if r.get("mri_number")), "")
    shared_model_info = next(
        (r.get("model_info") for r in valid_results if r.get("model_info")),
        "V7 Multimodal CNN | EfficientNet-B2 + ConvNeXt-Tiny",
    )
    base_report_id = (
        first_result.get("case_id")
        or first_result.get("filename")
        or mri_number
        or "UNKNOWN"
    )
    report_id = f"{base_report_id}-BOTH-EYES"

    story: list[Any] = [
        _build_patient_info_table(
            styles=styles,
            patient_name=patient_name,
            mri_number=mri_number,
            username=username,
            model_info=shared_model_info,
        ),
        Spacer(1, 10),
    ]

    for idx, result in enumerate(valid_results):
        if idx > 0:
            story.append(PageBreak())

        grade = result.get("grade") or "Unknown"
        severity = SEVERITY_LABELS.get(grade, "Unknown")
        confidence = result.get("confidence")
        confidence_pct = f"{round(confidence * 100)}%" if confidence is not None else "N/A"
        _append_eye_report_sections(
            story=story,
            styles=styles,
            eye_side=result.get("eye_side") or f"Eye {idx + 1}",
            case_id=result.get("case_id") or result.get("filename") or "UNKNOWN",
            grade=grade,
            severity=severity,
            confidence_pct=confidence_pct,
            probabilities=result.get("probabilities") or {},
            attention=result.get("attention") or {},
            needs_review=bool(result.get("needs_review", False)),
            review_reason=result.get("review_reason"),
            texts=_get_report_texts(result),
            ant_b64=result.get("anterior_segment_base64"),
            rg_b64=result.get("red_glow_base64"),
            sl_b64=result.get("slit_lamp_base64"),
            model_info=result.get("model_info", shared_model_info),
        )

    pdf_bytes = _render_story_pdf(
        story=story,
        report_id=report_id,
        report_date_str=generated_at.strftime("%d-%b-%Y"),
        report_time_str=generated_at.strftime("%H:%M"),
    )
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Combined PDF generation produced invalid output")
    return pdf_bytes


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------
def _build_report(
    report_id: str,
    report_date_str: str,
    report_time_str: str,
    patient_name: str,
    mri_number: str,
    username: str,
    model_info: str,
    grade: str,
    severity: str,
    confidence_pct: str,
    probabilities: dict[str, float],
    attention: dict[str, float],
    needs_review: bool,
    review_reason: str | None,
    texts: dict[str, str],
    ant_b64: str | None,
    rg_b64: str | None,
    sl_b64: str | None,
) -> bytes:
    """Shared builder for both single and batch reports."""

    buffer = io.BytesIO()
    styles = _build_styles()

    # Frame sits below letterhead (top ~80pt reserved) and above footer (~30pt)
    frame_top = PAGE_H - MARGIN_TOP - 74  # below letterhead
    frame_bottom = MARGIN_BOTTOM + 12 * mm   # above footer
    frame_height = frame_top - frame_bottom

    frame = Frame(
        MARGIN_LEFT, frame_bottom,
        CONTENT_WIDTH, frame_height,
        leftPadding=0, rightPadding=0,
        topPadding=4, bottomPadding=4,
        id="main",
    )

    page_cb = _make_page_callback(report_id, report_date_str, report_time_str)

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
    )
    doc.addPageTemplates([
        PageTemplate(id="report", frames=[frame],
                     onPage=page_cb),
    ])

    story: list[Any] = []

    # â”€â”€ SECTION 2: Patient & Encounter Information â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pn = patient_name if patient_name else "\u2014"
    mn = mri_number if mri_number else "\u2014"

    info_data = [
        [Paragraph("<b>Patient Name:</b>", styles["label_bold"]),
         Paragraph(pn, styles["value"]),
         Paragraph("<b>MRI Number:</b>", styles["label_bold"]),
         Paragraph(mn, styles["value"])],
        [Paragraph("<b>Clinician:</b>", styles["label_bold"]),
         Paragraph(username, styles["value"]),
         Paragraph("<b>Model:</b>", styles["label_bold"]),
         Paragraph(model_info, styles["value"])],
    ]

    col_w = CONTENT_WIDTH / 4
    info_table = Table(info_data, colWidths=[col_w] * 4)
    info_style_cmds = [
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.5, LIGHT_BORDER),
    ]
    # Alternating row backgrounds
    for i in range(len(info_data)):
        bg = ROW_ALT if i % 2 == 0 else colors.white
        info_style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

    info_table.setStyle(TableStyle(info_style_cmds))
    story.append(info_table)
    story.append(Spacer(1, 8))

    # â”€â”€ SECTION 3: Diagnosis Banner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    banner = DiagnosisBannerFlowable(
        grade=grade, severity=severity, confidence_pct=confidence_pct,
        model_info=model_info, needs_review=needs_review,
        review_reason=review_reason,
    )
    story.append(banner)
    story.append(Spacer(1, 8))

    # â”€â”€ SECTION 4: Clinical Images â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dominant_key = max(attention, key=attention.get) if attention else None

    image_cells: list[list[Any]] = [[], [], []]  # label row, image row, attn row
    modalities = [
        ("Anterior Segment", ant_b64, "anterior"),
        ("Red Glow (Retroillumination)", rg_b64, "red_glow"),
        ("Slit Lamp", sl_b64, "slit_lamp"),
    ]

    for label, b64, key in modalities:
        # Label
        image_cells[0].append(Paragraph(label, styles["caption"]))
        # Image
        if b64:
            image_cells[1].append(_decode_image_el(b64, max_w=150, max_h=120))
        else:
            image_cells[1].append(ImagePlaceholderFlowable(150, 120))
        # Attention + badge
        attn_pct = round(attention.get(key, 0) * 100, 1)
        attn_text = f"Attention: {attn_pct}%"
        if key == dominant_key:
            attn_text += "<br/><b>\u2605 Primary Driver</b>"
        image_cells[2].append(Paragraph(attn_text, styles["caption_sm"]))

    img_col_w = CONTENT_WIDTH / 3
    img_table = Table(image_cells, colWidths=[img_col_w] * 3)
    img_table.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 1, LIGHT_BORDER),
        ("LINEBEFORE", (1, 0), (1, -1), 0.5, LIGHT_BORDER),
        ("LINEBEFORE", (2, 0), (2, -1), 0.5, LIGHT_BORDER),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # â”€â”€ SECTION 5: Imaging Findings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(Paragraph("IMAGING FINDINGS", styles["section_heading"]))
    story.append(SectionRule(NAVY, 0.5))
    story.append(Spacer(1, 4))

    findings_data = []
    for mod_label, key in [
        ("Anterior Segment", "anterior"),
        ("Red Glow", "red_glow"),
        ("Slit Lamp", "slit_lamp"),
    ]:
        label_str = mod_label
        if key == dominant_key:
            label_str += " \u2605"
        findings_data.append([
            Paragraph(f"<b>{label_str}</b>", styles["label_bold"]),
            Paragraph(texts[key], styles["body"]),
        ])

    findings_table = Table(findings_data, colWidths=[90, CONTENT_WIDTH - 90])
    findings_style_cmds = [
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]
    for i in range(len(findings_data)):
        bg = ROW_ALT if i % 2 == 0 else colors.white
        findings_style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        if i < len(findings_data) - 1:
            findings_style_cmds.append(
                ("LINEBELOW", (0, i), (-1, i), 0.3, colors.HexColor("#D0D7E2"))
            )
    findings_table.setStyle(TableStyle(findings_style_cmds))
    story.append(findings_table)
    story.append(Spacer(1, 10))

    # â”€â”€ SECTION 6: Classification Probabilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if probabilities:
        story.append(Paragraph("CLASSIFICATION PROBABILITIES", styles["section_heading"]))
        story.append(SectionRule(NAVY, 0.5))
        story.append(Spacer(1, 4))
        story.append(ProbabilityBarsFlowable(probabilities, grade))
        story.append(Spacer(1, 10))

    # â”€â”€ SECTION 7: Conclusion & Recommendation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(Paragraph("CONCLUSION &amp; RECOMMENDATION", styles["section_heading"]))
    story.append(Spacer(1, 4))
    story.append(ConclusionBoxFlowable(texts["conclusion"], styles))
    story.append(Spacer(1, 10))

    # â”€â”€ SECTION 8: Model Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if texts["attention_narrative"] or texts["probability_commentary"]:
        story.append(Paragraph("MODEL ANALYSIS", styles["section_heading"]))
        story.append(SectionRule(NAVY, 0.5))
        story.append(Spacer(1, 4))
        if texts["attention_narrative"]:
            story.append(Paragraph(texts["attention_narrative"], styles["body"]))
            story.append(Spacer(1, 6))
        if texts["probability_commentary"]:
            story.append(Paragraph(texts["probability_commentary"], styles["body_italic"]))

    if not story:
        raise ValueError("PDF story is empty â€” no content was generated")

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Public API: Single-case report
# ---------------------------------------------------------------------------
def generate_pdf_report(username: str, session_record: dict[str, Any]) -> bytes:
    result = session_record["result"]
    analyzed_at: datetime = session_record.get("analyzed_at", datetime.now())

    grade = result.get("grade") or "Unknown"
    severity = SEVERITY_LABELS.get(grade, "Unknown")
    confidence = result.get("confidence")
    confidence_pct = f"{round(confidence * 100)}%" if confidence is not None else "N/A"
    probabilities = result.get("probabilities") or {}
    attention = result.get("attention") or {}
    needs_review = bool(result.get("needs_review", False))
    review_reason = result.get("review_reason")
    model_info = result.get("model_info", "V7 Multimodal CNN | EfficientNet-B2 + ConvNeXt-Tiny")
    patient_name = result.get("patient_name") or ""
    mri_number = result.get("mri_number") or ""

    texts = _get_report_texts(result)

    report_id = result.get("case_id") or "UNKNOWN"
    report_date_str = analyzed_at.strftime("%d-%b-%Y")
    report_time_str = analyzed_at.strftime("%H:%M")

    return _build_report(
        report_id=report_id,
        report_date_str=report_date_str,
        report_time_str=report_time_str,
        patient_name=patient_name,
        mri_number=mri_number,
        username=username,
        model_info=model_info,
        grade=grade,
        severity=severity,
        confidence_pct=confidence_pct,
        probabilities=probabilities,
        attention=attention,
        needs_review=needs_review,
        review_reason=review_reason,
        texts=texts,
        ant_b64=result.get("anterior_segment_base64"),
        rg_b64=result.get("red_glow_base64"),
        sl_b64=result.get("slit_lamp_base64"),
    )


# ---------------------------------------------------------------------------
# Public API: Batch report
# Each case gets the same full-quality layout as a single report.
# Pages are merged by concatenating individual PDF bytes using PyPDF2/pypdf.
# Falls back to a simple per-page canvas render if merge library is absent.
# ---------------------------------------------------------------------------

def _single_case_pdf_bytes(
    username: str,
    result: dict[str, Any],
    generated_at: datetime,
) -> bytes:
    """
    Render one result dict through the full _build_report() pipeline.
    Returns raw PDF bytes for that case.
    """
    grade      = result.get("grade") or "Unknown"
    severity   = SEVERITY_LABELS.get(grade, "Unknown")
    confidence = result.get("confidence")
    confidence_pct = f"{round(confidence * 100)}%" if confidence is not None else "N/A"

    report_id    = result.get("case_id") or result.get("filename") or "UNKNOWN"
    patient_name = result.get("patient_name") or ""
    mri_number   = result.get("mri_number")   or ""
    model_info   = result.get("model_info", "V7 Multimodal CNN")

    # Append eye side to report_id so each page is clearly labelled
    eye_side = result.get("eye_side") or ""
    if eye_side and eye_side not in report_id:
        report_id = f"{report_id} ({eye_side})"

    texts = _get_report_texts(result)

    pdf_bytes = _build_report(
        report_id        = report_id,
        report_date_str  = generated_at.strftime("%d-%b-%Y"),
        report_time_str  = generated_at.strftime("%H:%M"),
        patient_name     = patient_name,
        mri_number       = mri_number,
        username         = username,
        model_info       = model_info,
        grade            = grade,
        severity         = severity,
        confidence_pct   = confidence_pct,
        probabilities    = result.get("probabilities") or {},
        attention        = result.get("attention")     or {},
        needs_review     = bool(result.get("needs_review", False)),
        review_reason    = result.get("review_reason"),
        texts            = texts,
        ant_b64          = result.get("anterior_segment_base64"),
        rg_b64           = result.get("red_glow_base64"),
        sl_b64           = result.get("slit_lamp_base64"),
    )

    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(
            f"_build_report returned invalid PDF bytes for case {result.get('case_id')}"
        )

    return pdf_bytes


def generate_batch_pdf_report(
    username: str,
    results: list[dict[str, Any]],
    generated_at: datetime | None = None,
) -> bytes:
    """
    Generate a full-quality multi-case PDF report.

    Each case is rendered through the same _build_report() pipeline as a
    single report, then all per-case PDFs are merged into one file.
    This means every page has images, imaging findings, probability bars,
    model analysis â€” identical to the single-case report.

    Works with 1 result (single eye) or 2 results (both eyes).
    """
    created_at = generated_at or datetime.now()

    # Filter out any None / empty entries so a single-eye submission works
    valid_results = [r for r in results if r and r.get("grade")]

    if not valid_results:
        # Nothing to render â€” return a minimal error PDF
        buf = io.BytesIO()
        c = Canvas(buf, pagesize=A4)
        c.setFont("Helvetica", 12)
        c.drawString(MARGIN_LEFT, PAGE_H / 2, "No valid classification results to report.")
        c.save()
        buf.seek(0)
        return buf.read()

    if len(valid_results) == 1:
        # Single eye â€” return directly, no merge needed
        result = _single_case_pdf_bytes(username, valid_results[0], created_at)
        if not result or not result.startswith(b"%PDF"):
            raise ValueError("Single-eye PDF generation produced invalid output")
        return result
    return _build_combined_batch_report(username, valid_results, created_at) 

