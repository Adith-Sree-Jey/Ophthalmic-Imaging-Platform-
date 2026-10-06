"""
report_text_engine.py - Ophthalmic Imaging Dynamic Report Text Generator

Generates clinical report text using two strategies:
  1. MedGemma 4B â€” if medgemma_engine is available and gradcam_region_info
     is present in the V7Result, MedGemma writes grounded clinical narrative.
  2. Deterministic fallback â€” if MedGemma is unavailable or fails, the
     original rule-based text is returned. This keeps the PDF working always.

Public API (unchanged â€” pdf_report.py needs no changes):
    generate_report_texts(result: V7Result) -> dict
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data contract â€” matches what pdf_report.py passes in
# ---------------------------------------------------------------------------

@dataclass
class V7Result:
    grade: str
    confidence: float
    probabilities: dict[str, float]
    attention: dict[str, float]
    case_id: Optional[str] = None
    clinician: Optional[str] = None
    # Grad-CAM region info attached by classifier.py â€” used for MedGemma prompts
    gradcam_region_info: dict = field(default_factory=dict)
    # Raw result dict (optionally passed through for MedGemma context)
    raw_result: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MedGemma â€” optional import (non-fatal)
# ---------------------------------------------------------------------------

try:
    from medgemma_engine import generate as _medgemma_generate
    _MEDGEMMA_AVAILABLE = True
except ImportError:
    _MEDGEMMA_AVAILABLE = False


def _safe_medgemma(prompt: str, fallback: str) -> str:
    if not _MEDGEMMA_AVAILABLE:
        return fallback
    try:
        return _medgemma_generate(prompt)
    except Exception as exc:
        print(f"[ReportTextEngine] MedGemma call failed: {exc}")
        return fallback


# ---------------------------------------------------------------------------
# Deterministic knowledge base (fallback when MedGemma unavailable/fails)
# ---------------------------------------------------------------------------

GRADE_SUMMARY = {
    "NS1": (
        "Early nuclear sclerosis (NS1). The lens nucleus shows minimal optical density change. "
        "Visual acuity impact is negligible at this stage and surgical intervention is not indicated."
    ),
    "NS2": (
        "Mild nuclear sclerosis (NS2). Moderate increase in nuclear optical density is present. "
        "The patient may report mild myopic shift or reduced contrast sensitivity under low-light conditions. "
        "Monitoring is recommended; surgery is not yet required."
    ),
    "NS3": (
        "Advanced nuclear sclerosis (NS3). Significant nuclear brunescence and increased light scattering "
        "are evident. Visual function is likely impaired. Surgical consultation is recommended."
    ),
    "NS4": (
        "Severe nuclear sclerosis (NS4). Dense nuclear opacity with marked brunescence or "
        "near-complete opacification of the lens nucleus. Visual acuity is substantially reduced. "
        "Surgical intervention is strongly indicated."
    ),
}

MODALITY_FINDINGS = {
    "anterior": {
        "NS1": (
            "The anterior segment image shows a clear, transparent lens with minimal nuclear haze. "
            "No significant light scatter is observed in the pupillary zone. "
            "The nuclear region appears optically homogeneous."
        ),
        "NS2": (
            "The anterior segment image reveals a subtle increase in nuclear optical density. "
            "Early haze is visible in the central pupillary zone under focal illumination. "
            "The lens capsule and cortex appear clear."
        ),
        "NS3": (
            "The anterior segment image demonstrates moderate nuclear opacification with visible brownish "
            "discolouration in the central lens. Increased light reflection from the nuclear region is noted. "
            "The pupillary reflex is partially attenuated."
        ),
        "NS4": (
            "The anterior segment image shows dense nuclear opacity occupying the central pupillary zone. "
            "Marked brunescence or whitish opacification is evident. "
            "The posterior capsule and fundal details are obscured by the opacity."
        ),
    },
    "red_glow": {
        "NS1": (
            "The red glow (retroillumination) image shows a uniform, bright reflex with no focal "
            "shadowing in the nuclear zone. Lens transmission is unimpeded."
        ),
        "NS2": (
            "The red glow image reveals a faint central shadow or mild attenuation of the fundal reflex "
            "consistent with early nuclear sclerosis. The peripheral lens zones remain clear."
        ),
        "NS3": (
            "The red glow image shows a prominent central shadow with moderate attenuation of the "
            "fundal reflex in the nuclear zone. The boundary between the sclerotic nucleus and the "
            "clearer cortex is visible as a ring of relative brightness."
        ),
        "NS4": (
            "The red glow image demonstrates near-complete loss of the fundal reflex in the central zone. "
            "Dense nuclear opacity blocks retroillumination, producing a dark central shadow that "
            "corresponds to the extent of the opacification."
        ),
    },
    "slit_lamp": {
        "NS1": (
            "The slit lamp image shows a clear, optically empty nuclear zone. The slit beam passes "
            "through the lens with minimal deviation. No brunescence or increased backscatter is detected."
        ),
        "NS2": (
            "The slit lamp image reveals mild nuclear brunescence with a subtle golden-yellow tinge. "
            "Early increased backscatter is visible within the slit beam as it traverses the nucleus."
        ),
        "NS3": (
            "The slit lamp image shows moderate-to-dense nuclear brunescence with brownish discolouration "
            "of the lens nucleus. The slit beam demonstrates increased light scattering and the nuclear "
            "zone appears optically denser than the surrounding cortex."
        ),
        "NS4": (
            "The slit lamp image reveals dense nuclear brunescence or advanced cataractous change. "
            "The nuclear zone appears as a deeply opaque, amber-to-brown structure within the slit beam. "
            "Optical sectioning clearly delineates the hard nucleus from the liquefied or sclerotic cortex."
        ),
    },
}

RECOMMENDATIONS = {
    "NS1": "No surgical intervention required. Annual follow-up with visual acuity monitoring is advised.",
    "NS2": "Conservative management. Six-monthly monitoring recommended. Patient education on expected progression.",
    "NS3": "Surgical consultation recommended. Phacoemulsification with IOL implantation should be discussed.",
    "NS4": "Surgical intervention is strongly indicated. Prompt referral for phacoemulsification or ECCE as appropriate.",
}

_SEVERITY = {"NS1": "Minimal", "NS2": "Mild", "NS3": "Moderate", "NS4": "Severe"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_modality(attention: dict[str, float]) -> tuple[str, float]:
    full = {
        "anterior":  float(attention.get("anterior",  0.0)),
        "red_glow":  float(attention.get("red_glow",  0.0)),
        "slit_lamp": float(attention.get("slit_lamp", 0.0)),
    }
    return max(full.items(), key=lambda x: x[1])


def _modality_label(key: str) -> str:
    return {
        "anterior":  "Anterior Segment",
        "red_glow":  "Red Glow (Retroillumination)",
        "slit_lamp": "Slit Lamp",
    }.get(key, key)


def _gradcam_narrative(region_info: dict) -> str:
    if not region_info:
        return ""
    top     = region_info.get("top_regions", [])
    spread  = region_info.get("activation_spread", "regional")
    intense = region_info.get("activation_intensity", "moderate")
    central = region_info.get("central_fraction", 0.0)
    top_str = ", ".join(top) if top else "unspecified regions"
    central_desc = (
        "predominantly central (axial)"    if central > 0.35
        else "mixed central and peripheral" if central > 0.20
        else "predominantly peripheral"
    )
    return (
        f"Morphological image assessment showed {intense} {spread} activation "
        f"concentrated in the {top_str} region(s) of the anterior segment, "
        f"with {central_desc} distribution (central fraction: {central*100:.0f}%)."
    )


# ---------------------------------------------------------------------------
# Deterministic text functions (used as fallback AND by generate_report_texts)
# ---------------------------------------------------------------------------

def grade_summary_text(result: V7Result) -> str:
    conf_pct = round(result.confidence * 100)
    grade    = result.grade
    if result.confidence >= 0.85:
        conf_descriptor = "high confidence"
    elif result.confidence >= 0.70:
        conf_descriptor = "moderate-to-high confidence"
    elif result.confidence >= 0.55:
        conf_descriptor = "moderate confidence"
    else:
        conf_descriptor = "borderline confidence â€” clinical correlation advised"
    summary        = GRADE_SUMMARY[grade]
    recommendation = RECOMMENDATIONS[grade]
    return (
        f"{summary}\n\n"
        f"The V7 multimodal model assigned this classification with {conf_descriptor} "
        f"({conf_pct}%). {recommendation}"
    )


def modality_finding_text(result: V7Result, modality: str) -> str:
    grade = result.grade
    return MODALITY_FINDINGS[modality][grade]


def attention_narrative_text(result: V7Result) -> str:
    dominant_key, dominant_weight = _dominant_modality(result.attention)
    dominant_label = _modality_label(dominant_key)
    grade          = result.grade
    dominance_rationale = {
        ("anterior",  "NS1"): "At early nuclear sclerosis stages, subtle haze in the anterior segment image is often the most detectable sign before brunescence develops.",
        ("anterior",  "NS2"): "Early nuclear haze in the anterior segment is the most diagnostically informative finding at NS2, as brunescence is still developing.",
        ("anterior",  "NS3"): "At NS3, anterior segment opacification is sufficiently dense to be the primary discriminating feature.",
        ("anterior",  "NS4"): "Dense anterior segment opacity at NS4 provides strong signal for the final grade.",
        ("red_glow",  "NS1"): "Retroillumination is sensitive to early nuclear changes that may be subtle on direct imaging.",
        ("red_glow",  "NS2"): "Fundal reflex attenuation on retroillumination is an early, reliable indicator of nuclear sclerosis progression.",
        ("red_glow",  "NS3"): "The central shadow on retroillumination at NS3 provides high-contrast delineation of the sclerotic nucleus.",
        ("red_glow",  "NS4"): "Near-complete fundal reflex loss is a definitive retroillumination sign of NS4.",
        ("slit_lamp", "NS1"): "The slit beam's optical sectioning capability reveals subtle nuclear density changes not apparent on other modalities at early stages.",
        ("slit_lamp", "NS2"): "Early brunescence is best appreciated on slit lamp optical sectioning.",
        ("slit_lamp", "NS3"): "Slit lamp brunescence and increased backscatter are the hallmark features of NS3 and the most reliable discriminator from NS2.",
        ("slit_lamp", "NS4"): "Dense brunescence and optical sectioning of the hard nucleus are the definitive slit lamp features of NS4.",
    }
    rationale   = dominance_rationale.get((dominant_key, grade), f"The {dominant_label} provided the most diagnostically informative features for this grade.")
    weights_str = ", ".join(f"{_modality_label(k)}: {v:.2f}" for k, v in sorted(result.attention.items(), key=lambda x: -x[1]))
    return (
        f"The model relied primarily on the {dominant_label} image "
        f"(attention weight: {dominant_weight:.2f}) to reach this classification. "
        f"{rationale}\n\nModel attention weights â€” {weights_str}."
    )


def probability_commentary_text(result: V7Result) -> str:
    grade    = result.grade
    probs    = result.probabilities
    conf_pct = round(result.confidence * 100)
    others   = sorted([(g, p) for g, p in probs.items() if g != grade], key=lambda x: -x[1])
    parts    = [f"Classification probability for {grade}: {conf_pct}%."]
    if others and others[0][1] >= 0.15:
        runner_grade, runner_prob = others[0]
        runner_pct = round(runner_prob * 100)
        parts.append(
            f"The next most likely grade was {runner_grade} ({runner_pct}%), "
            f"which represents a meaningful differential. Clinical judgement should be "
            f"applied to borderline cases, particularly if symptoms align more closely "
            f"with {runner_grade} severity."
        )
    elif others and others[0][1] >= 0.08:
        runner_grade, runner_prob = others[0]
        runner_pct = round(runner_prob * 100)
        parts.append(
            f"{runner_grade} probability was {runner_pct}% â€” insufficient to alter the "
            f"primary classification but noted for completeness."
        )
    else:
        parts.append(
            f"The probability distribution is strongly concentrated on {grade}, "
            f"with no other grade exceeding 8%. The classification is unambiguous."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# MedGemma prompt builders â€” 2 calls total (was 5)
# Call 1: combined findings for all 3 modalities in one prompt
# Call 2: conclusion + recommendation
# ---------------------------------------------------------------------------

def _build_combined_findings_prompt(result: V7Result, gradcam_txt: str) -> str:
    """Single prompt covering all 3 modality findings â€” replaces 3 separate calls."""
    grade    = result.grade
    severity = _SEVERITY.get(grade, "Unknown")
    attn     = result.attention
    ant_w    = attn.get("anterior",  0.0)
    rg_w     = attn.get("red_glow",  0.0)
    sl_w     = attn.get("slit_lamp", 0.0)
    dominant = _modality_label(_dominant_modality(attn)[0])

    ant_det  = MODALITY_FINDINGS.get("anterior",  {}).get(grade, "")
    rg_det   = MODALITY_FINDINGS.get("red_glow",  {}).get(grade, "")
    sl_det   = MODALITY_FINDINGS.get("slit_lamp", {}).get(grade, "")

    return textwrap.dedent(f"""
        You are a senior ophthalmologist writing a clinical report.
        Write exactly THREE short paragraphs, one per imaging modality, separated by blank lines.
        Paragraph 1: Anterior Segment findings.
        Paragraph 2: Red Glow (Retroillumination) findings.
        Paragraph 3: Slit Lamp findings.
        Each paragraph must be 2 sentences maximum.
        Integrate the image region analysis naturally. Use formal clinical language.
        Do not mention AI, Grad-CAM, or machine learning. No headings, no greeting, no sign-off.

        Grade: {grade} ({severity} nuclear sclerosis)
        Dominant modality: {dominant}
        Attention weights â€” anterior: {ant_w:.2f}, red glow: {rg_w:.2f}, slit lamp: {sl_w:.2f}
        Image region analysis: {gradcam_txt}
        Expected anterior findings: {ant_det}
        Expected red glow findings: {rg_det}
        Expected slit lamp findings: {sl_det}

        Write only the three paragraphs.
    """).strip()


def _build_conclusion_prompt(result: V7Result, gradcam_txt: str) -> str:
    """Conclusion + recommendation in one prompt."""
    grade    = result.grade
    severity = _SEVERITY.get(grade, "Unknown")
    conf_pct = round(result.confidence * 100)
    rec      = RECOMMENDATIONS.get(grade, "")
    return textwrap.dedent(f"""
        You are a senior ophthalmologist writing a clinical report conclusion.
        Write 2 sentences: one stating the grade and clinical significance,
        one giving the management recommendation.
        Use formal clinical language. Do not mention AI or machine learning.
        No greeting or sign-off.

        Grade: {grade} ({severity} nuclear sclerosis)
        Confidence: {conf_pct}%
        Standard recommendation: {rec}
        Image region analysis: {gradcam_txt}

        Write only the 2-sentence conclusion.
    """).strip()


def _parse_three_paragraphs(text: str, fallbacks: list[str]) -> list[str]:
    """
    Split MedGemma combined findings output into 3 modality paragraphs.
    Falls back to individual deterministic text if parsing fails.
    """
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs[:3]
    # If MedGemma didn't split cleanly, split by sentences
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) >= 3:
        mid = len(sentences) // 3
        return [
            ". ".join(sentences[:mid]) + ".",
            ". ".join(sentences[mid:2*mid]) + ".",
            ". ".join(sentences[2*mid:]) + ".",
        ]
    # Full fallback
    return fallbacks


def _build_verbose_modality_prompt(result: V7Result, modality: str, gradcam_txt: str) -> str:
    grade = result.grade
    severity = _SEVERITY.get(grade, "Unknown")
    attn = result.attention
    weight = float(attn.get(modality, 0.0))
    dominant_key, _ = _dominant_modality(attn)
    dominant_label = _modality_label(dominant_key)
    label = _modality_label(modality)
    expected = MODALITY_FINDINGS.get(modality, {}).get(grade, "")

    modality_context = (
        "This modality is the primary contributor to the fixed grade."
        if modality == dominant_key
        else f"The dominant modality overall is {dominant_label}."
    )

    image_region_line = (
        f"Image region analysis: {gradcam_txt}"
        if modality == "anterior" and gradcam_txt
        else "Image region analysis: keep the wording aligned to the fixed grade and the expected visible findings."
    )

    return textwrap.dedent(f"""
        You are a senior ophthalmologist writing the {label} subsection of a cataract report.
        Write exactly one paragraph of 3 to 4 sentences.
        Start the paragraph with "{label} findings."
        Use formal clinical language and keep the paragraph visually grounded to the fixed grade.
        Do not mention AI, Grad-CAM, attention weights, model confidence, or machine learning.
        Do not add headings, bullets, greetings, or sign-off.

        Fixed grade: {grade} ({severity} nuclear sclerosis)
        Modality: {label}
        Modality attention weight: {weight:.2f}
        Modality context: {modality_context}
        {image_region_line}
        Expected findings to expand naturally: {expected}

        Write only the paragraph.
    """).strip()


def _build_verbose_conclusion_prompt(result: V7Result, gradcam_txt: str) -> str:
    grade = result.grade
    severity = _SEVERITY.get(grade, "Unknown")
    conf_pct = round(result.confidence * 100)
    rec = RECOMMENDATIONS.get(grade, "")
    return textwrap.dedent(f"""
        You are a senior ophthalmologist writing a clinical report conclusion.
        Write one paragraph of 3 sentences.
        The first sentence should state the grade and clinical significance.
        The second sentence should summarise the visible lens changes in concise terms.
        The third sentence should give the management recommendation.
        Use formal clinical language. Do not mention AI or machine learning.
        No greeting or sign-off.

        Grade: {grade} ({severity} nuclear sclerosis)
        Confidence: {conf_pct}%
        Standard recommendation: {rec}
        Image region analysis: {gradcam_txt}

        Write only the paragraph.
    """).strip()


# ---------------------------------------------------------------------------
# Public API â€” called from pdf_report.py
# ---------------------------------------------------------------------------

def generate_report_texts(result: V7Result) -> dict:
    """
    Fuller MedGemma path with one dedicated modality paragraph per call.
    This avoids the compressed combined prompt that can shorten or mis-assign
    findings across rows.
    """
    gradcam_txt = _gradcam_narrative(result.gradcam_region_info)
    use_medgemma = _MEDGEMMA_AVAILABLE

    if use_medgemma:
        print(f"[ReportTextEngine] MedGemma enabled â€” grade={result.grade}, fuller 4-call mode")

        ant_fallback = modality_finding_text(result, "anterior")
        rg_fallback = modality_finding_text(result, "red_glow")
        sl_fallback = modality_finding_text(result, "slit_lamp")

        print("[ReportTextEngine] Generating anterior findingsâ€¦")
        ant_text = _safe_medgemma(
            _build_verbose_modality_prompt(result, "anterior", gradcam_txt),
            fallback=ant_fallback,
        )

        print("[ReportTextEngine] Generating red glow findingsâ€¦")
        rg_text = _safe_medgemma(
            _build_verbose_modality_prompt(result, "red_glow", gradcam_txt),
            fallback=rg_fallback,
        )

        print("[ReportTextEngine] Generating slit lamp findingsâ€¦")
        sl_text = _safe_medgemma(
            _build_verbose_modality_prompt(result, "slit_lamp", gradcam_txt),
            fallback=sl_fallback,
        )

        print("[ReportTextEngine] Generating conclusionâ€¦")
        conclusion_text = _safe_medgemma(
            _build_verbose_conclusion_prompt(result, gradcam_txt),
            fallback=grade_summary_text(result),
        )

        attn_text = attention_narrative_text(result)
        prob_text = probability_commentary_text(result)

        print("[ReportTextEngine] Done â€” 4 MedGemma calls complete.")

    else:
        print("[ReportTextEngine] MedGemma not available â€” using deterministic text")
        conclusion_text = grade_summary_text(result)
        ant_text = modality_finding_text(result, "anterior")
        rg_text = modality_finding_text(result, "red_glow")
        sl_text = modality_finding_text(result, "slit_lamp")
        attn_text = attention_narrative_text(result)
        prob_text = probability_commentary_text(result)

    return {
        "conclusion": conclusion_text,
        "anterior": ant_text,
        "red_glow": rg_text,
        "slit_lamp": sl_text,
        "attention_narrative": attn_text,
        "probability_commentary": prob_text,
    }


# ---------------------------------------------------------------------------
# Quick test â€” run: python report_text_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = V7Result(
        grade="NS3",
        confidence=0.79,
        probabilities={"NS1": 0.03, "NS2": 0.04, "NS3": 0.79, "NS4": 0.14},
        attention={"anterior": 0.22, "red_glow": 0.16, "slit_lamp": 0.62},
        case_id="CASE-2024-0042",
        clinician="Dr. A. Ramesh",
        gradcam_region_info={
            "top_regions": ["central", "superior"],
            "activation_spread": "focal",
            "activation_intensity": "high",
            "central_fraction": 0.48,
        },
    )
    texts = generate_report_texts(sample)
    for key, val in texts.items():
        print(f"\n{'='*60}\n{key.upper()}\n{'='*60}\n{val}")

