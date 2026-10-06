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


GLAUCOMA_REPORT_PROMPT = """
You are a clinical AI assistant supporting ophthalmologists at Ophthalmic Imaging.
Generate a structured clinical report for the following glaucoma classification result.

Patient Information:
  Name       : {patient_name}
  MRI Number : {mri_number}
  UID        : {uid}
  Eye Side   : {eye_side}

Classification Result:
  Finding        : {predicted_class}
  Confidence     : {confidence}%
  Glaucoma Prob  : {glaucoma_prob}%
  Non-Glaucoma   : {non_glaucoma_prob}%
  Review Flag    : {needs_review}
  Review Reason  : {review_reason}

Write a structured clinical report with these exact sections:
1. Summary - one sentence finding
2. Classification Detail - interpret the confidence and probability values clinically
3. Clinical Recommendation - next steps based on finding and confidence level
4. Review Note - only if needs_review is true, explain why clinical review is needed; omit this section entirely if needs_review is false
5. Disclaimer - standard AI-assisted diagnosis disclaimer

Be concise, clinical, and factual. Do not speculate beyond the provided data.
""".strip()
