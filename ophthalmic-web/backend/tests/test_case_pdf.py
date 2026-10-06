"""Regression test for case-review PDF clinician attribution (Finding 4).

Previously the PDF hardcoded the clinician as the literal string "admin".
It must now reflect the actual reviewing clinician.
"""
from __future__ import annotations

import datetime as dt

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - pypdf is optional
    PdfReader = None

from database import Case, CaseModuleResult, InferenceJob, Patient
import main


def _make_case(db, seed_user):
    reviewer = seed_user("dr_smith", role="doctor", password="VeryStrongPass123!")
    patient = Patient(
        mri_number="MRI-PDF",
        patient_name="Jane Doe",
        created_by=reviewer.id,
        created_at=dt.datetime.utcnow(),
    )
    db.add(patient)
    db.flush()
    db.add(
        InferenceJob(
            id="pdf-job",
            status="done",
            result_path="/jobs/pdf-job/result",
            created_at=dt.datetime.utcnow(),
            created_by=reviewer.id,
        )
    )
    case = Case(
        id="pdf-case",
        patient_id=patient.id,
        created_by=reviewer.id,
        status="report_approved",
        module_types=["cataract"],
        priority="routine",
        report_status="approved",
        report_approved_by=reviewer.id,
        created_at=dt.datetime.utcnow(),
    )
    db.add(case)
    db.flush()
    db.add(
        CaseModuleResult(
            case_id=case.id,
            module="cataract",
            job_id="pdf-job",
            model_prediction={"grade": "NS3", "probabilities": {"NS3": 0.9}},
            model_confidence=0.9,
            model_grade="NS3",
            review_status="approved",
            reviewer_id=reviewer.id,
            reviewed_at=dt.datetime.utcnow(),
        )
    )
    db.commit()
    return case


def _pdf_text(path):
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            pass
    # Fall back to raw bytes scan if text extraction is unavailable.
    return path.read_bytes().decode("latin-1", errors="ignore")


def test_case_pdf_uses_real_clinician_not_hardcoded_admin(db, seed_user, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "JOB_RESULTS_DIR", tmp_path)
    case = _make_case(db, seed_user)

    pdf_path = main._build_case_review_pdf(case, db)
    text = _pdf_text(pdf_path)

    assert "dr_smith" in text, "PDF should name the actual reviewing clinician"
    # The clinician field must not be the old hardcoded literal.
    assert "Clinician:admin" not in text.replace(" ", "")
