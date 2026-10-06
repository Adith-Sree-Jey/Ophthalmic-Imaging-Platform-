from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

import main
from database import Case, CaseModuleResult, Classification, InferenceJob, Patient, Visit


def test_stats_overview_is_scoped_for_doctor_and_global_for_admin(
    client,
    auth_headers,
    seed_visit,
    db,
):
    owner_patient = seed_visit(
        mri_number="MRI-STATS-A",
        patient_name="Stats Patient A",
        graded_by="doctor_a",
        case_id="STATS-A",
    )
    other_patient = seed_visit(
        mri_number="MRI-STATS-B",
        patient_name="Stats Patient B",
        graded_by="doctor_b",
        case_id="STATS-B",
    )
    for patient, pdf_path in ((owner_patient, "owner.pdf"), (other_patient, "other.pdf")):
        visit = db.query(Visit).filter(Visit.patient_id == patient.id).one()
        visit.report_generated = True
        visit.report_timestamp = dt.datetime.utcnow()
        visit.pdf_path = pdf_path
    owner_visit = db.query(Visit).filter(Visit.patient_id == owner_patient.id).one()
    db.query(Classification).filter(
        Classification.visit_id == owner_visit.id
    ).one().needs_review = True
    db.commit()

    doctor_payload = client.get("/stats/overview", headers=auth_headers("doctor_a")).json()
    assert doctor_payload["total_patients"] == 1
    assert doctor_payload["needs_review"] == 1
    assert doctor_payload["reports_generated"] == 1
    assert {row["mri_number"] for row in doctor_payload["today_cases"]} == {"MRI-STATS-A"}
    assert {row["pdf_path"] for row in doctor_payload["latest_reports"]} == {"owner.pdf"}

    admin_payload = client.get("/stats/overview", headers=auth_headers("admin")).json()
    assert admin_payload["total_patients"] == 2
    assert {row["mri_number"] for row in admin_payload["today_cases"]} == {
        "MRI-STATS-A",
        "MRI-STATS-B",
    }
    assert admin_payload["needs_review"] == 1
    assert admin_payload["reports_generated"] == 2


def test_stats_boolean_filter_compiles_for_sql_server_bit_columns():
    from sqlalchemy import select
    from sqlalchemy.dialects import mssql

    statement = select(Classification.id).where(
        main._database_true(Classification.needs_review)
    )
    sql = str(
        statement.compile(
            dialect=mssql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "needs_review = 1" in sql
    assert "needs_review IS 1" not in sql


def test_glaucoma_report_download_is_admin_only(
    client,
    auth_headers,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(main, "_resolve_glaucoma_report_download_path", lambda _path: report_path)

    assert client.get(
        "/glaucoma/report/download",
        params={"pdf_path": "report.pdf"},
        headers=auth_headers("doctor_a"),
    ).status_code == 403
    assert client.get(
        "/glaucoma/report/download",
        params={"pdf_path": "report.pdf"},
        headers=auth_headers("admin"),
    ).status_code == 200


def _add_review_case(db, owner, *, suffix, assigned_reviewer=None):
    patient = Patient(
        mri_number=f"MRI-QUEUE-{suffix}",
        patient_name=f"Queue Patient {suffix}",
        created_by=owner.id,
    )
    db.add(patient)
    db.flush()
    job = InferenceJob(
        id=f"JOB-QUEUE-{suffix}",
        status="done",
        created_by=owner.id,
        created_at=dt.datetime.utcnow(),
    )
    db.add(job)
    case = Case(
        id=f"CASE-QUEUE-{suffix}",
        patient_id=patient.id,
        created_by=owner.id,
        assigned_reviewer=assigned_reviewer,
        status="under_review" if assigned_reviewer else "pending_review",
        module_types=["cataract"],
        priority="routine",
        report_status="pending",
        created_at=dt.datetime.utcnow(),
    )
    db.add(case)
    db.flush()
    db.add(CaseModuleResult(
        case_id=case.id,
        module="cataract",
        job_id=job.id,
        model_prediction={"grade": "NS2"},
        model_confidence=0.8,
        model_grade="NS2",
        review_status="pending",
        flagged_for_review=True,
    ))
    db.commit()
    return case


def test_review_queue_is_owner_or_assignee_scoped(client, auth_headers, seed_user, db):
    doctor_a = seed_user("doctor_a", role="doctor")
    doctor_b = seed_user("doctor_b", role="doctor")
    own = _add_review_case(db, doctor_a, suffix="OWN")
    other = _add_review_case(db, doctor_b, suffix="OTHER")
    assigned = _add_review_case(db, doctor_b, suffix="ASSIGNED", assigned_reviewer=doctor_a.id)

    doctor_ids = {
        row["id"]
        for row in client.get("/review-queue", headers=auth_headers("doctor_a")).json()
    }
    assert doctor_ids == {own.id, assigned.id}

    admin_ids = {
        row["id"]
        for row in client.get("/review-queue", headers=auth_headers("admin")).json()
    }
    assert {own.id, other.id, assigned.id} <= admin_ids


def test_workspace_status_hides_internal_details_from_non_admin(
    client,
    auth_headers,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(main, "get_glaucoma_module_status", lambda: {
        "available": True,
        "status": "Active workspace",
        "reason": "internal diagnostic",
        "checkpoint_path": "C:/secret/model.pth",
        "mode": "hybrid",
    })
    doctor_status = client.get("/workspace-status", headers=auth_headers("doctor_a")).json()["glaucoma"]
    assert doctor_status == {"available": True, "status": "Active workspace"}

    admin_status = client.get("/workspace-status", headers=auth_headers("admin")).json()["glaucoma"]
    assert admin_status["checkpoint_path"] == "C:/secret/model.pth"
    assert admin_status["reason"] == "internal diagnostic"


@pytest.mark.parametrize("path", [
    "/validation-report/cataract",
    "/validation-report/cataract/pdf",
    "/validation-report/cataract/history",
])
def test_validation_routes_reject_non_admin(client, auth_headers, path):
    assert client.get(path, headers=auth_headers("doctor_a")).status_code == 403


def test_validation_routes_allow_admin(
    client,
    auth_headers,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeValidationReport:
        def __init__(self, _db, generated_by="system"):
            self.generated_by = generated_by

        def generate(self, module, version):
            return SimpleNamespace(module=module, model_version=version, generated_by=self.generated_by)

        def render_pdf(self, _report, _bins):
            return b"%PDF-1.4 validation"

    monkeypatch.setattr(main, "ModuleValidationReport", FakeValidationReport)
    assert client.get("/validation-report/cataract", headers=auth_headers("admin")).status_code == 200
    assert client.get("/validation-report/cataract/pdf", headers=auth_headers("admin")).status_code == 200
    assert client.get("/validation-report/cataract/history", headers=auth_headers("admin")).status_code == 200
