"""Regression tests for the cross-user authorization fixes.

These cover the IDOR / missing-ownership gaps confirmed in the audit:
  * /jobs/{id} and /jobs/{id}/result must not be readable across users
  * case detail / list / review actions must be owner/reviewer/admin only
  * /audit-log and /audit-log/export must be admin-only

Denial paths short-circuit before any audit-log write, so they remain
portable across SQLite (tests) and MSSQL (production).
"""
from __future__ import annotations

import datetime as dt

import pytest

from database import (
    AuditLog,
    Case,
    CaseModuleResult,
    InferenceJob,
    Patient,
)


def _seed_case_chain(db, seed_user, owner_username, *, job_id, case_id, reviewer_id=None):
    """Create owner + patient + job + case + module_result owned by one user."""
    owner = seed_user(owner_username, role="doctor")
    patient = Patient(
        mri_number=f"MRI-{owner_username}",
        patient_name=f"Patient {owner_username}",
        created_by=owner.id,
        created_at=dt.datetime.utcnow(),
    )
    db.add(patient)
    db.flush()

    db.add(
        InferenceJob(
            id=job_id,
            status="done",
            result_path=f"/jobs/{job_id}/result",
            created_at=dt.datetime.utcnow(),
            created_by=owner.id,
        )
    )
    case = Case(
        id=case_id,
        patient_id=patient.id,
        created_by=owner.id,
        assigned_reviewer=reviewer_id,
        status="pending_review",
        module_types=["cataract"],
        priority="routine",
        report_status="pending",
        created_at=dt.datetime.utcnow(),
    )
    db.add(case)
    db.flush()
    db.add(
        CaseModuleResult(
            case_id=case.id,
            module="cataract",
            job_id=job_id,
            model_prediction={"grade": "NS3", "probabilities": {"NS3": 0.9}},
            model_confidence=0.9,
            model_grade="NS3",
            review_status="pending",
        )
    )
    db.commit()
    return owner, patient, case


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

def test_job_status_blocked_for_non_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-1", case_id="case-1")
    res = client.get("/jobs/job-1", headers=auth_headers("doctor_b"))
    assert res.status_code == 403


def test_job_result_blocked_for_non_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-2", case_id="case-2")
    res = client.get("/jobs/job-2/result", headers=auth_headers("doctor_b"))
    assert res.status_code == 403


def test_job_status_allowed_for_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-3", case_id="case-3")
    res = client.get("/jobs/job-3", headers=auth_headers("doctor_a"))
    assert res.status_code == 200
    assert res.json()["status"] == "done"


def test_job_status_allowed_for_admin(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-4", case_id="case-4")
    res = client.get("/jobs/job-4", headers=auth_headers("admin"))
    assert res.status_code == 200


def test_legacy_job_without_owner_denied_to_non_admin(client, auth_headers, seed_user, db):
    # Rows created before the migration have created_by IS NULL -> fail closed.
    seed_user("doctor_a", role="doctor")
    db.add(
        InferenceJob(
            id="legacy-job",
            status="done",
            result_path="/jobs/legacy-job/result",
            created_at=dt.datetime.utcnow(),
            created_by=None,
        )
    )
    db.commit()
    assert client.get("/jobs/legacy-job", headers=auth_headers("doctor_a")).status_code == 403
    assert client.get("/jobs/legacy-job", headers=auth_headers("admin")).status_code == 200


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def test_case_detail_blocked_for_non_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-5", case_id="case-5")
    res = client.get("/cases/case-5", headers=auth_headers("doctor_b"))
    assert res.status_code == 403


def test_case_detail_allowed_for_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-6", case_id="case-6")
    res = client.get("/cases/case-6", headers=auth_headers("doctor_a"))
    assert res.status_code == 200
    assert res.json()["patient_summary"]["mri_number"] == "MRI-doctor_a"


def test_case_detail_allowed_for_assigned_reviewer(client, auth_headers, seed_user, db):
    reviewer = seed_user("doctor_b", role="doctor")
    _seed_case_chain(
        db, seed_user, "doctor_a", job_id="job-7", case_id="case-7", reviewer_id=reviewer.id
    )
    res = client.get("/cases/case-7", headers=auth_headers("doctor_b"))
    assert res.status_code == 200


def test_list_cases_excludes_other_users_cases(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-8", case_id="case-8")
    _seed_case_chain(db, seed_user, "doctor_b", job_id="job-9", case_id="case-9")
    res = client.get("/cases", headers=auth_headers("doctor_a"))
    assert res.status_code == 200
    ids = {item["id"] for item in res.json()["items"]}
    assert ids == {"case-8"}


def test_list_cases_admin_sees_all(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-10", case_id="case-10")
    _seed_case_chain(db, seed_user, "doctor_b", job_id="job-11", case_id="case-11")
    res = client.get("/cases", headers=auth_headers("admin"))
    ids = {item["id"] for item in res.json()["items"]}
    assert {"case-10", "case-11"} <= ids


def test_approve_result_blocked_for_non_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-12", case_id="case-12")
    res = client.post(
        "/cases/case-12/results/cataract/approve",
        headers=auth_headers("doctor_b"),
        json={"notes": "should be blocked"},
    )
    assert res.status_code == 403


def test_override_result_blocked_for_non_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-13", case_id="case-13")
    res = client.post(
        "/cases/case-13/results/cataract/override",
        headers=auth_headers("doctor_b"),
        json={"clinician_grade": "NS4", "clinician_notes": "blocked"},
    )
    assert res.status_code == 403


def test_report_download_blocked_for_non_owner(client, auth_headers, seed_user, db):
    _seed_case_chain(db, seed_user, "doctor_a", job_id="job-14", case_id="case-14")
    res = client.get("/cases/case-14/report/download", headers=auth_headers("doctor_b"))
    assert res.status_code == 403


# --------------------------------------------------------------------------
# Audit log (admin-only)
# --------------------------------------------------------------------------

@pytest.fixture()
def _seed_audit_row(db, seed_user):
    actor = seed_user("doctor_a", role="doctor")
    db.add(
        AuditLog(
            actor_id=actor.id,
            action="module_result.approved",
            before_state={"phi": "secret"},
            after_state={"phi": "secret"},
            timestamp=dt.datetime.utcnow(),
        )
    )
    db.commit()


def test_audit_log_blocked_for_non_admin(client, auth_headers, _seed_audit_row):
    res = client.get("/audit-log", headers=auth_headers("doctor_b"))
    assert res.status_code == 403


def test_audit_log_allowed_for_admin(client, auth_headers, _seed_audit_row):
    res = client.get("/audit-log", headers=auth_headers("admin"))
    assert res.status_code == 200
    assert res.json()["total"] >= 1


def test_audit_export_blocked_for_non_admin(client, auth_headers, _seed_audit_row):
    res = client.get("/audit-log/export", headers=auth_headers("doctor_b"))
    assert res.status_code == 403
