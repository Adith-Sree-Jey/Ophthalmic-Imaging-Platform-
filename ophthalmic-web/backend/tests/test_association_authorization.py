from __future__ import annotations

import datetime as dt

import pytest

import main
from database import Case, Patient, get_or_create_case


INFERENCE_ENDPOINTS = [
    "/classify",
    "/retina/probability",
    "/retina-segmentation",
    "/glaucoma/predict",
    "/glaucoma/report",
]


@pytest.fixture()
def no_op_inference_jobs(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "run_cataract_classification_job",
        "run_retina_probability_job",
        "run_retina_segmentation_job",
        "run_glaucoma_prediction_job",
        "run_glaucoma_report_job",
    ):
        monkeypatch.setattr(main, name, lambda *args, **kwargs: None)


@pytest.fixture()
def owned_patient_case(db, seed_user):
    owner = seed_user("doctor_a", role="doctor")
    patient = Patient(
        mri_number="MRI-DOCTOR-A",
        patient_name="Patient A",
        created_by=owner.id,
    )
    db.add(patient)
    db.flush()
    case = Case(
        id="CASE-DOCTOR-A",
        patient_id=patient.id,
        created_by=owner.id,
        status="pending_review",
        module_types=["cataract"],
        priority="routine",
        report_status="pending",
        created_at=dt.datetime.utcnow(),
    )
    db.add(case)
    db.commit()
    return owner, patient, case


def _post_inference(client, endpoint, headers, tiny_png, *, mri_number, case_id):
    data = {
        "case_id": case_id,
        "patient_name": "Association Test",
        "mri_number": mri_number,
        "eye_side": "OD",
    }
    if endpoint == "/classify":
        data["types"] = "anterior_segment"
        files = {"files": ("eye.png", tiny_png, "image/png")}
    else:
        files = {"file": ("eye.png", tiny_png, "image/png")}
    if endpoint == "/glaucoma/report":
        data["patient_id"] = "REPORT-UID"
    return client.post(endpoint, headers=headers, files=files, data=data)


@pytest.mark.parametrize("endpoint", INFERENCE_ENDPOINTS)
def test_inference_rejects_other_users_mri(
    endpoint,
    client,
    auth_headers,
    tiny_png,
    owned_patient_case,
    no_op_inference_jobs,
):
    response = _post_inference(
        client,
        endpoint,
        auth_headers("doctor_b"),
        tiny_png,
        mri_number=owned_patient_case[1].mri_number,
        case_id="NEW-CASE",
    )
    assert response.status_code == 403


@pytest.mark.parametrize("endpoint", INFERENCE_ENDPOINTS)
def test_inference_rejects_other_users_case(
    endpoint,
    client,
    auth_headers,
    tiny_png,
    owned_patient_case,
    no_op_inference_jobs,
):
    response = _post_inference(
        client,
        endpoint,
        auth_headers("doctor_b"),
        tiny_png,
        mri_number="MRI-NEW-FOR-CASE-CHECK",
        case_id=owned_patient_case[2].id,
    )
    assert response.status_code == 403


@pytest.mark.parametrize("endpoint", INFERENCE_ENDPOINTS)
def test_inference_allows_owner_identifiers(
    endpoint,
    client,
    auth_headers,
    tiny_png,
    owned_patient_case,
    no_op_inference_jobs,
):
    response = _post_inference(
        client,
        endpoint,
        auth_headers("doctor_a"),
        tiny_png,
        mri_number=owned_patient_case[1].mri_number,
        case_id=owned_patient_case[2].id,
    )
    assert response.status_code == 200


@pytest.mark.parametrize("endpoint", INFERENCE_ENDPOINTS)
def test_inference_allows_new_mri(
    endpoint,
    client,
    auth_headers,
    tiny_png,
    no_op_inference_jobs,
):
    response = _post_inference(
        client,
        endpoint,
        auth_headers("doctor_b"),
        tiny_png,
        mri_number=f"MRI-NEW-{endpoint}",
        case_id=f"CASE-NEW-{endpoint}",
    )
    assert response.status_code == 200


def test_create_case_rejects_other_users_patient(client, auth_headers, owned_patient_case):
    response = client.post(
        "/cases",
        headers=auth_headers("doctor_b"),
        json={"patient_id": owned_patient_case[1].id, "module_types": ["cataract"]},
    )
    assert response.status_code == 403


def test_create_case_allows_patient_owner(client, auth_headers, owned_patient_case):
    response = client.post(
        "/cases",
        headers=auth_headers("doctor_a"),
        json={"patient_id": owned_patient_case[1].id, "module_types": ["cataract"]},
    )
    assert response.status_code == 200


def test_get_or_create_case_defensively_rejects_wrong_user(db, seed_user, owned_patient_case):
    other = seed_user("doctor_b", role="doctor")
    with pytest.raises(PermissionError):
        get_or_create_case(
            db,
            patient=owned_patient_case[1],
            case_id=owned_patient_case[2].id,
            created_by=other.id,
            module_types=["glaucoma"],
        )
