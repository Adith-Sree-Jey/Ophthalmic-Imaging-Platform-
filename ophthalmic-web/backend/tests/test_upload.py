from __future__ import annotations

from database import InferenceJob


def test_upload_valid_image_returns_job_id(
    client,
    db,
    auth_headers,
    tiny_png,
    no_op_glaucoma_job,
):
    response = client.post(
        "/glaucoma/predict",
        headers=auth_headers("admin"),
        files={"file": ("eye.png", tiny_png, "image/png")},
        data={
            "case_id": "CASE-123",
            "patient_name": "Test Patient",
            "mri_number": "MRI-123",
            "eye_side": "OD",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["job_id"]

    db.expire_all()
    job = db.query(InferenceJob).filter(InferenceJob.id == payload["job_id"]).first()
    assert job is not None
    assert job.status == "pending"


def test_upload_invalid_extension_rejected(
    client,
    auth_headers,
    no_op_glaucoma_job,
):
    response = client.post(
        "/glaucoma/predict",
        headers=auth_headers("admin"),
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
        data={"case_id": "CASE-124", "mri_number": "MRI-124", "eye_side": "OD"},
    )

    assert response.status_code == 400


def test_upload_oversized_file_rejected(
    client,
    auth_headers,
    no_op_glaucoma_job,
):
    huge_png = b"\x89PNG\r\n\x1a\n" + (b"0" * (10 * 1024 * 1024 + 1))

    response = client.post(
        "/glaucoma/predict",
        headers=auth_headers("admin"),
        files={"file": ("huge.png", huge_png, "image/png")},
        data={"case_id": "CASE-125", "mri_number": "MRI-125", "eye_side": "OD"},
    )

    assert response.status_code == 413
