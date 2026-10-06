from __future__ import annotations


def test_user_cannot_see_other_users_patients(client, auth_headers, seed_visit):
    seed_visit(mri_number="MRI-A", graded_by="doctor_a", patient_name="Alice")
    seed_visit(mri_number="MRI-B", graded_by="doctor_b", patient_name="Bob")

    response = client.get("/patients", headers=auth_headers("doctor_a"))

    assert response.status_code == 200
    payload = response.json()
    assert [patient["mri_number"] for patient in payload] == ["MRI-A"]


def test_admin_can_see_all_patients(client, auth_headers, seed_visit):
    seed_visit(mri_number="MRI-A", graded_by="doctor_a", patient_name="Alice")
    seed_visit(mri_number="MRI-B", graded_by="doctor_b", patient_name="Bob")

    response = client.get("/patients", headers=auth_headers("admin"))

    assert response.status_code == 200
    assert {patient["mri_number"] for patient in response.json()} == {"MRI-A", "MRI-B"}


def test_patient_history_unauthorized(client, auth_headers, seed_visit):
    seed_visit(mri_number="MRI-B", graded_by="doctor_b", patient_name="Bob")

    response = client.get("/history/MRI-B", headers=auth_headers("doctor_a"))

    assert response.status_code == 403
