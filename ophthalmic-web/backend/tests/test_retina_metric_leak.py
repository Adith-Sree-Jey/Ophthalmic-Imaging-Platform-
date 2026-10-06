from __future__ import annotations


def test_retina_validation_report_is_blocked(client, auth_headers):
    response = client.get(
        "/validation-report/retina",
        headers=auth_headers("admin"),
    )

    assert response.status_code == 409
    assert "invalidated" in response.json()["detail"].lower()


def test_retina_validation_pdf_is_blocked(client, auth_headers):
    response = client.get(
        "/validation-report/retina/pdf",
        headers=auth_headers("admin"),
    )

    assert response.status_code == 409
    assert "invalidated" in response.json()["detail"].lower()

