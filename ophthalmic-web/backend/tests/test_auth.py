from __future__ import annotations


def test_login_with_valid_credentials(client, seed_user):
    seed_user("test-admin", role="admin", password="VeryStrongPass123!")

    response = client.post(
        "/token",
        data={"username": "test-admin", "password": "VeryStrongPass123!"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["access_token"]


def test_login_with_wrong_password(client, seed_user):
    seed_user("test-admin", role="admin", password="VeryStrongPass123!")

    response = client.post(
        "/token",
        data={"username": "test-admin", "password": "wrong-password"},
    )

    assert response.status_code == 401


def test_login_with_unknown_user(client):
    response = client.post(
        "/token",
        data={"username": "unknown-user", "password": "irrelevant"},
    )

    assert response.status_code == 401


def test_protected_route_without_token(client):
    response = client.get("/patients")

    assert response.status_code == 401


def test_protected_route_with_valid_token(client, auth_headers):
    response = client.get("/patients", headers=auth_headers("admin"))

    assert response.status_code == 200
