from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import auth
import main
from database import Base


NUMERIC_FIELDS = (
    "total_patients",
    "this_week",
    "needs_review",
    "reports_generated",
    "total_cases",
)


@pytest.fixture()
def db():
    """Shared-connection in-memory SQLite database for endpoint regression tests."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def register_mssql_utc_default(dbapi_connection, _connection_record):
        dbapi_connection.create_function(
            "GETUTCDATE", 0, lambda: datetime.utcnow().isoformat(" ")
        )

    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)
    session = testing_session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def client(db, monkeypatch: pytest.MonkeyPatch):
    main.SESSION_RESULTS.clear()
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "preload_glaucoma_model", lambda: None)

    def override_get_db():
        yield db

    main.app.dependency_overrides[main.get_db] = override_get_db
    main.app.dependency_overrides[auth.get_db] = override_get_db
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()


def _assert_overview_schema(payload: dict) -> None:
    for field in NUMERIC_FIELDS:
        assert field in payload
        assert isinstance(payload[field], int)
        assert not isinstance(payload[field], bool)
    assert isinstance(payload["today_cases"], list)
    assert isinstance(payload["latest_reports"], list)


def test_stats_overview_empty_database_returns_zeroed_200(client, auth_headers):
    response = client.get("/stats/overview", headers=auth_headers("admin"))

    assert response.status_code == 200
    payload = response.json()
    _assert_overview_schema(payload)
    assert all(payload[field] == 0 for field in NUMERIC_FIELDS)
    assert payload["today_cases"] == []
    assert payload["latest_reports"] == []


def test_stats_overview_minimal_database_returns_numeric_schema(
    client, auth_headers, seed_visit
):
    seed_visit(
        mri_number="MRI-STATS-MINIMAL",
        patient_name="Stats Minimal",
        graded_by="admin",
        case_id="STATS-MINIMAL",
    )

    response = client.get("/stats/overview", headers=auth_headers("admin"))

    assert response.status_code == 200
    payload = response.json()
    _assert_overview_schema(payload)
    assert payload["total_patients"] == 1
    assert payload["total_cases"] == 1
    assert payload["this_week"] == 1
    assert payload["needs_review"] == 0
    assert payload["reports_generated"] == 0


def test_stats_overview_requires_authentication(client):
    response = client.get("/stats/overview")

    assert response.status_code in {401, 403}
    assert response.status_code != 500

