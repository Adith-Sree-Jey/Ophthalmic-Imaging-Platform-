from __future__ import annotations

import json
import os
import secrets
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("JWT_SECRET", secrets.token_hex(32))
os.environ.setdefault("MEDGEMMA_MAX_TOKENS", "512")
os.environ.setdefault("MEDGEMMA_TEMPERATURE", "0.2")

import auth
import main
from auth import create_access_token, hash_password
from database import Base, Classification, InferenceJob, Patient, User, Visit


@pytest.fixture()
def db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def register_mssql_utc_default(dbapi_connection, _connection_record):
        dbapi_connection.create_function("GETUTCDATE", 0, lambda: datetime.utcnow().isoformat(" "))

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


@pytest.fixture()
def seed_user(db):
    def _seed_user(
        username: str,
        *,
        role: str = "doctor",
        password: str = "VeryStrongPass123!",
        is_active: bool = True,
    ) -> User:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            user = User(
                username=username,
                hashed_password=hash_password(password),
                role=role,
                is_active=is_active,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    return _seed_user


@pytest.fixture()
def auth_headers(seed_user):
    def _headers(username: str = "admin") -> dict[str, str]:
        role = "admin" if username == "admin" else "doctor"
        user = seed_user(username, role=role)
        token = create_access_token(
            {"sub": str(user.id), "username": user.username, "role": user.role}
        )
        return {"Authorization": f"Bearer {token}"}

    return _headers


@pytest.fixture()
def seed_visit(db, seed_user):
    def _seed_visit(
        *,
        mri_number: str,
        graded_by: str,
        patient_name: str = "Test Patient",
        eye_side: str = "OD",
        module: str = "cataract",
        case_id: str = "CASE-001",
        with_classification: bool = True,
    ) -> Patient:
        role = "admin" if graded_by == "admin" else "doctor"
        owner = seed_user(graded_by, role=role)
        patient = Patient(
            mri_number=mri_number,
            patient_name=patient_name,
            created_by=owner.id,
        )
        db.add(patient)
        db.flush()

        visit = Visit(
            patient_id=patient.id,
            case_id=case_id,
            eye_side=eye_side,
            module=module,
            result_summary=f"{module} result",
            graded_by=graded_by,
        )
        db.add(visit)
        db.flush()

        if with_classification and module == "cataract":
            db.add(
                Classification(
                    visit_id=visit.id,
                    grade="NS2",
                    grade_index=2,
                    predicted_class="NS2",
                    confidence=0.91,
                    probabilities_json=json.dumps(
                        {"NS1": 0.02, "NS2": 0.91, "NS3": 0.05, "NS4": 0.02}
                    ),
                    prob_ns1=0.02,
                    prob_ns2=0.91,
                    prob_ns3=0.05,
                    prob_ns4=0.02,
                    attn_anterior=0.5,
                    attn_red_glow=0.3,
                    attn_slit_lamp=0.2,
                    needs_review=False,
                    pupil_crop_applied=False,
                    recommendation="Follow up.",
                    severity="Mild",
                    model_info="test-model",
                )
            )

        db.commit()
        return patient

    return _seed_visit


@pytest.fixture()
def tiny_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
        b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture()
def no_op_glaucoma_job(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "run_glaucoma_prediction_job", lambda *args, **kwargs: None)


__all__ = [
    "Classification",
    "InferenceJob",
]
