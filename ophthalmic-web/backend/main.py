from __future__ import annotations

import base64
import csv
import json
import io
import shutil
import tempfile
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
import uuid

import cv2
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, true
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user, verify_password
from calibration import TemperatureScaler
from classifier import ClassifierService
from database import (
    AuditLog,
    CalibrationConfig,
    Case,
    CaseModuleResult,
    Classification,
    InferenceJob,
    Patient,
    ReviewThresholdConfig,
    SessionLocal,
    User,
    ValidationReportSnapshot,
    Visit,
    case_report_hash,
    get_db,
    get_patient_history,
    init_db,
    log_audit_event,
    normalize_confidence,
    normalize_module_name,
    save_full_result,
    save_retina_result,
)
from glaucoma_service import (
    GLAUCOMA_TEMP_DIR,
    generate_report_from_file,
    get_glaucoma_module_status,
    predict_glaucoma_file,
    preload_glaucoma_model,
)
from glaucoma.src.pdf_generator import generate_glaucoma_pdf
from image_classifier import classify_image_type
from model_performance import calibration_drift, override_analysis, performance_overview, reviewer_agreement, throughput_metrics
from pdf_report import generate_batch_pdf_report, generate_pdf_report
from retina_segmentation.src.infer import predict_probability_map
from retina_segmentation_service import segment_retina_image
from review_thresholds import ReviewThresholdEngine
from validation_report import ModuleValidationReport


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SESSION_RESULTS: dict[str, dict[str, Any]] = {}
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RETINA_MODULE_DIR = WORKSPACE_ROOT / "retina_segmentation"
RETINA_TEMP_DIR = RETINA_MODULE_DIR / "temp"
GLAUCOMA_MODULE_DIR = WORKSPACE_ROOT / "glaucoma"
GLAUCOMA_REPORTS_DIR = GLAUCOMA_TEMP_DIR / "reports"
JOB_RUNTIME_DIR = Path(tempfile.gettempdir()) / "ophthalmic-imaging-web-jobs"
JOB_UPLOADS_DIR = JOB_RUNTIME_DIR / "uploads"
JOB_RESULTS_DIR = JOB_RUNTIME_DIR / "results"
CLASSIFIER_JOB_LOCK = threading.Lock()
RETINA_JOB_LOCK = threading.Lock()
GLAUCOMA_JOB_LOCK = threading.Lock()
app = FastAPI(title="Ophthalmic Imaging Pupil Classifier API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://((localhost|127\.0\.0\.1|0\.0\.0\.0)|([a-z0-9-]+)|(192\.168\.\d{1,3}\.\d{1,3})|(10\.\d{1,3}\.\d{1,3}\.\d{1,3})|(172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}))(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    _ensure_job_runtime_dirs()
    try:
        init_db()
    except Exception as exc:
        print(f"Warning: database initialization skipped: {exc}")
    cataract_error = get_classifier_service().availability_error
    if cataract_error:
        print(f"[Startup] {cataract_error}")
    preload_glaucoma_model()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str | None = None


class PatientCreate(BaseModel):
    mri_number: str
    patient_name: str | None = None


class JobQueuedResponse(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result_path: str | None = None
    error: str | None = None


class BatchReportItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    result: dict[str, Any] | None = None
    analyzed_at: str | None = None
    case_id: str | None = None
    filename: str | None = None
    patient_name: str | None = None
    mri_number: str | None = None
    eye_side: str | None = None
    detected: bool = False
    grade: str | None = None
    confidence: float | None = None
    confidence_pct: str | None = None
    probabilities: dict[str, float] | None = None
    needs_review: bool = False
    review_reason: str | None = None
    attention: dict[str, float] | None = None
    bbox: list[int] | None = None
    quality_score: int | None = None
    crop_image_base64: str | None = None
    red_glow_crop_base64: str | None = None
    slit_lamp_crop_base64: str | None = None
    anterior_segment_base64: str | None = None
    red_glow_base64: str | None = None
    slit_lamp_base64: str | None = None
    gradcam_heatmap_base64: str | None = None
    red_glow_gradcam_base64: str | None = None
    slit_lamp_gradcam_base64: str | None = None
    recommendation: str | None = None
    severity: str | None = None
    model_info: str | None = None
    report_texts: dict[str, str] | None = None
    gradcam_region_info: dict | None = None   # must be present for MedGemma
    error: str | None = None


class RetinaSegmentationResponse(BaseModel):
    module: str
    case_id: str
    patient_name: str
    mri_number: str
    eye_side: str
    filename: str
    threshold: float
    result_summary: str
    ground_truth_available: bool = False
    dice: float | None = None
    iou: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    original_image_base64: str
    mask_base64: str
    overlay_image_base64: str
    mask_filename: str
    db_ids: dict[str, Any] | None = None
    db_error: str | None = None


class GlaucomaPredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool
    review_reason: str | None = None
    timestamp: str
    module: str
    case_id: str
    patient_name: str
    mri_number: str
    eye_side: str
    db_ids: dict[str, Any] | None = None
    db_error: str | None = None


class GlaucomaReportRequest(BaseModel):
    patient_name: str
    mri_number: str
    uid: str
    eye_side: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool
    review_reason: str | None = None


class OverviewStatsResponse(BaseModel):
    total_patients: int
    this_week: int
    needs_review: int
    reports_generated: int
    total_cases: int = 0
    today_cases: list[dict[str, Any]] = []
    latest_reports: list[dict[str, Any]] = []


class CaseCreateRequest(BaseModel):
    patient_id: int
    module_types: list[str]
    priority: str = "routine"
    notes: str | None = None


class CaseUpdateRequest(BaseModel):
    status: str | None = None
    assigned_reviewer: int | None = None
    priority: str | None = None
    notes: str | None = None


class AssignReviewerRequest(BaseModel):
    reviewer_id: int


class ReviewApproveRequest(BaseModel):
    notes: str | None = None


class ReviewRejectRequest(BaseModel):
    reason: str


class ReviewOverrideRequest(BaseModel):
    clinician_grade: str
    clinician_notes: str | None = None


class ReportRejectRequest(BaseModel):
    reason: str


class CalibrationFitRequest(BaseModel):
    logits: list[list[float]]
    labels: list[int]


class ThresholdUpdateRequest(BaseModel):
    threshold_key: str
    threshold_value: float
    priority_level: str
    reason_template: str


# ---------------------------------------------------------------------------
# Dependency: singleton classifier
# ---------------------------------------------------------------------------

@lru_cache
def get_classifier_service() -> ClassifierService:
    return ClassifierService()


def _is_admin_user(current_user: User) -> bool:
    return current_user.role == "admin"


def _database_true(column):
    """Return a portable SQL true comparison (including SQL Server BIT)."""
    return column == true()


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not _is_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user


def _user_can_access_case(case: Case, current_user: User) -> bool:
    """A case is accessible to admins, its creator, or its assigned reviewer."""
    if _is_admin_user(current_user):
        return True
    return current_user.id in {case.created_by, case.assigned_reviewer}


def _authorize_case_or_403(case: Case, current_user: User) -> None:
    if not _user_can_access_case(case, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to access this case.")


def _authorize_patient_case_association(
    db: Session,
    current_user: User,
    *,
    patient_id: int | None = None,
    mri_number: str | None = None,
    case_id: str | None = None,
) -> None:
    """Authorize user-supplied identifiers before attaching new clinical data."""
    patients: list[Patient] = []
    if patient_id is not None:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if patient is not None:
            patients.append(patient)

    normalized_mri = (mri_number or "").strip()
    if normalized_mri:
        patient = db.query(Patient).filter(Patient.mri_number == normalized_mri).first()
        if patient is not None and all(row.id != patient.id for row in patients):
            patients.append(patient)

    if not _is_admin_user(current_user):
        for patient in patients:
            if patient.created_by != current_user.id:
                raise HTTPException(status_code=403, detail="Not authorized to use this patient record.")

    normalized_case_id = (case_id or "").strip()
    if normalized_case_id:
        case = db.query(Case).filter(Case.id == normalized_case_id).first()
        if case is not None:
            _authorize_case_or_403(case, current_user)


def _user_can_access_job(job: InferenceJob, current_user: User) -> bool:
    if _is_admin_user(current_user):
        return True
    # Fail closed for legacy rows with no recorded owner.
    return job.created_by is not None and job.created_by == current_user.id


def _request_metadata(request: Request) -> tuple[str | None, str | None]:
    return request.client.host if request.client else None, request.headers.get("x-session-id")


def _module_labels(modules: list[str] | None) -> list[str]:
    return [normalize_module_name(module) for module in (modules or [])]


def _safe_job_payload(job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    try:
        return _read_job_result(job_id)
    except Exception:
        return None


def _serialize_case(case: Case) -> dict[str, Any]:
    patient = case.patient
    flag_count = sum(1 for row in case.module_results if row.flagged_for_review)
    return {
        "id": case.id,
        "patient_id": case.patient_id,
        "patient_name": patient.patient_name if patient else None,
        "mri_number": patient.mri_number if patient else None,
        "status": case.status,
        "module_types": case.module_types or [],
        "assigned_reviewer": case.assigned_reviewer,
        "priority": case.priority,
        "notes": case.notes,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "report_status": case.report_status,
        "flag_count": flag_count,
    }


def _serialize_module_result(module_result: CaseModuleResult) -> dict[str, Any]:
    return {
        "id": module_result.id,
        "module": module_result.module,
        "job_id": module_result.job_id,
        "job_result_path": f"/jobs/{module_result.job_id}/result",
        "job_result": _safe_job_payload(module_result.job_id),
        "model_prediction": module_result.model_prediction,
        "model_confidence": module_result.model_confidence,
        "model_grade": module_result.model_grade,
        "review_status": module_result.review_status,
        "reviewer_id": module_result.reviewer_id,
        "reviewed_at": module_result.reviewed_at.isoformat() if module_result.reviewed_at else None,
        "clinician_grade": module_result.clinician_grade,
        "clinician_notes": module_result.clinician_notes,
        "flagged_for_review": module_result.flagged_for_review,
        "flag_reason": module_result.flag_reason,
    }


def _audit_payload(entry: AuditLog, db: Session, slim: bool = False) -> dict[str, Any]:
    actor_name = db.query(User.username).filter(User.id == entry.actor_id).scalar()
    payload = {
        "id": entry.id,
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "actor_id": entry.actor_id,
        "actor_name": actor_name,
        "action": entry.action,
        "case_id": entry.case_id,
        "module_result_id": entry.module_result_id,
    }
    if not slim:
        payload["before_state"] = entry.before_state
        payload["after_state"] = entry.after_state
    return payload


def _assign_patient_owner(db: Session, patient_id: int | None, owner_id: int) -> None:
    if patient_id is None:
        return

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is not None and patient.created_by is None:
        patient.created_by = owner_id
        db.commit()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/workspace-status")
def workspace_status(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    cataract_error = get_classifier_service().availability_error
    cataract_status = {
        "available": cataract_error is None,
        "status": "Ready" if cataract_error is None else "Checkpoint required",
    }
    glaucoma_status = get_glaucoma_module_status()
    if not _is_admin_user(current_user):
        glaucoma_status = {
            "available": bool(glaucoma_status.get("available")),
            "status": glaucoma_status.get("status") or "Unavailable",
        }
    return {
        "cataract": cataract_status,
        "glaucoma": glaucoma_status,
    }


@app.get("/stats/overview", response_model=OverviewStatsResponse)
def stats_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OverviewStatsResponse:
    now = datetime.utcnow()
    week_start = (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    patient_count_query = db.query(func.count(Patient.id))
    visit_count_query = db.query(func.count(Visit.id)).join(Patient, Visit.patient_id == Patient.id)
    weekly_query = visit_count_query.filter(Visit.visit_date >= week_start)
    review_query = (
        db.query(func.count(Classification.id))
        .join(Visit, Classification.visit_id == Visit.id)
        .join(Patient, Visit.patient_id == Patient.id)
        .filter(_database_true(Classification.needs_review))
    )
    report_count_query = visit_count_query.filter(_database_true(Visit.report_generated))
    today_query = (
        db.query(Visit, Patient, Classification)
        .join(Patient, Visit.patient_id == Patient.id)
        .outerjoin(Classification, Classification.visit_id == Visit.id)
        .filter(Visit.visit_date >= day_start)
    )
    latest_report_query = (
        db.query(Visit, Patient, Classification)
        .join(Patient, Visit.patient_id == Patient.id)
        .outerjoin(Classification, Classification.visit_id == Visit.id)
        .filter(_database_true(Visit.report_generated))
    )

    if not _is_admin_user(current_user):
        owner_filter = Patient.created_by == current_user.id
        patient_count_query = patient_count_query.filter(owner_filter)
        visit_count_query = visit_count_query.filter(owner_filter)
        weekly_query = weekly_query.filter(owner_filter)
        review_query = review_query.filter(owner_filter)
        report_count_query = report_count_query.filter(owner_filter)
        today_query = today_query.filter(owner_filter)
        latest_report_query = latest_report_query.filter(owner_filter)

    total_patients = patient_count_query.scalar() or 0
    total_cases = visit_count_query.scalar() or 0
    this_week = weekly_query.scalar() or 0
    needs_review = review_query.scalar() or 0
    reports_generated = report_count_query.scalar() or 0
    today_rows = today_query.order_by(Visit.visit_date.desc()).limit(5).all()
    latest_report_rows = (
        latest_report_query
        .order_by(Visit.report_timestamp.desc(), Visit.visit_date.desc())
        .limit(4)
        .all()
    )

    return OverviewStatsResponse(
        total_patients=int(total_patients),
        this_week=int(this_week),
        needs_review=int(needs_review),
        reports_generated=int(reports_generated),
        total_cases=int(total_cases),
        today_cases=[_serialize_visit_stat_row(row) for row in today_rows],
        latest_reports=[_serialize_report_stat_row(row) for row in latest_report_rows],
    )


@app.get("/stats/module-activity")
def stats_module_activity(
    period: str = "6months",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    period_map = {
        "1month": 1,
        "3months": 3,
        "6months": 6,
        "1year": 12,
    }
    months = period_map.get(period, 6)
    now = datetime.utcnow()

    month_slots: list[tuple[int, int, str]] = []
    year = now.year
    month = now.month
    for offset in range(months - 1, -1, -1):
        calc_month = month - offset
        calc_year = year
        while calc_month <= 0:
            calc_month += 12
            calc_year -= 1
        month_slots.append((calc_year, calc_month, datetime(calc_year, calc_month, 1).strftime("%b")))

    start_year, start_month, _ = month_slots[0]
    period_start = datetime(start_year, start_month, 1)

    rows = (
        db.query(Visit.module, Visit.visit_date)
        .filter(Visit.visit_date >= period_start)
        .all()
    )

    aggregated: dict[tuple[int, int], dict[str, Any]] = {
        (slot_year, slot_month): {
            "month": slot_label,
            "glaucoma": 0,
            "cataract": 0,
            "retina": 0,
        }
        for slot_year, slot_month, slot_label in month_slots
    }

    for module, visit_date in rows:
        if visit_date is None:
            continue
        key = (visit_date.year, visit_date.month)
        if key not in aggregated:
            continue
        normalized_module = (module or "cataract").lower()
        if normalized_module == "glaucoma":
            aggregated[key]["glaucoma"] += 1
        elif normalized_module == "retina_segmentation":
            aggregated[key]["retina"] += 1
        else:
            aggregated[key]["cataract"] += 1

    return [aggregated[(slot_year, slot_month)] for slot_year, slot_month, _ in month_slots]


@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(
        User.username == form_data.username,
        User.is_active == True
    ).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }
    access_token = create_access_token(data=token_data)
    return {"access_token": access_token, "token_type": "bearer", "role": user.role, "username": user.username}


@app.post("/auth/login", response_model=LoginResponse)
def login_legacy(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    user = db.query(User).filter(
        User.username == payload.username,
        User.is_active == True
    ).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    token = create_access_token(
        {"sub": str(user.id), "username": user.username, "role": user.role}
    )
    return LoginResponse(access_token=token, username=user.username, role=user.role)


@app.post("/patients")
async def create_patient(
    patient_data: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_patient = Patient(
        **patient_data.dict(),
        created_by=current_user.id
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    
    job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not _user_can_access_job(job, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to access this job.")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        result_path=job.result_path,
        error=job.error,
    )


@app.get("/jobs/{job_id}/result")
def get_job_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
   
    job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not _user_can_access_job(job, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to access this job.")
    if job.status == "error":
        raise HTTPException(status_code=409, detail=job.error or "Job failed.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="Job is not finished yet.")
    return _read_job_result(job_id)


@app.post("/classify")
async def classify_image(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    types: list[str] = Form(...),
    case_id: str = Form("UNKNOWN"),
    patient_name: str = Form(""),
    mri_number: str = Form(""),
    eye_side: str = Form("UNKNOWN"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobQueuedResponse:
    """
    Classify a set of eye images using the V7 multimodal model.

    Expects one file per modality tagged via the matching `types` array.
    Anterior segment is mandatory; red glow and slit lamp are optional.
    """
    _authorize_patient_case_association(db, current_user, mri_number=mri_number, case_id=case_id)
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(files) != len(types):
        raise HTTPException(status_code=400, detail="Number of files must match number of types.")

    assigned: dict[str, tuple[str, bytes]] = {}
    duplicates: list[str] = []
    normalized_eye_side = _normalize_eye_side(eye_side)

    for file, raw_type in zip(files, types):
        filename, image_bytes = await _validate_and_read_upload(file)

        norm = raw_type.strip().lower()
        if norm in {"anterior_segment", "red_glow", "slit_lamp"}:
            if norm in assigned:
                duplicates.append(norm)
            else:
                assigned[norm] = (filename, image_bytes)

    if duplicates:
        labels = ", ".join(sorted({n.replace("_", " ") for n in duplicates}))
        raise HTTPException(status_code=400, detail=f"Duplicate modality assignments: {labels}.")

    if "anterior_segment" not in assigned:
        raise HTTPException(
            status_code=400,
            detail="One image must be assigned as Anterior Segment before classification.",
        )

    job_id = str(uuid.uuid4())
    saved_modalities: dict[str, dict[str, str]] = {}
    for modality, (filename, image_bytes) in assigned.items():
        saved_path = _save_job_upload(job_id, filename, image_bytes, modality)
        saved_modalities[modality] = {
            "filename": filename,
            "path": str(saved_path),
        }

    db.add(InferenceJob(id=job_id, status="pending", created_by=current_user.id))
    db.commit()
    background_tasks.add_task(
        run_cataract_classification_job,
        job_id,
        saved_modalities,
        case_id or "UNKNOWN",
        patient_name,
        mri_number,
        normalized_eye_side,
        current_user.username,
        current_user.id,
    )
    return JobQueuedResponse(job_id=job_id)


@app.post("/detect-image-types")
def detect_image_types(
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
) -> list[dict[str, str]]:
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results: list[dict[str, str]] = []
    for file in files:
        filename = file.filename or "uploaded"
        try:
            if _ext(filename) not in ALLOWED_EXTENSIONS:
                results.append(_type_fallback(filename, "Unsupported file type."))
                continue
            image_bytes = file.file.read()
            if not image_bytes:
                results.append(_type_fallback(filename, "Empty file."))
                continue
            results.append(classify_image_type(image_bytes, filename))
        except Exception as exc:
            results.append(_type_fallback(filename, f"Detection failed: {exc}"))

    return results


@app.post("/retina/probability")
async def retina_probability(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    case_id: str = Form("UNKNOWN"),
    patient_name: str = Form(""),
    mri_number: str = Form(""),
    eye_side: str = Form("UNKNOWN"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobQueuedResponse:
    _authorize_patient_case_association(db, current_user, mri_number=mri_number, case_id=case_id)
    filename, image_bytes = await _validate_and_read_upload(file)

    normalized_eye_side = _normalize_eye_side(eye_side)
    job_id = str(uuid.uuid4())
    saved_path = _save_job_upload(job_id, filename, image_bytes, "retina_probability")
    db.add(InferenceJob(id=job_id, status="pending", created_by=current_user.id))
    db.commit()
    background_tasks.add_task(
        run_retina_probability_job,
        job_id,
        str(saved_path),
        case_id or "UNKNOWN",
        patient_name,
        mri_number,
        normalized_eye_side,
        current_user.username,
        current_user.id,
    )
    return JobQueuedResponse(job_id=job_id)


@app.post("/retina-segmentation", response_model=JobQueuedResponse)
async def retina_segmentation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    case_id: str = Form("UNKNOWN"),
    patient_name: str = Form(""),
    mri_number: str = Form(""),
    eye_side: str = Form("UNKNOWN"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobQueuedResponse:
    _authorize_patient_case_association(db, current_user, mri_number=mri_number, case_id=case_id)
    if threshold < 0.1 or threshold > 0.9:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.1 and 0.9.")

    filename, image_bytes = await _validate_and_read_upload(file)
    normalized_eye_side = _normalize_eye_side(eye_side)
    job_id = str(uuid.uuid4())
    saved_path = _save_job_upload(job_id, filename, image_bytes, "retina_segmentation")
    db.add(InferenceJob(id=job_id, status="pending", created_by=current_user.id))
    db.commit()
    background_tasks.add_task(
        run_retina_segmentation_job,
        job_id,
        str(saved_path),
        filename,
        threshold,
        case_id or "UNKNOWN",
        patient_name,
        mri_number,
        normalized_eye_side,
        current_user.username,
        current_user.id,
    )
    return JobQueuedResponse(job_id=job_id)


@app.post("/glaucoma/predict", response_model=JobQueuedResponse)
async def glaucoma_predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    case_id: str = Form("UNKNOWN"),
    patient_name: str = Form(""),
    mri_number: str = Form(""),
    eye_side: str = Form("UNKNOWN"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobQueuedResponse:
    _authorize_patient_case_association(db, current_user, mri_number=mri_number, case_id=case_id)
    filename, image_bytes = await _validate_and_read_upload(file)
    normalized_eye_side = _normalize_eye_side(eye_side)
    job_id = str(uuid.uuid4())
    saved_path = _save_job_upload(job_id, filename, image_bytes, "glaucoma_predict")
    db.add(InferenceJob(id=job_id, status="pending", created_by=current_user.id))
    db.commit()
    background_tasks.add_task(
        run_glaucoma_prediction_job,
        job_id,
        str(saved_path),
        case_id or "UNKNOWN",
        patient_name,
        mri_number,
        normalized_eye_side,
        current_user.username,
        current_user.id,
    )
    return JobQueuedResponse(job_id=job_id)


@app.post("/glaucoma/report")
async def glaucoma_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: str = Form(default=""),
    eye_side: str = Form(default=""),
    patient_name: str = Form(default=""),
    mri_number: str = Form(default=""),
    case_id: str = Form(default=""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobQueuedResponse:
    """
    Full glaucoma analysis pipeline:
    1. EfficientNet-B4 prediction on disc-cropped image
    2. DeepVesselNet vessel mask -> biomarker extraction
    3. MedGemma grounded report from quantitative features
    Returns prediction + vessel_features + structured report + grounding_score
    """
    # patient_id is a report UID string here, not the patients.id primary key.
    _authorize_patient_case_association(db, current_user, mri_number=mri_number, case_id=case_id)
    filename, image_bytes = await _validate_and_read_upload(file)
    job_id = str(uuid.uuid4())
    saved_path = _save_job_upload(job_id, filename, image_bytes, "glaucoma_report")
    db.add(InferenceJob(id=job_id, status="pending", created_by=current_user.id))
    db.commit()
    background_tasks.add_task(
        run_glaucoma_report_job,
        job_id,
        str(saved_path),
        patient_id,
        eye_side,
        patient_name,
        mri_number,
        case_id,
        current_user.username,
        current_user.id,
    )
    return JobQueuedResponse(job_id=job_id)


@app.get("/glaucoma/report/download")
def glaucoma_report_download(
    pdf_path: str,
    admin_user: User = Depends(require_admin),
) -> FileResponse:
    # Interim fail-closed policy: report files predate persisted ownership metadata.
    del admin_user
    resolved_path = _resolve_glaucoma_report_download_path(pdf_path)
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Report PDF not found.")
    return FileResponse(
        path=str(resolved_path),
        media_type="application/pdf",
        filename=resolved_path.name,
        headers={"Content-Disposition": f'attachment; filename="{resolved_path.name}"'},
    )


@app.post("/classify-batch", response_model=JobQueuedResponse)
async def classify_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobQueuedResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    job_id = str(uuid.uuid4())
    saved_files: list[dict[str, str]] = []
    for file in files:
        filename, image_bytes = await _validate_and_read_upload(file)
        saved_path = _save_job_upload(job_id, filename, image_bytes, "batch")
        saved_files.append({"filename": filename, "path": str(saved_path)})

    db.add(InferenceJob(id=job_id, status="pending", created_by=current_user.id))
    db.commit()
    background_tasks.add_task(
        run_cataract_batch_job,
        job_id,
        saved_files,
        current_user.username,
    )
    return JobQueuedResponse(job_id=job_id)


@app.get("/report")
def download_report(
    current_user: User = Depends(get_current_user),
) -> Response:
    record = SESSION_RESULTS.get(current_user.username)
    if record is None:
        raise HTTPException(status_code=404, detail="No classification result found for this session.")

    # Safety guard for any stale raw-dict entries created by older code paths.
    if "result" not in record:
        record = {
            "result": record,
            "analyzed_at": datetime.now(),
        }

    pdf_bytes = generate_pdf_report(current_user.username, record)
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ophthalmic-imaging-report.pdf"'},
    )


@app.get("/history/{mri_number}")
def patient_history(
    mri_number: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    patient = db.query(Patient).filter(Patient.mri_number == mri_number).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if current_user.role != "admin" and patient.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this patient")

    try:
        history = get_patient_history(db, mri_number)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    if not history:
        raise HTTPException(status_code=404, detail=f"No records found for MRI number: {mri_number}")
    return {
        "mri_number": mri_number,
        "patient_name": history[0].get("patient_name"),
        "total_visits": len(history),
        "visits": history,
    }


@app.get("/longitudinal/{mri_number}/{eye_side}")
def longitudinal(
    mri_number: str,
    eye_side: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_eye_side = _normalize_eye_side(eye_side)
    patient = db.query(Patient).filter(Patient.mri_number == mri_number).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if current_user.role != "admin" and patient.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this patient")

    try:
        history = get_patient_history(db, mri_number)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    if not history:
        raise HTTPException(status_code=404, detail=f"No records found for MRI number: {mri_number}")
    filtered = [v for v in history if (v.get("eye_side") or "").upper() == normalized_eye_side]
    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"No {normalized_eye_side} records found for MRI: {mri_number}",
        )

    series = sorted(
        [
            {
                "visit_date": v["visit_date"],
                "grade": v.get("grade"),
                "grade_index": {"NS1": 1, "NS2": 2, "NS3": 3, "NS4": 4}.get(v.get("grade"), 0),
                "confidence": v.get("confidence"),
                "needs_review": v.get("needs_review", False),
            }
            for v in filtered
            if v.get("grade") in {"NS1", "NS2", "NS3", "NS4"} and (v.get("module") or "cataract") == "cataract"
        ],
        key=lambda x: x["visit_date"] or "",
    )

    return {
        "mri_number": mri_number,
        "eye_side": normalized_eye_side,
        "data_points": len(series),
        "progression": series,
    }


@app.get("/patients")
def list_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    try:
        if _is_admin_user(current_user):
            patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
        else:
            patients = (
                db.query(Patient)
                .filter(Patient.created_by == current_user.id)
                .order_by(Patient.created_at.desc())
                .all()
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    return [
        {
            "id": patient.id,
            "mri_number": patient.mri_number,
            "patient_name": patient.patient_name,
            "visit_count": len(patient.visits),
            "created_at": patient.created_at.isoformat() if patient.created_at else None,
        }
        for patient in patients
    ]


@app.post("/report-batch")
def download_batch_report(
    results: list[BatchReportItem],
    current_user: User = Depends(get_current_user),
) -> Response:
    if not results:
        raise HTTPException(status_code=400, detail="At least one result is required.")

    payload = [_coerce_batch_report_item(result) for result in results]

    # ── Restore gradcam_region_info from SESSION_RESULTS cache ───────────────
    # The frontend sends result objects back but Pydantic may strip unknown
    # fields like gradcam_region_info. We restore it from the server-side
    # session cache which always has the full original result.
    session = SESSION_RESULTS.get(current_user.username, {})
    cached_results = session.get("all_results", {})   # keyed by eye_side

    for item in payload:
        eye_side = item.get("eye_side", "")
        cached   = cached_results.get(eye_side) or cached_results.get("latest")
        if cached and not item.get("gradcam_region_info"):
            item["gradcam_region_info"] = cached.get("gradcam_region_info", {})
            print(f"[Report] Restored gradcam_region_info for {eye_side} from session cache")

    # Generate MedGemma texts on demand with GPU swap
    for item in payload:
        record = {"result": item, "analyzed_at": item.get("_analyzed_at", datetime.now())}
        _generate_medgemma_texts_for_record(record)

    pdf_bytes = generate_batch_pdf_report(current_user.username, payload)
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ophthalmic-imaging-batch-report.pdf"'},
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_job_runtime_dirs() -> None:
    JOB_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    JOB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(filename: str | None, fallback: str) -> str:
    candidate = Path(filename or fallback).name.strip()
    if not candidate:
        return fallback
    return candidate.replace("..", "_")


async def _validate_and_read_upload(file: UploadFile) -> tuple[str, bytes]:
    filename = file.filename or "uploaded"
    if _ext(filename) not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for '{filename}'. Upload JPG or PNG only.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported MIME type for '{filename}'. Upload JPEG or PNG only.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"'{filename}' is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{filename}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit.",
        )

    return filename, image_bytes


def _job_upload_dir(job_id: str) -> Path:
    return JOB_UPLOADS_DIR / job_id


def _job_result_file(job_id: str) -> Path:
    return JOB_RESULTS_DIR / f"{job_id}.json"


def _job_result_endpoint(job_id: str) -> str:
    return f"/jobs/{job_id}/result"


def _save_job_upload(job_id: str, filename: str | None, data: bytes, prefix: str) -> Path:
    _ensure_job_runtime_dirs()
    upload_dir = _job_upload_dir(job_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename, f"{prefix}.bin")
    path = upload_dir / f"{prefix}_{safe_name}"
    path.write_bytes(data)
    return path


def _cleanup_paths(paths: list[Path | None]) -> None:
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            continue
        try:
            path.unlink()
        except OSError:
            pass


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_job_result(job_id: str, payload: dict[str, Any]) -> str:
    _ensure_job_runtime_dirs()
    result_file = _job_result_file(job_id)
    result_file.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    return _job_result_endpoint(job_id)


def _read_job_result(job_id: str) -> dict[str, Any]:
    result_file = _job_result_file(job_id)
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Job result not found.")
    return json.loads(result_file.read_text(encoding="utf-8"))


def _update_job_status(
    job_id: str,
    *,
    status_value: str,
    error: str | None = None,
    result_path: str | None = None,
) -> None:
    db = SessionLocal()
    try:
        job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()
        if job is None:
            return
        job.status = status_value
        job.error = error
        job.result_path = result_path
        db.commit()
    finally:
        db.close()


def _run_background_job(job_id: str, work_fn: Any, cleanup_paths: list[Path | None]) -> None:
    _update_job_status(job_id, status_value="running", error=None, result_path=None)
    try:
        payload = work_fn()
        result_path = _write_job_result(job_id, payload)
        _update_job_status(job_id, status_value="done", error=None, result_path=result_path)
    except Exception as exc:
        _update_job_status(job_id, status_value="error", error=str(exc), result_path=None)
    finally:
        _cleanup_paths(cleanup_paths)


def _apply_probability_calibration(module: str, probabilities: dict[str, float]) -> tuple[dict[str, float], bool]:
    if not probabilities:
        return probabilities, False
    db = SessionLocal()
    try:
        scaler = TemperatureScaler(module, db)
        calibrated_values, is_calibrated = scaler.calibrate_probabilities(probabilities)
        if not is_calibrated:
            return probabilities, False
        keys = list(probabilities.keys())
        calibrated = {key: float(calibrated_values[idx]) for idx, key in enumerate(keys)}
        return calibrated, True
    finally:
        db.close()


def _apply_binary_probability_calibration(module: str, positive_probability: float) -> tuple[float, bool]:
    probabilities = {
        "negative": 1.0 - positive_probability,
        "positive": positive_probability,
    }
    calibrated, is_calibrated = _apply_probability_calibration(module, probabilities)
    return float(calibrated.get("positive", positive_probability)), is_calibrated


def _evaluate_review_decision(module: str, result: dict[str, Any]) -> tuple[bool, str | None, str]:
    db = SessionLocal()
    try:
        engine = ReviewThresholdEngine(db)
        decision = engine.evaluate(module, result)
        return decision.flagged, "; ".join(decision.reasons) if decision.reasons else None, decision.priority
    finally:
        db.close()


def _save_classification_result_for_session(
    *,
    username: str,
    eye_side: str,
    result: dict[str, Any],
    analyzed_at: datetime,
) -> None:
    SESSION_RESULTS[username] = {
        "result": result,
        "analyzed_at": analyzed_at,
    }
    session = SESSION_RESULTS[username]
    if "all_results" not in session:
        session["all_results"] = {}
    session["all_results"][eye_side] = result
    session["all_results"]["latest"] = result


def run_cataract_classification_job(
    job_id: str,
    saved_modalities: dict[str, dict[str, str]],
    case_id: str,
    patient_name: str,
    mri_number: str,
    eye_side: str,
    username: str,
    user_id: int,
) -> None:
    cleanup_paths: list[Path | None] = [_job_upload_dir(job_id)]

    def work() -> dict[str, Any]:
        anterior = saved_modalities["anterior_segment"]
        anterior_name = anterior["filename"]
        anterior_bytes = Path(anterior["path"]).read_bytes()
        red_glow = saved_modalities.get("red_glow")
        slit_lamp = saved_modalities.get("slit_lamp")
        red_glow_bytes = Path(red_glow["path"]).read_bytes() if red_glow else None
        slit_lamp_bytes = Path(slit_lamp["path"]).read_bytes() if slit_lamp else None

        classifier = get_classifier_service()
        with CLASSIFIER_JOB_LOCK:
            artifacts = classifier.classify_exam(
                anterior_segment_bytes=anterior_bytes,
                red_glow_bytes=red_glow_bytes,
                slit_lamp_bytes=slit_lamp_bytes,
                case_id=case_id or "UNKNOWN",
                filename=anterior_name,
            )

        calibrated_probabilities, is_calibrated = _apply_probability_calibration(
            "cataract",
            artifacts.result.get("probabilities") or {},
        )
        if calibrated_probabilities:
            artifacts.result["probabilities"] = calibrated_probabilities
            top_grade = max(calibrated_probabilities, key=calibrated_probabilities.get)
            artifacts.result["confidence"] = float(calibrated_probabilities[top_grade])
            artifacts.result["predicted_class"] = top_grade
            artifacts.result["grade"] = top_grade
        artifacts.result["is_calibrated"] = is_calibrated
        artifacts.result["modality_count"] = 1 + int(red_glow is not None) + int(slit_lamp is not None)
        flagged, flag_reason, priority = _evaluate_review_decision("cataract", artifacts.result)
        artifacts.result["needs_review"] = flagged
        artifacts.result["review_reason"] = flag_reason
        artifacts.result["review_priority"] = priority
        artifacts.result["patient_name"] = patient_name
        artifacts.result["mri_number"] = mri_number
        artifacts.result["eye_side"] = eye_side
        artifacts.result["case_id"] = case_id or artifacts.result.get("case_id") or "UNKNOWN"

        if mri_number.strip():
            db = SessionLocal()
            try:
                artifacts.result["db_ids"] = save_full_result(
                    db=db,
                    mri_number=mri_number.strip(),
                    patient_name=patient_name.strip() or None,
                    case_id=(case_id.strip() if case_id else None) or None,
                    eye_side=eye_side,
                    graded_by=username,
                    result=artifacts.result,
                    job_id=job_id,
                    created_by_user_id=user_id,
                    flagged_for_review=flagged,
                    flag_reason=flag_reason,
                )
                _assign_patient_owner(
                    db,
                    artifacts.result["db_ids"].get("patient_id"),
                    user_id,
                )
            except Exception as exc:
                artifacts.result["db_error"] = f"Database save skipped: {exc}"
            finally:
                db.close()

        _save_classification_result_for_session(
            username=username,
            eye_side=eye_side,
            result=artifacts.result,
            analyzed_at=artifacts.analyzed_at,
        )
        return artifacts.result

    _run_background_job(job_id, work, cleanup_paths)


def run_cataract_batch_job(
    job_id: str,
    saved_files: list[dict[str, str]],
    username: str,
) -> None:
    cleanup_paths: list[Path | None] = [_job_upload_dir(job_id)]

    def work() -> dict[str, Any]:
        prepared = [
            (item["filename"], Path(item["path"]).read_bytes())
            for item in saved_files
        ]
        classifier = get_classifier_service()
        with CLASSIFIER_JOB_LOCK:
            artifacts = classifier.classify_batch(prepared)
        payload = {"results": [artifact.result for artifact in artifacts]}
        SESSION_RESULTS[username] = {"batch_results": payload["results"]}
        return payload

    _run_background_job(job_id, work, cleanup_paths)


def run_retina_probability_job(
    job_id: str,
    saved_path: str,
    case_id: str,
    patient_name: str,
    mri_number: str,
    eye_side: str,
    username: str,
    user_id: int,
) -> None:
    cleanup_paths: list[Path | None] = [_job_upload_dir(job_id)]

    def work() -> dict[str, Any]:
        with RETINA_JOB_LOCK:
            probability_map = predict_probability_map(str(saved_path))
        probability_u8 = np.clip(probability_map * 255.0, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", probability_u8)
        if not ok:
            raise RuntimeError("Failed to encode probability map.")

        payload: dict[str, Any] = {
            "probability_map": base64.b64encode(encoded.tobytes()).decode("utf-8"),
            "width": int(probability_u8.shape[1]),
            "height": int(probability_u8.shape[0]),
        }

        if mri_number.strip():
            db = SessionLocal()
            try:
                flagged, flag_reason, priority = _evaluate_review_decision("retina", payload)
                payload["needs_review"] = flagged
                payload["review_reason"] = flag_reason
                payload["review_priority"] = priority
                payload["db_ids"] = save_retina_result(
                    db=db,
                    mri_number=mri_number.strip(),
                    patient_name=patient_name.strip() or None,
                    case_id=(case_id.strip() if case_id else None) or None,
                    eye_side=eye_side,
                    graded_by=username,
                    result_summary="Segmentation complete",
                    result_payload=payload,
                    job_id=job_id,
                    created_by_user_id=user_id,
                    flagged_for_review=flagged,
                    flag_reason=flag_reason,
                )
                _assign_patient_owner(
                    db,
                    payload["db_ids"].get("patient_id"),
                    user_id,
                )
            except Exception as exc:
                payload["db_error"] = f"Database save skipped: {exc}"
            finally:
                db.close()

        return payload

    _run_background_job(job_id, work, cleanup_paths)


def run_retina_segmentation_job(
    job_id: str,
    saved_path: str,
    filename: str,
    threshold: float,
    case_id: str,
    patient_name: str,
    mri_number: str,
    eye_side: str,
    username: str,
    user_id: int,
) -> None:
    cleanup_paths: list[Path | None] = [_job_upload_dir(job_id)]

    def work() -> dict[str, Any]:
        image_bytes = Path(saved_path).read_bytes()
        with RETINA_JOB_LOCK:
            artifacts = segment_retina_image(
                image_bytes=image_bytes,
                filename=filename or "fundus.png",
                threshold=threshold,
            )

        temp_input_path = artifacts.pop("temp_input_path", None)
        temp_mask_path = artifacts.pop("temp_mask_path", None)
        cleanup_paths.extend(
            [
                Path(temp_input_path) if temp_input_path else None,
                Path(temp_mask_path) if temp_mask_path else None,
            ]
        )

        result_summary = f"Segmentation complete (threshold {threshold:.2f})"
        payload: dict[str, Any] = {
            "module": "retina_segmentation",
            "case_id": case_id or "UNKNOWN",
            "patient_name": patient_name,
            "mri_number": mri_number,
            "eye_side": eye_side,
            "filename": filename or "fundus.png",
            "threshold": threshold,
            "result_summary": result_summary,
            "ground_truth_available": False,
            "dice": None,
            "iou": None,
            "sensitivity": None,
            "specificity": None,
            **artifacts,
        }
        payload["is_calibrated"] = False
        flagged, flag_reason, priority = _evaluate_review_decision("retina", payload)
        payload["needs_review"] = flagged
        payload["review_reason"] = flag_reason
        payload["review_priority"] = priority

        if mri_number.strip():
            db = SessionLocal()
            try:
                payload["db_ids"] = save_retina_result(
                    db=db,
                    mri_number=mri_number.strip(),
                    patient_name=patient_name.strip() or None,
                    case_id=(case_id.strip() if case_id else None) or None,
                    eye_side=eye_side,
                    graded_by=username,
                    result_summary=result_summary,
                    result_payload=payload,
                    job_id=job_id,
                    created_by_user_id=user_id,
                    flagged_for_review=flagged,
                    flag_reason=flag_reason,
                )
                _assign_patient_owner(
                    db,
                    payload["db_ids"].get("patient_id"),
                    user_id,
                )
            except Exception as exc:
                payload["db_error"] = f"Database save skipped: {exc}"
            finally:
                db.close()

        return payload

    _run_background_job(job_id, work, cleanup_paths)


def run_glaucoma_prediction_job(
    job_id: str,
    saved_path: str,
    case_id: str,
    patient_name: str,
    mri_number: str,
    eye_side: str,
    username: str,
    user_id: int,
) -> None:
    cleanup_paths: list[Path | None] = [_job_upload_dir(job_id)]

    def work() -> dict[str, Any]:
        with GLAUCOMA_JOB_LOCK:
            raw_result = predict_glaucoma_file(str(saved_path))

        probabilities = raw_result.get("probabilities", {})
        calibrated_probabilities, is_calibrated = _apply_probability_calibration("glaucoma", probabilities)
        if calibrated_probabilities:
            probabilities = calibrated_probabilities
        predicted_class = raw_result.get("predicted_class", "Unknown")
        confidence = normalize_confidence(probabilities.get(predicted_class, raw_result.get("confidence") or 0.0))
        payload: dict[str, Any] = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "timestamp": datetime.now().replace(microsecond=0).isoformat(),
            "module": "glaucoma",
            "case_id": case_id or "UNKNOWN",
            "patient_name": patient_name,
            "mri_number": mri_number,
            "eye_side": eye_side,
            "is_calibrated": is_calibrated,
            "gradcam_b64": raw_result.get("gradcam_b64"),
            "overlay_b64": raw_result.get("overlay_b64"),
        }
        flagged, flag_reason, priority = _evaluate_review_decision("glaucoma", payload)
        payload["needs_review"] = flagged
        payload["review_reason"] = flag_reason
        payload["review_priority"] = priority

        if mri_number.strip():
            db = SessionLocal()
            try:
                payload["db_ids"] = save_full_result(
                    db=db,
                    mri_number=mri_number.strip(),
                    patient_name=patient_name.strip() or None,
                    case_id=(case_id.strip() if case_id else None) or None,
                    eye_side=eye_side,
                    graded_by=username,
                    result=payload,
                    module="glaucoma",
                    job_id=job_id,
                    created_by_user_id=user_id,
                    flagged_for_review=flagged,
                    flag_reason=flag_reason,
                )
                _assign_patient_owner(
                    db,
                    payload["db_ids"].get("patient_id"),
                    user_id,
                )
            except Exception as exc:
                payload["db_error"] = f"Database save skipped: {exc}"
            finally:
                db.close()

        return payload

    _run_background_job(job_id, work, cleanup_paths)


def run_glaucoma_report_job(
    job_id: str,
    saved_path: str,
    patient_id: str,
    eye_side: str,
    patient_name: str = "",
    mri_number: str = "",
    case_id: str = "",
    username: str = "",
    user_id: int | None = None,
) -> None:
    cleanup_paths: list[Path | None] = [_job_upload_dir(job_id)]

    def work() -> dict[str, Any]:
        normalized_eye_side = _normalize_eye_side(eye_side)

        with GLAUCOMA_JOB_LOCK:
            result = generate_report_from_file(
                image_path=saved_path,
                patient_info={
                    "patient_id": patient_id,
                    "eye_side": normalized_eye_side,
                },
                backend="local",
            )

        try:
            output_path = _build_glaucoma_report_output_path(patient_id or "UNKNOWN", normalized_eye_side)
            report_payload = result.get("report", {}) if isinstance(result, dict) else {}
            report_text = report_payload.get("raw_text") or report_payload.get("findings") or ""
            if report_text:
                saved_pdf_path = generate_glaucoma_pdf(
                    report_text=report_text,
                    patient_name="",
                    mri_number=patient_id or "",
                    uid=patient_id or "UNKNOWN",
                    eye_side=normalized_eye_side,
                    predicted_class=result.get("prediction", "Unknown"),
                    confidence=float(result.get("confidence", 0.0) or 0.0),
                    probabilities=result.get("probabilities", {}),
                    needs_review=bool(result.get("needs_review", False)),
                    output_path=str(output_path),
                )
                result["pdf_path"] = _relative_workspace_path(Path(saved_pdf_path))
        except Exception as exc:
            print(f"[GlaucomaPDF] Failed to generate PDF: {exc}")

        # Save to database if mri_number provided
        if mri_number.strip():
            db = SessionLocal()
            try:
                predicted_class = result.get("prediction") or result.get("predicted_class") or "Unknown"
                confidence = float(result.get("confidence") or 0.0)
                probabilities = result.get("probabilities") or {}
                flagged, flag_reason, priority = _evaluate_review_decision("glaucoma", {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                })
                result["needs_review"] = flagged
                result["review_reason"] = flag_reason
                result["db_ids"] = save_full_result(
                    db=db,
                    mri_number=mri_number.strip(),
                    patient_name=patient_name.strip() or None,
                    case_id=(case_id.strip() if case_id else None) or None,
                    eye_side=normalized_eye_side,
                    graded_by=username,
                    result={
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "needs_review": flagged,
                        "review_reason": flag_reason,
                        "review_priority": priority,
                    },
                    module="glaucoma",
                    job_id=job_id,
                    created_by_user_id=user_id,
                    flagged_for_review=flagged,
                    flag_reason=flag_reason,
                )
                _assign_patient_owner(db, result["db_ids"].get("patient_id"), user_id or 1)
            except Exception as exc:
                result["db_error"] = f"Database save skipped: {exc}"
            finally:
                db.close()

        return result

    _run_background_job(job_id, work, cleanup_paths)


def _ext(filename: str | None) -> str:
    if not filename:
        return ""
    parts = filename.lower().rsplit(".", 1)
    return f".{parts[1]}" if len(parts) == 2 else ""


def _type_fallback(filename: str, reasoning: str) -> dict[str, str]:
    return {
        "filename": filename,
        "detected_type": "unknown",
        "confidence": "low",
        "reasoning": reasoning,
    }


def _normalize_eye_side(eye_side: str | None) -> str:
    normalized = (eye_side or "UNKNOWN").strip().upper()
    if normalized not in {"OD", "OS", "UNKNOWN"}:
        return "UNKNOWN"
    return normalized


def _serialize_visit_stat_row(row: tuple[Visit, Patient, Classification | None]) -> dict[str, Any]:
    visit, patient, classification = row
    module = visit.module or "cataract"
    result_label = visit.result_summary or "-"
    confidence = None
    needs_review = False

    if classification is not None:
        needs_review = bool(classification.needs_review)
        confidence = classification.confidence
        if module == "glaucoma":
            result_label = classification.predicted_class or classification.grade or result_label
        elif module == "cataract":
            result_label = classification.grade or result_label

    return {
        "visit_id": visit.id,
        "patient_name": patient.patient_name or patient.mri_number,
        "mri_number": patient.mri_number,
        "module": module,
        "eye_side": visit.eye_side,
        "result": result_label,
        "confidence": confidence,
        "needs_review": needs_review,
        "timestamp": visit.visit_date.isoformat() if visit.visit_date else None,
    }


def _serialize_report_stat_row(row: tuple[Visit, Patient, Classification | None]) -> dict[str, Any]:
    visit, patient, classification = row
    module = visit.module or "cataract"
    result_label = visit.result_summary or "-"
    confidence = None
    needs_review = False

    if classification is not None:
        confidence = classification.confidence
        needs_review = bool(classification.needs_review)
        if module == "glaucoma":
            result_label = classification.predicted_class or classification.grade or result_label
        elif module == "cataract":
            result_label = classification.grade or result_label

    return {
        "visit_id": visit.id,
        "patient_name": patient.patient_name or patient.mri_number,
        "mri_number": patient.mri_number,
        "module": module,
        "eye_side": visit.eye_side,
        "result": result_label,
        "confidence": confidence,
        "needs_review": needs_review,
        "timestamp": (visit.report_timestamp or visit.visit_date).isoformat() if (visit.report_timestamp or visit.visit_date) else None,
        "pdf_path": visit.pdf_path,
        "report_generated": bool(visit.report_generated),
    }


def _build_glaucoma_report_output_path(uid: str, eye_side: str) -> Path:
    GLAUCOMA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_uid = (uid or "UNKNOWN").strip().replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return GLAUCOMA_REPORTS_DIR / f"{safe_uid}_{eye_side}_{timestamp}.pdf"


def _relative_workspace_path(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve())).replace("\\", "/")


def _resolve_glaucoma_report_download_path(pdf_path: str) -> Path:
    allowed_root = GLAUCOMA_REPORTS_DIR.resolve()
    candidate = Path(pdf_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (WORKSPACE_ROOT / candidate).resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Requested PDF path is outside the Glaucoma reports directory.") from exc
    return resolved


@app.post("/cases")
def create_case_record(
    payload: CaseCreateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    _authorize_patient_case_association(db, current_user, patient_id=payload.patient_id)
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found.")
    ip_address, session_id = _request_metadata(request)
    case = Case(
        patient_id=payload.patient_id,
        created_by=current_user.id,
        status="pending_review",
        module_types=_module_labels(payload.module_types),
        priority=payload.priority,
        notes=payload.notes,
    )
    try:
        db.add(case)
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="case.created",
            case_id=case.id,
            after_state=case,
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"case_id": case.id}


@app.get("/cases")
def list_cases(
    status: str | None = None,
    module: str | None = None,
    reviewer_id: int | None = None,
    priority: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    
    query = db.query(Case).join(Patient, Patient.id == Case.patient_id)
    if not _is_admin_user(current_user):
        query = query.filter(
            (Case.created_by == current_user.id)
            | (Case.assigned_reviewer == current_user.id)
        )
    if status:
        query = query.filter(Case.status == status)
    if reviewer_id:
        query = query.filter(Case.assigned_reviewer == reviewer_id)
    if priority:
        query = query.filter(Case.priority == priority)
    if date_from:
        query = query.filter(Case.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(Case.created_at <= datetime.fromisoformat(date_to))
    rows = (
        query.order_by(Case.created_at.desc())
        .all()
    )
    if module:
        module_name = normalize_module_name(module)
        rows = [row for row in rows if module_name in (row.module_types or [])]
    total = len(rows)
    rows = rows[max(page - 1, 0) * page_size : max(page - 1, 0) * page_size + page_size]
    return {"items": [_serialize_case(row) for row in rows], "total": total, "page": page, "page_size": page_size}


@app.get("/cases/{case_id}")
def get_case_detail(
    case_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    
    case = db.query(Case).filter(Case.id == case_id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    _authorize_case_or_403(case, current_user)
    audit_rows = (
        db.query(AuditLog)
        .filter(AuditLog.case_id == case_id)
        .order_by(AuditLog.timestamp.asc())
        .all()
    )
    patient = case.patient
    return {
        "case": _serialize_case(case),
        "patient_summary": {
            "id": patient.id if patient else None,
            "patient_name": patient.patient_name if patient else None,
            "mri_number": patient.mri_number if patient else None,
        },
        "module_results": [_serialize_module_result(row) for row in case.module_results],
        "audit_trail": [_audit_payload(row, db) for row in audit_rows],
    }


@app.patch("/cases/{case_id}")
def update_case_record(
    case_id: str,
    payload: CaseUpdateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    case = db.query(Case).filter(Case.id == case_id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    _authorize_case_or_403(case, current_user)
    before_state = _serialize_case(case)
    if payload.status is not None:
        case.status = payload.status
    if payload.assigned_reviewer is not None:
        case.assigned_reviewer = payload.assigned_reviewer
    if payload.priority is not None:
        case.priority = payload.priority
    if payload.notes is not None:
        case.notes = payload.notes
    ip_address, session_id = _request_metadata(request)
    try:
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="case.status_changed" if payload.status is not None else "case.assigned",
            case_id=case.id,
            before_state=before_state,
            after_state=case,
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return _serialize_case(case)


@app.post("/cases/{case_id}/assign")
def assign_case_reviewer(
    case_id: str,
    payload: AssignReviewerRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    case = db.query(Case).filter(Case.id == case_id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    _authorize_case_or_403(case, current_user)
    before_state = _serialize_case(case)
    case.assigned_reviewer = payload.reviewer_id
    if case.status == "pending_review":
        case.status = "under_review"
    ip_address, session_id = _request_metadata(request)
    try:
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="case.assigned",
            case_id=case.id,
            before_state=before_state,
            after_state=case,
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return _serialize_case(case)


def _get_case_module_result_or_404(db: Session, case_id: str, module: str) -> CaseModuleResult:
    row = (
        db.query(CaseModuleResult)
        .filter(CaseModuleResult.case_id == case_id, CaseModuleResult.module == normalize_module_name(module))
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Module result not found.")
    return row


def _get_authorized_module_result(db: Session, case_id: str, module: str, current_user: User) -> CaseModuleResult:
    case = db.query(Case).filter(Case.id == case_id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    _authorize_case_or_403(case, current_user)
    return _get_case_module_result_or_404(db, case_id, module)


@app.post("/cases/{case_id}/results/{module}/approve")
def approve_case_result(
    case_id: str,
    module: str,
    payload: ReviewApproveRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = _get_authorized_module_result(db, case_id, module, current_user)
    before_state = _serialize_module_result(row)
    row.review_status = "approved"
    row.reviewer_id = current_user.id
    row.reviewed_at = datetime.utcnow()
    row.clinician_notes = payload.notes
    ip_address, session_id = _request_metadata(request)
    try:
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="module_result.approved",
            case_id=case_id,
            module_result_id=row.id,
            before_state=before_state,
            after_state=row,
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return _serialize_module_result(row)


@app.post("/cases/{case_id}/results/{module}/reject")
def reject_case_result(
    case_id: str,
    module: str,
    payload: ReviewRejectRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = _get_authorized_module_result(db, case_id, module, current_user)
    before_state = _serialize_module_result(row)
    row.review_status = "rejected"
    row.reviewer_id = current_user.id
    row.reviewed_at = datetime.utcnow()
    row.clinician_notes = payload.reason
    ip_address, session_id = _request_metadata(request)
    try:
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="module_result.rejected",
            case_id=case_id,
            module_result_id=row.id,
            before_state=before_state,
            after_state=row,
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return _serialize_module_result(row)


@app.post("/cases/{case_id}/results/{module}/override")
def override_case_result(
    case_id: str,
    module: str,
    payload: ReviewOverrideRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    row = _get_authorized_module_result(db, case_id, module, current_user)
    before_state = _serialize_module_result(row)
    row.review_status = "overridden"
    row.reviewer_id = current_user.id
    row.reviewed_at = datetime.utcnow()
    row.clinician_grade = payload.clinician_grade
    row.clinician_notes = payload.clinician_notes
    ip_address, session_id = _request_metadata(request)
    try:
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="module_result.overridden",
            case_id=case_id,
            module_result_id=row.id,
            before_state=before_state,
            after_state=row,
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return _serialize_module_result(row)


def _build_case_review_pdf(case: Case, db: Session) -> Path:
    import base64
    import io
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas

    # Resolve the actual reviewing clinician instead of hardcoding "admin".
    clinician_id = case.report_approved_by
    if clinician_id is None:
        for module_result in case.module_results:
            if module_result.reviewer_id is not None:
                clinician_id = module_result.reviewer_id
                break
    clinician_name = "-"
    if clinician_id is not None:
        clinician_name = (
            db.query(User.username).filter(User.id == clinician_id).scalar() or f"User {clinician_id}"
        )

    output_dir = JOB_RESULTS_DIR / "case_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.id}.pdf"

    W, H = A4
    c = Canvas(str(output_path), pagesize=A4)
    c.setTitle(f"Ophthalmic Imaging Clinician Review Report {case.id}")

    def new_page():
        c.showPage()
        return H - 20 * mm

    def draw_header(y):
        c.setFillColor(colors.HexColor("#0A2342"))
        c.rect(0, H - 28 * mm, W, 28 * mm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(14 * mm, H - 13 * mm, "Ophthalmic Imaging")
        c.setFont("Helvetica", 8)
        c.drawString(14 * mm, H - 18 * mm, "Ophthalmology AI Diagnostic Centre")
        c.drawString(14 * mm, H - 22 * mm, "Clinician Review Report")
        from datetime import datetime
        now = datetime.now()
        report_no = f"{case.patient.mri_number if case.patient else 'UNKNOWN'}"
        c.setFont("Helvetica-Bold", 8)
        c.drawRightString(W - 14 * mm, H - 13 * mm, f"Report No: {report_no}")
        c.setFont("Helvetica", 8)
        c.drawRightString(W - 14 * mm, H - 18 * mm, f"Date: {now.strftime('%d-%b-%Y')}  Time: {now.strftime('%H:%M')}")
        c.setFillColor(colors.black)
        return H - 35 * mm

    def draw_footer():
        c.setFont("Helvetica-Oblique", 7)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(14 * mm, 10 * mm,
            "This report reflects clinician review of AI-assisted analysis. "
            "A qualified ophthalmologist has reviewed all findings before clinical decisions.")
        c.setFillColor(colors.HexColor("#0A2342"))
        c.rect(0, 6 * mm, W, 0.5 * mm, fill=1, stroke=0)
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(14 * mm, 3 * mm, "Ophthalmic Imaging | Ophthalmology AI Diagnostic Centre")
        c.setFillColor(colors.black)

    # -- Page 1 --
    y = draw_header(H)

    # Patient info bar
    c.setFillColor(colors.HexColor("#F4F7FB"))
    c.rect(14 * mm, y - 18 * mm, W - 28 * mm, 16 * mm, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#526074"))
    c.drawString(16 * mm, y - 8 * mm, "Patient Name:")
    c.drawString(80 * mm, y - 8 * mm, "MRI Number:")
    c.drawString(140 * mm, y - 8 * mm, "Clinician:")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.black)
    patient_name = case.patient.patient_name if case.patient else "-"
    mri_number = case.patient.mri_number if case.patient else "-"
    c.drawString(16 * mm, y - 14 * mm, patient_name)
    c.drawString(80 * mm, y - 14 * mm, mri_number)
    c.drawString(140 * mm, y - 14 * mm, clinician_name)
    y -= 22 * mm

    for module_result in case.module_results:
        job_data = {}
        if module_result.model_prediction:
            try:
                if isinstance(module_result.model_prediction, dict):
                    job_data = module_result.model_prediction
                else:
                    import json
                    job_data = json.loads(module_result.model_prediction)
            except Exception:
                pass
        if not job_data and module_result.job_id:
            try:
                job_data = _read_job_result(module_result.job_id)
            except Exception:
                job_data = {}

        module_name = module_result.module.upper()
        clinician_grade = module_result.clinician_grade or module_result.model_grade or "-"
        model_grade = module_result.model_grade or "-"
        review_status = module_result.review_status or "-"
        clinician_notes = module_result.clinician_notes or ""
        confidence = module_result.model_confidence or 0.0
        confidence_pct = f"{round(confidence * 100)}%"
        flag_reason = module_result.flag_reason or ""
        probabilities = {}
        if isinstance(module_result.model_prediction, dict):
            probabilities = module_result.model_prediction.get("probabilities") or {}

        # Grade banner
        is_overridden = review_status == "overridden"
        banner_color = colors.HexColor("#1D4ED8") if not is_overridden else colors.HexColor("#92400E")
        c.setFillColor(banner_color)
        c.rect(14 * mm, y - 22 * mm, W - 28 * mm, 20 * mm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(16 * mm, y - 8 * mm, f"{module_name} MODULE — CLINICIAN REVIEW")
        c.setFont("Helvetica-Bold", 14)
        c.drawString(16 * mm, y - 17 * mm, f"Grade: {clinician_grade}")
        c.setFont("Helvetica", 8)
        c.drawString(80 * mm, y - 11 * mm, f"Confidence: {confidence_pct}")
        c.drawString(80 * mm, y - 16 * mm, f"Model grade: {model_grade}  |  Review: {review_status.title()}")
        # Large grade on right
        c.setFont("Helvetica-Bold", 28)
        c.drawRightString(W - 16 * mm, y - 18 * mm, clinician_grade)
        c.setFillColor(colors.black)
        y -= 26 * mm

        # Flag reason
        if flag_reason:
            c.setFillColor(colors.HexColor("#FFF8E1"))
            c.rect(14 * mm, y - 10 * mm, W - 28 * mm, 9 * mm, fill=1, stroke=0)
            c.setStrokeColor(colors.HexColor("#F9A825"))
            c.setLineWidth(2)
            c.line(14 * mm, y - 10 * mm, 14 * mm, y - 1 * mm)
            c.setLineWidth(1)
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(colors.HexColor("#7A5000"))
            c.drawString(17 * mm, y - 6 * mm, f"Flag: {flag_reason}")
            c.setFillColor(colors.black)
            y -= 13 * mm

        # Clinician notes
        if clinician_notes:
            c.setFont("Helvetica-Bold", 9)
            c.drawString(14 * mm, y - 6 * mm, "Clinician Notes")
            y -= 9 * mm
            c.setFillColor(colors.HexColor("#F4F7FB"))
            note_lines = []
            words = clinician_notes.split()
            line = ""
            for word in words:
                if c.stringWidth(line + " " + word, "Helvetica", 9) < (W - 32 * mm):
                    line = (line + " " + word).strip()
                else:
                    note_lines.append(line)
                    line = word
            if line:
                note_lines.append(line)
            note_h = max(len(note_lines) * 5 * mm + 6 * mm, 14 * mm)
            c.rect(14 * mm, y - note_h, W - 28 * mm, note_h, fill=1, stroke=0)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 9)
            ty = y - 6 * mm
            for nl in note_lines:
                c.drawString(16 * mm, ty, nl)
                ty -= 5 * mm
            y -= note_h + 4 * mm

        # Images for cataract
        if module_result.module == "cataract" and job_data:
            modalities = [
                ("Anterior Segment", job_data.get("anterior_segment_base64")),
                ("Red Glow", job_data.get("red_glow_base64")),
                ("Slit Lamp", job_data.get("slit_lamp_base64")),
            ]
            gradcams = [
                job_data.get("gradcam_heatmap_base64"),
                job_data.get("red_glow_gradcam_base64"),
                job_data.get("slit_lamp_gradcam_base64"),
            ]
            attention = job_data.get("attention", {})
            attn_values = [
                attention.get("anterior", 0),
                attention.get("red_glow", 0),
                attention.get("slit_lamp", 0),
            ]

            if y < 80 * mm:
                draw_footer()
                y = new_page()
                draw_header(H)
                y = H - 38 * mm

            img_w = (W - 32 * mm) / 3
            img_h = 42 * mm
            col_x = [14 * mm + i * img_w for i in range(3)]

            # Header row
            c.setFont("Helvetica-Bold", 8)
            for i, (label, _) in enumerate(modalities):
                c.setFillColor(colors.HexColor("#0A2342"))
                c.rect(col_x[i] + 1 * mm, y - 6 * mm, img_w - 2 * mm, 5 * mm, fill=1, stroke=0)
                c.setFillColor(colors.white)
                c.drawCentredString(col_x[i] + img_w / 2, y - 4 * mm, label)
            c.setFillColor(colors.black)
            y -= 7 * mm

            # Original images
            for i, (label, img_b64) in enumerate(modalities):
                if img_b64:
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        img_buf = io.BytesIO(img_bytes)
                        from reportlab.lib.utils import ImageReader
                        img_reader = ImageReader(img_buf)
                        c.drawImage(
                            img_reader, col_x[i] + 1 * mm, y - img_h,
                            width=img_w - 2 * mm, height=img_h - 1 * mm,
                            preserveAspectRatio=True, anchor="c"
                        )
                    except Exception:
                        c.setFont("Helvetica", 7)
                        c.drawCentredString(col_x[i] + img_w / 2, y - img_h / 2, "Image unavailable")
                attn_pct = f"Attention: {attn_values[i]*100:.1f}%"
                c.setFont("Helvetica", 7)
                c.setFillColor(colors.HexColor("#526074"))
                c.drawCentredString(col_x[i] + img_w / 2, y - img_h - 3 * mm, attn_pct)
                c.setFillColor(colors.black)
            y -= img_h + 8 * mm

            # Grad-CAM images
            c.setFont("Helvetica-Bold", 8)
            c.drawString(14 * mm, y - 5 * mm, "Grad-CAM Activation Maps")
            y -= 7 * mm
            for i, gcam_b64 in enumerate(gradcams):
                if gcam_b64:
                    try:
                        img_bytes = base64.b64decode(gcam_b64)
                        img_buf = io.BytesIO(img_bytes)
                        from reportlab.lib.utils import ImageReader
                        img_reader = ImageReader(img_buf)
                        c.drawImage(
                            img_reader, col_x[i] + 1 * mm, y - img_h,
                            width=img_w - 2 * mm, height=img_h - 1 * mm,
                            preserveAspectRatio=True, anchor="c"
                        )
                    except Exception:
                        pass
            y -= img_h + 6 * mm

        # Probabilities
        if probabilities and y > 40 * mm:
            if y < 60 * mm:
                draw_footer()
                y = new_page()
                y = draw_header(H)
                y = H - 38 * mm
            c.setFont("Helvetica-Bold", 9)
            c.drawString(14 * mm, y - 5 * mm, "Classification Probabilities")
            y -= 9 * mm
            bar_w = W - 28 * mm - 30 * mm
            for grade, prob in sorted(probabilities.items()):
                pct = float(prob) * 100 if float(prob) <= 1 else float(prob)
                is_top = grade == (module_result.model_grade or "")
                c.setFont("Helvetica-Bold" if is_top else "Helvetica", 9)
                c.setFillColor(colors.HexColor("#1D4ED8") if is_top else colors.black)
                c.drawString(14 * mm, y - 5 * mm, grade)
                c.setFillColor(colors.HexColor("#E5E7EB"))
                c.rect(30 * mm, y - 6 * mm, bar_w, 4 * mm, fill=1, stroke=0)
                fill_color = colors.HexColor("#1D4ED8") if is_top else colors.HexColor("#B0BEC5")
                c.setFillColor(fill_color)
                c.rect(30 * mm, y - 6 * mm, bar_w * (pct / 100), 4 * mm, fill=1, stroke=0)
                c.setFillColor(colors.black)
                c.setFont("Helvetica", 8)
                c.drawString(30 * mm + bar_w + 3 * mm, y - 5 * mm, f"{pct:.1f}%")
                y -= 8 * mm

        y -= 8 * mm
        if y < 40 * mm:
            draw_footer()
            y = new_page()
            y = draw_header(H)
            y = H - 38 * mm

    draw_footer()
    c.save()
    return output_path


@app.post("/cases/{case_id}/report/approve")
def approve_case_report(
    case_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    case = db.query(Case).filter(Case.id == case_id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    _authorize_case_or_403(case, current_user)
    before_state = _serialize_case(case)
    version_hash = case_report_hash(case.id, case.module_results)
    case.report_status = "approved"
    case.report_version_hash = version_hash
    case.report_rejection_reason = None
    case.report_approved_at = datetime.utcnow()
    case.report_approved_by = current_user.id
    ip_address, session_id = _request_metadata(request)
    try:
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="report.approved",
            case_id=case.id,
            before_state=before_state,
            after_state={"case": _serialize_case(case), "report_version_hash": version_hash},
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"case_id": case.id, "report_status": case.report_status, "report_version_hash": version_hash}


@app.post("/cases/{case_id}/report/reject")
def reject_case_report(
    case_id: str,
    payload: ReportRejectRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    case = db.query(Case).filter(Case.id == case_id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    _authorize_case_or_403(case, current_user)
    before_state = _serialize_case(case)
    case.report_status = "rejected"
    case.report_rejection_reason = payload.reason
    ip_address, session_id = _request_metadata(request)
    try:
        db.flush()
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="report.rejected",
            case_id=case.id,
            before_state=before_state,
            after_state=case,
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"case_id": case.id, "report_status": case.report_status}


@app.get("/cases/{case_id}/report/download")
def download_case_report(
    case_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    case = db.query(Case).filter(Case.id == case_id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    _authorize_case_or_403(case, current_user)
    if case.report_status != "approved":
        raise HTTPException(status_code=403, detail="Report download is blocked until the report is approved.")
    visit_with_pdf = (
        db.query(Visit)
        .filter(Visit.case_id == case_id, Visit.pdf_path.isnot(None))
        .order_by(Visit.report_timestamp.desc(), Visit.visit_date.desc())
        .first()
    )
    if visit_with_pdf and visit_with_pdf.pdf_path:
        report_path = (WORKSPACE_ROOT / visit_with_pdf.pdf_path).resolve()
    else:
        report_path = _build_case_review_pdf(case, db)
        try:
            from datetime import datetime as dt_now
            visits = db.query(Visit).filter(Visit.case_id == case_id).all()
            for visit in visits:
                visit.report_generated = True
                visit.report_timestamp = dt_now.utcnow()
                visit.pdf_path = str(report_path)
            db.flush()
        except Exception:
            pass
    ip_address, session_id = _request_metadata(request)
    try:
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="report.downloaded",
            case_id=case.id,
            after_state={"path": str(report_path)},
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return FileResponse(str(report_path), media_type="application/pdf", filename=f"{case.id}.pdf")


@app.get("/review-queue")
def get_review_queue(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    query = (
        db.query(Case)
        .filter(Case.status.in_(["pending_review", "under_review"]))
        .filter(Case.module_results.any(CaseModuleResult.flagged_for_review == 1))
    )
    if not _is_admin_user(current_user):
        query = query.filter(
            (Case.created_by == current_user.id)
            | (Case.assigned_reviewer == current_user.id)
        )
    rows = query.all()
    priority_rank = {"critical": 0, "urgent": 1, "routine": 2}
    payload = []
    now = datetime.utcnow()
    for case in rows:
        item = _serialize_case(case)
        created_at_utc = case.created_at.replace(tzinfo=None) if case.created_at else None
        age_hours = (now - created_at_utc).total_seconds() / 3600.0 if created_at_utc else 0.0
        item["age_hours"] = round(age_hours, 2)
        payload.append(item)
    payload.sort(key=lambda item: (priority_rank.get(item["priority"], 2), item["created_at"] or ""))
    return payload


@app.get("/review-queue/stats")
def get_review_queue_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    
    pending_count = db.query(Case).filter(Case.status == "pending_review").count()
    under_review_count = db.query(Case).filter(Case.status == "under_review").count()
    reviewed_rows = db.query(CaseModuleResult).filter(CaseModuleResult.reviewed_at.isnot(None)).all()
    durations = [
        (row.reviewed_at - row.case.created_at).total_seconds() / 3600.0
        for row in reviewed_rows
        if row.case and row.case.created_at and row.reviewed_at
    ]
    override_stats = {}
    for module_name in {"cataract", "glaucoma", "retina"}:
        module_rows = [row for row in reviewed_rows if row.module == module_name]
        override_stats[module_name] = (
            sum(1 for row in module_rows if row.review_status == "overridden") / len(module_rows)
            if module_rows
            else 0.0
        )
    return {
        "pending_count": pending_count,
        "under_review_count": under_review_count,
        "avg_review_time": sum(durations) / len(durations) if durations else 0.0,
        "override_rate_per_module": override_stats,
    }


@app.get("/audit-log")
def get_audit_log(
    case_id: str | None = None,
    actor_id: int | None = None,
    action: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    

    query = db.query(AuditLog)
    if case_id:
        query = query.filter(AuditLog.case_id == case_id)
    if actor_id:
        query = query.filter(AuditLog.actor_id == actor_id)
    if action:
        query = query.filter(AuditLog.action == action)
    if date_from:
        query = query.filter(AuditLog.timestamp >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(AuditLog.timestamp <= datetime.fromisoformat(date_to))
    total = query.count()
    rows = (
        query.order_by(AuditLog.timestamp.desc())
        .offset(max(page - 1, 0) * page_size)
        .limit(page_size)
        .all()
    )
    return {"items": [_audit_payload(row, db, slim=True) for row in rows], "total": total, "page": page, "page_size": page_size}


@app.get("/audit-log/export")
def export_audit_log(
    request: Request,
    case_id: str | None = None,
    actor_id: int | None = None,
    action: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    query = db.query(AuditLog)
    if case_id:
        query = query.filter(AuditLog.case_id == case_id)
    if actor_id:
        query = query.filter(AuditLog.actor_id == actor_id)
    if action:
        query = query.filter(AuditLog.action == action)
    if date_from:
        query = query.filter(AuditLog.timestamp >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(AuditLog.timestamp <= datetime.fromisoformat(date_to))
    rows = query.order_by(AuditLog.timestamp.desc()).all()
    ip_address, session_id = _request_metadata(request)
    try:
        log_audit_event(
            db,
            actor_id=current_user.id,
            action="audit.exported",
            before_state={"filters": {"case_id": case_id, "actor_id": actor_id, "action": action, "date_from": date_from, "date_to": date_to}},
            after_state={"row_count": len(rows)},
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["timestamp", "actor_id", "action", "case_id", "module_result_id", "before_state", "after_state"])
    for row in rows:
        writer.writerow([
            row.timestamp.isoformat() if row.timestamp else "",
            row.actor_id,
            row.action,
            row.case_id or "",
            row.module_result_id or "",
            json.dumps(row.before_state or {}),
            json.dumps(row.after_state or {}),
        ])
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audit-log.csv"})


@app.get("/thresholds/{module}")
def get_threshold_config(
    module: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    
    return ReviewThresholdEngine(db).get_thresholds(module)


@app.patch("/thresholds/{module}")
def update_threshold_config(
    module: str,
    payload: ThresholdUpdateRequest,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    try:
        row = ReviewThresholdEngine(db).update_thresholds(module, payload.model_dump(), admin_user.id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {
        "id": row.id,
        "module": row.module,
        "threshold_key": row.threshold_key,
        "threshold_value": row.threshold_value,
        "priority_level": row.priority_level,
        "reason_template": row.reason_template,
    }


@app.get("/thresholds/history/{module}")
def threshold_history(
    module: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    
    rows = (
        db.query(ReviewThresholdConfig)
        .filter(ReviewThresholdConfig.module == normalize_module_name(module))
        .order_by(ReviewThresholdConfig.updated_at.desc())
        .all()
    )
    return [
        {
            "id": row.id,
            "threshold_key": row.threshold_key,
            "threshold_value": row.threshold_value,
            "priority_level": row.priority_level,
            "reason_template": row.reason_template,
            "is_active": row.is_active,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "updated_by": row.updated_by,
        }
        for row in rows
    ]


@app.post("/admin/calibration/{module}/fit")
def fit_calibration(
    module: str,
    payload: CalibrationFitRequest,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    scaler = TemperatureScaler(module, db)
    fit_result = scaler.fit(np.asarray(payload.logits, dtype=np.float64), np.asarray(payload.labels, dtype=np.int64), fitted_by=admin_user.id)
    probs = scaler.calibrate(np.asarray(payload.logits, dtype=np.float64))
    bins = scaler.reliability_bins(probs, np.asarray(payload.labels, dtype=np.int64))
    ip_address, session_id = _request_metadata(request)
    try:
        log_audit_event(
            db,
            actor_id=admin_user.id,
            action="calibration.applied",
            before_state={"module": normalize_module_name(module)},
            after_state={"module": normalize_module_name(module), "temperature": fit_result.temperature, "bins": bins},
            ip_address=ip_address,
            session_id=session_id,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"module": normalize_module_name(module), "temperature": fit_result.temperature, "ece_before": fit_result.baseline_ece, "ece_after": fit_result.validation_ece}


@app.get("/admin/calibration/{module}/status")
def get_calibration_status(
    module: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    del admin_user
    row = (
        db.query(CalibrationConfig)
        .filter(CalibrationConfig.module == normalize_module_name(module), CalibrationConfig.is_active == 1)
        .order_by(CalibrationConfig.fitted_at.desc())
        .first()
    )
    if row is None:
        return {"module": normalize_module_name(module), "is_calibrated": False}
    return {
        "module": row.module,
        "is_calibrated": True,
        "temperature": row.temperature,
        "ece": row.validation_ece,
        "baseline_ece": row.baseline_ece,
        "fitted_at": row.fitted_at.isoformat() if row.fitted_at else None,
        "sample_count": row.sample_count,
    }


@app.get("/admin/calibration/reliability-diagram/{module}")
def get_reliability_diagram(
    module: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    del admin_user
    audit_entry = (
        db.query(AuditLog)
        .filter(AuditLog.action == "calibration.applied")
        .order_by(AuditLog.timestamp.desc())
        .all()
    )
    target_module = normalize_module_name(module)
    for entry in audit_entry:
        after_state = entry.after_state or {}
        if after_state.get("module") == target_module:
            return after_state.get("bins") or []
    return []


@app.get("/validation-report/{module}")
def get_validation_report(
    module: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    del admin_user
    if normalize_module_name(module) == "retina":
        raise HTTPException(
            status_code=409,
            detail="Retina validation metrics are invalidated pending leakage-free re-evaluation; see PRIVACY_REVIEW.md.",
        )
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    report = ModuleValidationReport(db, generated_by="api").generate(module, version)
    db.commit()
    return json.loads(json.dumps(report.__dict__, default=str))


@app.get("/validation-report/{module}/pdf")
def download_validation_report_pdf(
    module: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Response:
    del admin_user
    if normalize_module_name(module) == "retina":
        raise HTTPException(
            status_code=409,
            detail="Retina validation metrics are invalidated pending leakage-free re-evaluation; see PRIVACY_REVIEW.md.",
        )
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    report_engine = ModuleValidationReport(db, generated_by="api")
    report = report_engine.generate(module, version)
    bins: list[dict[str, Any]] = []
    target_module = normalize_module_name(module)
    for entry in (
        db.query(AuditLog)
        .filter(AuditLog.action == "calibration.applied")
        .order_by(AuditLog.timestamp.desc())
        .all()
    ):
        after_state = entry.after_state or {}
        if after_state.get("module") == target_module:
            bins = after_state.get("bins") or []
            break
    pdf_bytes = report_engine.render_pdf(report, bins)
    db.commit()
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={normalize_module_name(module)}-validation.pdf"})


@app.get("/validation-report/{module}/history")
def validation_report_history(
    module: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    del admin_user
    rows = (
        db.query(ValidationReportSnapshot)
        .filter(ValidationReportSnapshot.module == normalize_module_name(module))
        .order_by(ValidationReportSnapshot.generated_at.desc())
        .all()
    )
    return [
        {"id": row.id, "module": row.module, "version": row.version, "generated_at": row.generated_at.isoformat() if row.generated_at else None}
        for row in rows
    ]


@app.get("/admin/performance/overview")
def get_performance_overview(
    date_from: str | None = None,
    date_to: str | None = None,
    module: str | None = None,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    del admin_user
    return performance_overview(db, date_from=date_from, date_to=date_to, module=module)


@app.get("/admin/performance/override-analysis")
def get_override_analysis(
    date_from: str | None = None,
    date_to: str | None = None,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    del admin_user
    return override_analysis(db, date_from=date_from, date_to=date_to)


@app.get("/admin/performance/calibration-drift")
def get_performance_calibration_drift(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    del admin_user
    return calibration_drift(db)


@app.get("/admin/performance/reviewer-agreement")
def get_performance_reviewer_agreement(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    del admin_user
    return reviewer_agreement(db)


@app.get("/admin/performance/throughput")
def get_performance_throughput(
    date_from: str | None = None,
    date_to: str | None = None,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    del admin_user
    return throughput_metrics(db, date_from=date_from, date_to=date_to)


def _coerce_batch_report_item(item: BatchReportItem) -> dict[str, Any]:
    if item.result is not None:
        payload = dict(item.result)
        if item.analyzed_at:
            try:
                payload["_analyzed_at"] = datetime.fromisoformat(item.analyzed_at)
            except ValueError:
                payload["_analyzed_at"] = datetime.now()
        else:
            payload["_analyzed_at"] = datetime.now()
        return payload

    payload = item.model_dump(exclude_none=True, exclude={"result", "analyzed_at"})
    if item.analyzed_at:
        try:
            payload["_analyzed_at"] = datetime.fromisoformat(item.analyzed_at)
        except ValueError:
            payload["_analyzed_at"] = datetime.now()
    return payload

def _generate_medgemma_texts_for_record(record: dict) -> None:
    result = record.get('result') if isinstance(record, dict) else record
    if not isinstance(result, dict):
        return
    grade = result.get('grade')
    if grade not in {'NS1', 'NS2', 'NS3', 'NS4'}:
        return
    if result.get('report_texts'):
        return
    try:
        from classifier import _cuda_available
        from report_text_engine import V7Result, generate_report_texts
        import torch
        classifier_svc = get_classifier_service()
        if _cuda_available() and classifier_svc._engine is not None:
            print('[Report] Unloading V7 from VRAM for MedGemma generation...')
            del classifier_svc._engine
            classifier_svc._engine = None
            classifier_svc._gradcam = None
            torch.cuda.empty_cache()
            print(f'[Report] VRAM freed: {torch.cuda.memory_allocated()/1e9:.1f}GB used')
        attention = result.get('attention') or {}
        v7_input = V7Result(
            grade=grade,
            confidence=float(result.get('confidence') or 0.0),
            probabilities={k: float(v) for k, v in (result.get('probabilities') or {}).items()},
            attention={
                'anterior': float(attention.get('anterior', 0.0)),
                'red_glow': float(attention.get('red_glow', 0.0)),
                'slit_lamp': float(attention.get('slit_lamp', 0.0)),
            },
            case_id=result.get('case_id'),
            gradcam_region_info=result.get('gradcam_region_info') or {},
            raw_result=result,
        )
        texts = generate_report_texts(v7_input)
        result['report_texts'] = texts
        print('[Report] MedGemma text generation complete.')
        try:
            from medgemma_engine import unload as medgemma_unload
            medgemma_unload()
        except Exception:
            pass
    except Exception as exc:
        print(f'[Report] MedGemma generation failed (non-fatal): {exc}')
