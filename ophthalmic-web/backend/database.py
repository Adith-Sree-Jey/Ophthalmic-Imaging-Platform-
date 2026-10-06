from __future__ import annotations

import hashlib
import json
import os
import uuid
import datetime as dt
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import (
  JSON,
  Boolean,
  Column,
  DateTime,
  Float,
  ForeignKey,
  Index,
  Integer,
  String,
  Text,
  create_engine,
  inspect,
  text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=ENV_PATH)

SERVER = os.getenv("MSSQL_SERVER", r"localhost\SQLEXPRESS")
DATABASE = os.getenv("MSSQL_DATABASE")
DRIVER = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
USERNAME = os.getenv("MSSQL_USERNAME", "")
PASSWORD = os.getenv("MSSQL_PASSWORD", "")
ENCRYPT = os.getenv("MSSQL_ENCRYPT", "no")

if not DATABASE:
  raise RuntimeError(f"MSSQL_DATABASE must be set in {ENV_PATH}.")

if USERNAME:
  auth = f"UID={USERNAME};PWD={PASSWORD}"
else:
  auth = "Trusted_Connection=yes"

odbc_connect = quote_plus(
  ";".join(
    [
      f"DRIVER={{{DRIVER}}}",
      f"SERVER={SERVER}",
      f"DATABASE={DATABASE}",
      auth,
      f"Encrypt={ENCRYPT}",
      "TrustServerCertificate=yes",
    ]
  )
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={odbc_connect}", echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()


def new_uuid() -> str:
  return str(uuid.uuid4())


def utcnow() -> datetime:
  return datetime.utcnow()


def utcnow_sql():
  return text("GETUTCDATE()")


class Base(DeclarativeBase):
  pass


class User(Base):
  __tablename__ = "users"

  id = Column(Integer, primary_key=True, index=True)
  username = Column(String(100), unique=True, nullable=False, index=True)
  hashed_password = Column(String(200), nullable=False)
  role = Column(String(50), nullable=False, default="doctor")
  is_active = Column(Boolean, default=True)
  created_at = Column(DateTime, default=dt.datetime.utcnow)


class Patient(Base):
  __tablename__ = "patients"

  id = Column(Integer, primary_key=True, index=True)
  mri_number = Column(String(100), unique=True, nullable=False, index=True)
  patient_name = Column(String(200), nullable=True)
  created_at = Column(DateTime, default=datetime.utcnow)
  created_by = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

  visits = relationship("Visit", back_populates="patient", order_by="Visit.visit_date")
  cases = relationship("Case", back_populates="patient", order_by="Case.created_at")


class Visit(Base):
  __tablename__ = "visits"

  id = Column(Integer, primary_key=True, index=True)
  patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
  case_id = Column(String(200), nullable=True, index=True)
  eye_side = Column(String(10), nullable=False, default="UNKNOWN")
  module = Column(String(50), nullable=False, default="cataract")
  result_summary = Column(String(500), nullable=True)
  report_generated = Column(Boolean, nullable=False, default=False)
  report_timestamp = Column(DateTime, nullable=True)
  pdf_path = Column(String(500), nullable=True)
  visit_date = Column(DateTime, default=datetime.utcnow)
  graded_by = Column(String(100), nullable=True)

  patient = relationship("Patient", back_populates="visits")
  classification = relationship("Classification", back_populates="visit", uselist=False)


class Classification(Base):
  __tablename__ = "classifications"

  id = Column(Integer, primary_key=True, index=True)
  visit_id = Column(Integer, ForeignKey("visits.id"), nullable=False, unique=True)
  grade = Column(String(10), nullable=False)
  grade_index = Column(Integer, nullable=False)
  predicted_class = Column(String(100), nullable=True)
  confidence = Column(Float, nullable=False)
  probabilities_json = Column(Text, nullable=True)
  prob_ns1 = Column(Float, nullable=True)
  prob_ns2 = Column(Float, nullable=True)
  prob_ns3 = Column(Float, nullable=True)
  prob_ns4 = Column(Float, nullable=True)
  attn_anterior = Column(Float, nullable=True)
  attn_red_glow = Column(Float, nullable=True)
  attn_slit_lamp = Column(Float, nullable=True)
  binary_ns2plus = Column(Float, nullable=True)
  needs_review = Column(Boolean, default=False)
  review_reason = Column(Text, nullable=True)
  pupil_crop_applied = Column(Boolean, default=False)
  recommendation = Column(Text, nullable=True)
  severity = Column(String(50), nullable=True)
  model_info = Column(String(200), nullable=True)

  visit = relationship("Visit", back_populates="classification")


class InferenceJob(Base):
  __tablename__ = "inference_jobs"

  id = Column(String(36), primary_key=True)
  status = Column(String(20), default="pending", nullable=False)
  result_path = Column(String(500), nullable=True)
  error = Column(Text, nullable=True)
  created_at = Column(DateTime, default=datetime.utcnow)
  created_by = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

  case_module_results = relationship("CaseModuleResult", back_populates="job")


class Case(Base):
  __tablename__ = "cases"

  id = Column(String(36), primary_key=True, default=new_uuid)
  patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
  created_at = Column(DateTime, nullable=False, server_default=utcnow_sql())
  created_by = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
  status = Column(String(30), nullable=False, default="pending_review", index=True)
  module_types = Column(JSON, nullable=False, default=list)
  assigned_reviewer = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
  priority = Column(String(20), nullable=False, default="routine", index=True)
  notes = Column(Text, nullable=True)
  report_status = Column(String(20), nullable=False, default="pending")
  report_version_hash = Column(String(128), nullable=True)
  report_rejection_reason = Column(Text, nullable=True)
  report_approved_at = Column(DateTime, nullable=True)
  report_approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)

  patient = relationship("Patient", back_populates="cases")
  module_results = relationship("CaseModuleResult", back_populates="case", cascade="all, delete-orphan")
  audit_entries = relationship("AuditLog", back_populates="case")


class CaseModuleResult(Base):
  __tablename__ = "case_module_results"

  id = Column(String(36), primary_key=True, default=new_uuid)
  case_id = Column(String(36), ForeignKey("cases.id"), nullable=False, index=True)
  module = Column(String(30), nullable=False, index=True)
  job_id = Column(String(36), ForeignKey("inference_jobs.id"), nullable=False, index=True)
  model_prediction = Column(JSON, nullable=False)
  model_confidence = Column(Float, nullable=False, default=0.0)
  model_grade = Column(String(100), nullable=True)
  review_status = Column(String(20), nullable=False, default="pending", index=True)
  reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
  reviewed_at = Column(DateTime, nullable=True)
  clinician_grade = Column(String(100), nullable=True)
  clinician_notes = Column(Text, nullable=True)
  flagged_for_review = Column(Boolean, nullable=False, default=False, index=True)
  flag_reason = Column(Text, nullable=True)

  case = relationship("Case", back_populates="module_results")
  job = relationship("InferenceJob", back_populates="case_module_results")
  audit_entries = relationship("AuditLog", back_populates="module_result")


class AuditLog(Base):
  __tablename__ = "audit_log"

  id = Column(String(36), primary_key=True, default=new_uuid)
  timestamp = Column(DateTime, nullable=False, server_default=utcnow_sql(), index=True)
  actor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
  action = Column(String(80), nullable=False, index=True)
  case_id = Column(String(36), ForeignKey("cases.id"), nullable=True, index=True)
  module_result_id = Column(String(36), ForeignKey("case_module_results.id"), nullable=True, index=True)
  before_state = Column(JSON, nullable=True)
  after_state = Column(JSON, nullable=True)
  ip_address = Column(String(100), nullable=True)
  session_id = Column(String(100), nullable=True)

  case = relationship("Case", back_populates="audit_entries")
  module_result = relationship("CaseModuleResult", back_populates="audit_entries")


class CalibrationConfig(Base):
  __tablename__ = "calibration_configs"

  id = Column(String(36), primary_key=True, default=new_uuid)
  module = Column(String(30), nullable=False, index=True)
  temperature = Column(Float, nullable=False)
  fitted_at = Column(DateTime, nullable=False, default=utcnow, server_default=utcnow_sql())
  fitted_by = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
  validation_ece = Column(Float, nullable=True)
  baseline_ece = Column(Float, nullable=True)
  sample_count = Column(Integer, nullable=False, default=0)
  is_active = Column(Boolean, nullable=False, default=True, index=True)


class ReviewThresholdConfig(Base):
  __tablename__ = "review_threshold_configs"

  id = Column(String(36), primary_key=True, default=new_uuid)
  module = Column(String(30), nullable=False, index=True)
  threshold_key = Column(String(100), nullable=False, index=True)
  threshold_value = Column(Float, nullable=False)
  priority_level = Column(String(20), nullable=False, default="routine")
  reason_template = Column(Text, nullable=False)
  is_active = Column(Boolean, nullable=False, default=True, index=True)
  updated_at = Column(DateTime, nullable=False, default=utcnow, server_default=utcnow_sql())
  updated_by = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)


class ValidationReportSnapshot(Base):
  __tablename__ = "validation_reports"

  id = Column(String(36), primary_key=True, default=new_uuid)
  module = Column(String(30), nullable=False, index=True)
  version = Column(String(100), nullable=False, index=True)
  generated_at = Column(DateTime, nullable=False, default=utcnow, server_default=utcnow_sql())
  generated_by = Column(String(100), nullable=True)
  report_json = Column(JSON, nullable=False)


Index("ix_case_module_results_case_module", CaseModuleResult.case_id, CaseModuleResult.module)
Index("ix_audit_log_case_action", AuditLog.case_id, AuditLog.action)
Index("ix_threshold_module_active", ReviewThresholdConfig.module, ReviewThresholdConfig.is_active)
Index("ix_calibration_module_active", CalibrationConfig.module, CalibrationConfig.is_active)
Index("ix_validation_reports_module_generated", ValidationReportSnapshot.module, ValidationReportSnapshot.generated_at)


DEFAULT_THRESHOLD_ROWS = [
  ("cataract", "confidence", 0.75, "routine", "Low model confidence"),
  ("cataract", "ns4_grade", 4.0, "urgent", "Severe nuclear sclerosis, surgical review required"),
  ("cataract", "ordinal_head_disagreement", 0.2, "routine", "Head disagreement detected"),
  ("cataract", "minimum_modality_count", 2.0, "routine", "Single modality input, reduced reliability"),
  ("glaucoma", "confidence", 0.80, "routine", "Low model confidence"),
  ("glaucoma", "predicted_glaucoma", 1.0, "urgent", "Positive glaucoma prediction"),
  ("glaucoma", "cup_to_disc_ratio", 0.7, "critical", "High CDR, immediate review required"),
  ("glaucoma", "poor_vessel_quality", 1.0, "routine", "Poor vessel segmentation quality"),
  ("retina", "vessel_coverage", 0.3, "routine", "Low vessel coverage, segmentation may be unreliable"),
  ("retina", "image_quality_score", 0.6, "routine", "Low input image quality"),
]


def init_db():
  Base.metadata.create_all(bind=engine)
  _ensure_visit_columns()
  _ensure_classification_columns()
  _ensure_case_support_columns()
  _ensure_inference_job_columns()
  _seed_default_review_thresholds()
  print("Database tables initialised.")


def _ensure_visit_columns() -> None:
  inspector = inspect(engine)
  if "visits" not in inspector.get_table_names():
    return

  visit_columns = {column["name"] for column in inspector.get_columns("visits")}
  with engine.begin() as conn:
    if "module" not in visit_columns:
      conn.execute(
        text(
          "ALTER TABLE visits "
          "ADD module VARCHAR(50) NOT NULL "
          "CONSTRAINT DF_visits_module DEFAULT 'cataract'"
        )
      )
    if "result_summary" not in visit_columns:
      conn.execute(text("ALTER TABLE visits ADD result_summary NVARCHAR(500) NULL"))
    if "report_generated" not in visit_columns:
      conn.execute(
        text(
          "ALTER TABLE visits "
          "ADD report_generated BIT NOT NULL "
          "CONSTRAINT DF_visits_report_generated DEFAULT 0"
        )
      )
    if "report_timestamp" not in visit_columns:
      conn.execute(text("ALTER TABLE visits ADD report_timestamp DATETIME NULL"))
    if "pdf_path" not in visit_columns:
      conn.execute(text("ALTER TABLE visits ADD pdf_path NVARCHAR(500) NULL"))


def _ensure_classification_columns() -> None:
  inspector = inspect(engine)
  if "classifications" not in inspector.get_table_names():
    return

  classification_columns = {column["name"] for column in inspector.get_columns("classifications")}
  with engine.begin() as conn:
    if "predicted_class" not in classification_columns:
      conn.execute(text("ALTER TABLE classifications ADD predicted_class NVARCHAR(100) NULL"))
    if "probabilities_json" not in classification_columns:
      conn.execute(text("ALTER TABLE classifications ADD probabilities_json NVARCHAR(MAX) NULL"))


def _ensure_case_support_columns() -> None:
  inspector = inspect(engine)
  if "cases" not in inspector.get_table_names():
    return
  case_columns = {column["name"] for column in inspector.get_columns("cases")}
  with engine.begin() as conn:
    if "report_status" not in case_columns:
      conn.execute(text("ALTER TABLE cases ADD report_status VARCHAR(20) NOT NULL CONSTRAINT DF_cases_report_status DEFAULT 'pending'"))
    if "report_version_hash" not in case_columns:
      conn.execute(text("ALTER TABLE cases ADD report_version_hash NVARCHAR(128) NULL"))
    if "report_rejection_reason" not in case_columns:
      conn.execute(text("ALTER TABLE cases ADD report_rejection_reason NVARCHAR(MAX) NULL"))
    if "report_approved_at" not in case_columns:
      conn.execute(text("ALTER TABLE cases ADD report_approved_at DATETIME NULL"))
    if "report_approved_by" not in case_columns:
      conn.execute(text("ALTER TABLE cases ADD report_approved_by INT NULL"))


def _ensure_inference_job_columns() -> None:
  # Emergency runtime compatibility only; Alembic owns the FK and index.
  inspector = inspect(engine)
  if "inference_jobs" not in inspector.get_table_names():
    return
  job_columns = {column["name"] for column in inspector.get_columns("inference_jobs")}
  with engine.begin() as conn:
    if "created_by" not in job_columns:
      conn.execute(text("ALTER TABLE inference_jobs ADD created_by INT NULL"))


def _seed_default_review_thresholds() -> None:
  db = SessionLocal()
  try:
    existing_keys = {
      (row.module, row.threshold_key)
      for row in db.query(ReviewThresholdConfig).filter(
        ReviewThresholdConfig.is_active == 1  # FIX: was .is_(True) which generates IS 1 on MSSQL
      ).all()
    }
    created = False
    for module, threshold_key, value, priority, reason in DEFAULT_THRESHOLD_ROWS:
      if (module, threshold_key) in existing_keys:
        continue
      db.add(
        ReviewThresholdConfig(
          module=module,
          threshold_key=threshold_key,
          threshold_value=value,
          priority_level=priority,
          reason_template=reason,
          is_active=True,
        )
      )
      created = True
    if created:
      db.commit()
  finally:
    db.close()


def normalize_module_name(module: Optional[str]) -> str:
  raw = (module or "cataract").strip().lower()
  if raw in {"retina", "retina_segmentation"}:
    return "retina"
  return raw


def normalize_confidence(value: Any) -> float:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return 0.0
  if number > 1.0:
    number = number / 100.0
  return max(0.0, min(1.0, number))


def serialize_for_audit(value: Any) -> Any:
  if value is None:
    return None
  if isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, datetime):
    return value.isoformat()
  if isinstance(value, list):
    return [serialize_for_audit(item) for item in value]
  if isinstance(value, dict):
    return {str(key): serialize_for_audit(val) for key, val in value.items()}
  if hasattr(value, "__table__"):
    payload: dict[str, Any] = {}
    for column in value.__table__.columns:
      payload[column.name] = serialize_for_audit(getattr(value, column.name))
    return payload
  return str(value)


def log_audit_event(
  db: Session,
  *,
  actor_id: int,
  action: str,
  case_id: Optional[str] = None,
  module_result_id: Optional[str] = None,
  before_state: Any = None,
  after_state: Any = None,
  ip_address: Optional[str] = None,
  session_id: Optional[str] = None,
) -> AuditLog:
  entry = AuditLog(
    actor_id=actor_id,
    action=action,
    case_id=case_id,
    module_result_id=module_result_id,
    before_state=serialize_for_audit(before_state),
    after_state=serialize_for_audit(after_state),
    ip_address=ip_address,
    session_id=session_id,
  )
  db.add(entry)
  db.flush()
  return entry


def upsert_patient(db: Session, mri_number: str, patient_name: Optional[str] = None) -> Patient:
  patient = db.query(Patient).filter(Patient.mri_number == mri_number).first()
  if patient is None:
    patient = Patient(mri_number=mri_number, patient_name=patient_name)
    db.add(patient)
    db.flush()
  elif patient_name and patient.patient_name != patient_name:
    patient.patient_name = patient_name
    db.flush()
  return patient


def get_or_create_case(
  db: Session,
  *,
  patient: Patient,
  case_id: Optional[str],
  created_by: Optional[int],
  module_types: list[str],
  priority: str = "routine",
  notes: Optional[str] = None,
) -> Case:
  normalized_case_id = (case_id or "").strip()
  if normalized_case_id:
    existing_case = db.query(Case).filter(Case.id == normalized_case_id).first()
    if existing_case is not None:
      requester = db.query(User).filter(User.id == created_by).first() if created_by is not None else None
      requester_is_admin = requester is not None and requester.role == "admin"
      if not requester_is_admin and created_by not in {
        existing_case.created_by,
        existing_case.assigned_reviewer,
      }:
        raise PermissionError("Requester is not authorized to reuse this existing case.")
      merged_types = sorted(set((existing_case.module_types or []) + module_types))
      if merged_types != (existing_case.module_types or []):
        existing_case.module_types = merged_types
      db.flush()
      return existing_case

  new_case = Case(
    id=new_uuid(),
    patient_id=patient.id,
    created_by=created_by or patient.created_by or 1,
    status="pending_review",
    module_types=module_types,
    priority=priority,
    notes=notes,
  )
  db.add(new_case)
  db.flush()
  return new_case


def create_visit(
  db: Session,
  patient: Patient,
  case_id: Optional[str],
  eye_side: str,
  graded_by: Optional[str],
  module: str = "cataract",
  result_summary: Optional[str] = None,
) -> Visit:
  visit = Visit(
    patient_id=patient.id,
    case_id=case_id,
    eye_side=(eye_side or "UNKNOWN").upper(),
    module=module,
    result_summary=result_summary,
    graded_by=graded_by,
  )
  db.add(visit)
  db.flush()
  return visit


def _grade_index_from_result(result: dict) -> int:
  explicit = result.get("grade_index")
  if isinstance(explicit, int):
    return explicit
  return {"NS1": 1, "NS2": 2, "NS3": 3, "NS4": 4}.get(result.get("grade"), 0)


def create_classification(db: Session, visit: Visit, result: dict, module: str = "cataract") -> Classification:
  probabilities = result.get("probabilities", {})
  attention = result.get("attention", {})
  predicted_class = result.get("predicted_class") or result.get("grade") or "Unknown"
  grade = result.get("grade") if module == "cataract" else predicted_class
  grade_index = _grade_index_from_result(result) if module == "cataract" else 0
  serialized_probabilities = json.dumps(probabilities) if probabilities else None

  classification = Classification(
    visit_id=visit.id,
    grade=grade or "Unknown",
    grade_index=grade_index,
    predicted_class=predicted_class,
    confidence=result.get("confidence", 0.0) or 0.0,
    probabilities_json=serialized_probabilities,
    prob_ns1=probabilities.get("NS1"),
    prob_ns2=probabilities.get("NS2"),
    prob_ns3=probabilities.get("NS3"),
    prob_ns4=probabilities.get("NS4"),
    attn_anterior=attention.get("anterior"),
    attn_red_glow=attention.get("red_glow"),
    attn_slit_lamp=attention.get("slit_lamp"),
    binary_ns2plus=result.get("binary_ns2plus_prob"),
    needs_review=result.get("needs_review", False),
    review_reason=result.get("review_reason"),
    pupil_crop_applied=result.get("pupil_crop_applied", False),
    recommendation=result.get("recommendation"),
    severity=result.get("severity"),
    model_info=result.get("model_info"),
  )
  db.add(classification)
  db.flush()
  return classification


def upsert_case_module_result(
  db: Session,
  *,
  case_id: str,
  module: str,
  job_id: str,
  result: dict,
  flagged_for_review: bool,
  flag_reason: Optional[str],
) -> CaseModuleResult:
  normalized_module = normalize_module_name(module)
  module_result = (
    db.query(CaseModuleResult)
    .filter(CaseModuleResult.case_id == case_id, CaseModuleResult.module == normalized_module)
    .first()
  )
  payload_copy = json.loads(json.dumps(result, default=str))
  if module_result is None:
    module_result = CaseModuleResult(
      case_id=case_id,
      module=normalized_module,
      job_id=job_id,
      model_prediction=payload_copy,
      model_confidence=normalize_confidence(result.get("confidence")),
      model_grade=result.get("grade") or result.get("predicted_class"),
      review_status="pending",
      flagged_for_review=flagged_for_review,
      flag_reason=flag_reason,
    )
    db.add(module_result)
  else:
    module_result.job_id = job_id
    module_result.model_prediction = payload_copy
    module_result.model_confidence = normalize_confidence(result.get("confidence"))
    module_result.model_grade = result.get("grade") or result.get("predicted_class")
    module_result.review_status = "pending"
    module_result.reviewer_id = None
    module_result.reviewed_at = None
    module_result.clinician_grade = None
    module_result.clinician_notes = None
    module_result.flagged_for_review = flagged_for_review
    module_result.flag_reason = flag_reason
  db.flush()
  return module_result


def save_full_result(
  db: Session,
  mri_number: str,
  patient_name: Optional[str],
  case_id: Optional[str],
  eye_side: str,
  graded_by: Optional[str],
  result: dict,
  module: str = "cataract",
  job_id: Optional[str] = None,
  created_by_user_id: Optional[int] = None,
  flagged_for_review: bool = False,
  flag_reason: Optional[str] = None,
) -> dict:
  patient = upsert_patient(db, mri_number, patient_name)
  case = get_or_create_case(
    db,
    patient=patient,
    case_id=case_id,
    created_by=created_by_user_id,
    module_types=[normalize_module_name(module)],
  )
  desired_priority = result.get("review_priority")
  priority_rank = {"routine": 0, "urgent": 1, "critical": 2}
  if desired_priority and priority_rank.get(desired_priority, 0) > priority_rank.get(case.priority or "routine", 0):
    case.priority = desired_priority
  visit_module = "retina_segmentation" if normalize_module_name(module) == "retina" else module
  visit = create_visit(db, patient, case.id, eye_side, graded_by, module=visit_module)
  classification = create_classification(db, visit, result, module=module)
  module_result = None
  if job_id:
    module_result = upsert_case_module_result(
      db,
      case_id=case.id,
      module=module,
      job_id=job_id,
      result=result,
      flagged_for_review=flagged_for_review,
      flag_reason=flag_reason,
    )
  db.commit()
  return {
    "patient_id": patient.id,
    "visit_id": visit.id,
    "clf_id": classification.id,
    "case_id": case.id,
    "module_result_id": module_result.id if module_result else None,
  }


def save_retina_result(
  db: Session,
  mri_number: str,
  patient_name: Optional[str],
  case_id: Optional[str],
  eye_side: str,
  graded_by: Optional[str],
  result_summary: Optional[str],
  result_payload: Optional[dict] = None,
  job_id: Optional[str] = None,
  created_by_user_id: Optional[int] = None,
  flagged_for_review: bool = False,
  flag_reason: Optional[str] = None,
) -> dict:
  patient = upsert_patient(db, mri_number, patient_name)
  case = get_or_create_case(
    db,
    patient=patient,
    case_id=case_id,
    created_by=created_by_user_id,
    module_types=["retina"],
  )
  desired_priority = (result_payload or {}).get("review_priority")
  priority_rank = {"routine": 0, "urgent": 1, "critical": 2}
  if desired_priority and priority_rank.get(desired_priority, 0) > priority_rank.get(case.priority or "routine", 0):
    case.priority = desired_priority
  visit = create_visit(
    db,
    patient,
    case.id,
    eye_side,
    graded_by,
    module="retina_segmentation",
    result_summary=result_summary,
  )
  module_result = None
  if job_id:
    module_result = upsert_case_module_result(
      db,
      case_id=case.id,
      module="retina",
      job_id=job_id,
      result=result_payload or {"result_summary": result_summary},
      flagged_for_review=flagged_for_review,
      flag_reason=flag_reason,
    )
  db.commit()
  return {
    "patient_id": patient.id,
    "visit_id": visit.id,
    "case_id": case.id,
    "module_result_id": module_result.id if module_result else None,
  }


def mark_report_generated(
  db: Session,
  *,
  mri_number: str,
  case_id: Optional[str],
  eye_side: str,
  module: str,
  pdf_path: str,
  report_timestamp: Optional[datetime] = None,
) -> Optional[int]:
  patient = db.query(Patient).filter(Patient.mri_number == mri_number).first()
  if patient is None:
    return None

  query = (
    db.query(Visit)
    .filter(Visit.patient_id == patient.id)
    .filter(Visit.module == module)
    .filter(Visit.eye_side == (eye_side or "UNKNOWN").upper())
  )
  if case_id:
    query = query.filter(Visit.case_id == case_id)

  visit = query.order_by(Visit.visit_date.desc()).first()
  if visit is None:
    return None

  visit.report_generated = True
  visit.report_timestamp = report_timestamp or datetime.utcnow()
  visit.pdf_path = pdf_path
  db.flush()
  db.commit()
  return visit.id


def get_patient_history(db: Session, mri_number: str) -> list[dict]:
  patient = db.query(Patient).filter(Patient.mri_number == mri_number).first()
  if not patient:
    return []

  rows: list[dict] = []
  for visit in reversed(patient.visits):
    classification = visit.classification
    parsed_probabilities = None
    if classification and classification.probabilities_json:
      try:
        parsed_probabilities = json.loads(classification.probabilities_json)
      except json.JSONDecodeError:
        parsed_probabilities = None
    row = {
      "patient_name": patient.patient_name,
      "visit_id": visit.id,
      "case_id": visit.case_id,
      "eye_side": visit.eye_side,
      "module": visit.module or "cataract",
      "result_summary": visit.result_summary,
      "report_generated": bool(visit.report_generated),
      "report_timestamp": visit.report_timestamp.isoformat() if visit.report_timestamp else None,
      "pdf_path": visit.pdf_path,
      "visit_date": visit.visit_date.isoformat() if visit.visit_date else None,
      "graded_by": visit.graded_by,
    }
    if classification:
      fallback_probabilities = {
        "NS1": classification.prob_ns1,
        "NS2": classification.prob_ns2,
        "NS3": classification.prob_ns3,
        "NS4": classification.prob_ns4,
      }
      row.update(
        {
          "grade": classification.grade,
          "grade_index": classification.grade_index,
          "predicted_class": classification.predicted_class or classification.grade,
          "confidence": classification.confidence,
          "probabilities": parsed_probabilities or fallback_probabilities,
          "attention": {
            "anterior": classification.attn_anterior,
            "red_glow": classification.attn_red_glow,
            "slit_lamp": classification.attn_slit_lamp,
          },
          "needs_review": classification.needs_review,
          "review_reason": classification.review_reason,
          "recommendation": classification.recommendation,
          "severity": classification.severity,
        }
      )
    rows.append(row)
  return rows


def case_report_hash(case_id: str, module_results: list[CaseModuleResult]) -> str:
  payload = {
    "case_id": case_id,
    "modules": [
      {
        "module": item.module,
        "model_grade": item.model_grade,
        "review_status": item.review_status,
        "clinician_grade": item.clinician_grade,
      }
      for item in sorted(module_results, key=lambda row: row.module)
    ],
  }
  return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
