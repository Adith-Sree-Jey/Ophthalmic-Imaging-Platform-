from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import CalibrationConfig, Case, CaseModuleResult, normalize_module_name


def _parse_date(value: str | None, fallback: datetime) -> datetime:
  if not value:
    return fallback
  return datetime.fromisoformat(value)


def performance_overview(db: Session, date_from: str | None = None, date_to: str | None = None, module: str | None = None) -> list[dict[str, Any]]:
  end_date = _parse_date(date_to, datetime.utcnow())
  start_date = _parse_date(date_from, end_date - timedelta(days=30))
  query = db.query(CaseModuleResult).join(Case, Case.id == CaseModuleResult.case_id).filter(Case.created_at >= start_date, Case.created_at <= end_date)
  if module:
    query = query.filter(CaseModuleResult.module == normalize_module_name(module))
  rows = query.all()
  output = []
  modules = sorted({row.module for row in rows} or {"cataract", "glaucoma", "retina"})
  for module_name in modules:
    module_rows = [row for row in rows if row.module == module_name]
    reviewed = [row for row in module_rows if row.reviewed_at]
    overrides = [row for row in module_rows if row.review_status == "overridden"]
    calibration = (
      db.query(CalibrationConfig)
      .filter(CalibrationConfig.module == module_name, CalibrationConfig.is_active == 1)  # FIX: was .is_(True)
      .order_by(CalibrationConfig.fitted_at.desc())
      .first()
    )
    output.append(
      {
        "module": module_name,
        "accuracy": _accuracy(module_rows),
        "ece": float(calibration.validation_ece or 0.0) if calibration else 0.0,
        "flag_rate": _ratio(sum(1 for row in module_rows if row.flagged_for_review), len(module_rows)),
        "override_rate": _ratio(len(overrides), len(module_rows)),
        "avg_review_time": _avg_review_time_hours(reviewed),
      }
    )
  return output


def override_analysis(db: Session, date_from: str | None = None, date_to: str | None = None) -> list[dict[str, Any]]:
  end_date = _parse_date(date_to, datetime.utcnow())
  start_date = _parse_date(date_from, end_date - timedelta(days=30))
  rows = (
    db.query(CaseModuleResult)
    .join(Case, Case.id == CaseModuleResult.case_id)
    .filter(Case.created_at >= start_date, Case.created_at <= end_date, CaseModuleResult.review_status == "overridden")
    .all()
  )
  bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
  payload = []
  for row in rows:
    confidence = float(row.model_confidence or 0.0)
    band = next((f"{low:.2f}-{high if high < 1 else 1.0:.2f}" for low, high in bands if low <= confidence < high), "0.00-1.00")
    payload.append(
      {
        "module": row.module,
        "model_grade": row.model_grade,
        "clinician_grade": row.clinician_grade,
        "confidence_band": band,
      }
    )
  return payload


def calibration_drift(db: Session) -> list[dict[str, Any]]:
  rows = (
    db.query(
      CalibrationConfig.module,
      func.datepart("iso_week", CalibrationConfig.fitted_at),
      func.avg(CalibrationConfig.validation_ece),
      func.max(CalibrationConfig.validation_ece),
    )
    .group_by(CalibrationConfig.module, func.datepart("iso_week", CalibrationConfig.fitted_at))
    .order_by(CalibrationConfig.module)
    .all()
  )
  return [
    {
      "module": module,
      "week": int(week),
      "ece": float(avg_ece or 0.0),
      "alert": bool((max_ece or 0.0) > 0.15),
    }
    for module, week, avg_ece, max_ece in rows
  ]


def reviewer_agreement(db: Session) -> list[dict[str, Any]]:
  rows = (
    db.query(CaseModuleResult.case_id, CaseModuleResult.module, CaseModuleResult.reviewer_id, CaseModuleResult.clinician_grade)
    .filter(CaseModuleResult.review_status == "overridden", CaseModuleResult.reviewer_id.isnot(None))
    .all()
  )
  grouped: dict[tuple[str, str], list[tuple[int, str | None]]] = {}
  for case_id, module, reviewer_id, clinician_grade in rows:
    grouped.setdefault((case_id, module), []).append((reviewer_id, clinician_grade))
  payload = []
  for (case_id, module), entries in grouped.items():
    if len(entries) < 2:
      continue
    unique_grades = {grade for _, grade in entries if grade}
    agreement = 1.0 if len(unique_grades) <= 1 else 0.0
    payload.append({"case_id": case_id, "module": module, "cohen_kappa": agreement})
  return payload


def throughput_metrics(db: Session, date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
  end_date = _parse_date(date_to, datetime.utcnow())
  start_date = _parse_date(date_from, end_date - timedelta(days=30))
  reviewed_rows = (
    db.query(CaseModuleResult)
    .join(Case, Case.id == CaseModuleResult.case_id)
    .filter(Case.created_at >= start_date, Case.created_at <= end_date, CaseModuleResult.reviewed_at.isnot(None))
    .all()
  )
  per_reviewer: dict[int, int] = {}
  review_durations = []
  queue_ages = []
  for row in reviewed_rows:
    if row.reviewer_id is not None:
      per_reviewer[row.reviewer_id] = per_reviewer.get(row.reviewer_id, 0) + 1
    if row.reviewed_at and row.case and row.case.created_at:
      review_durations.append((row.reviewed_at - row.case.created_at).total_seconds() / 3600.0)
      queue_ages.append((row.reviewed_at - row.case.created_at).days)
  return {
    "cases_reviewed_per_reviewer": [{"reviewer_id": key, "count": value} for key, value in per_reviewer.items()],
    "cases_reviewed_per_day": _cases_reviewed_per_day(reviewed_rows),
    "avg_time_to_first_review": sum(review_durations) / len(review_durations) if review_durations else 0.0,
    "queue_age_distribution": queue_ages,
  }


def _cases_reviewed_per_day(rows: list[CaseModuleResult]) -> list[dict[str, Any]]:
  by_day: dict[str, int] = {}
  for row in rows:
    if row.reviewed_at is None:
      continue
    day = row.reviewed_at.date().isoformat()
    by_day[day] = by_day.get(day, 0) + 1
  return [{"date": key, "count": value} for key, value in sorted(by_day.items())]


def _accuracy(rows: list[CaseModuleResult]) -> float:
  comparable = [row for row in rows if row.clinician_grade]
  if not comparable:
    return 0.0
  matches = sum(1 for row in comparable if row.model_grade == row.clinician_grade)
  return matches / len(comparable)


def _ratio(numerator: int, denominator: int) -> float:
  return numerator / denominator if denominator else 0.0


def _avg_review_time_hours(rows: list[CaseModuleResult]) -> float:
  durations = [
    (row.reviewed_at - row.case.created_at).total_seconds() / 3600.0
    for row in rows
    if row.reviewed_at and row.case and row.case.created_at
  ]
  return sum(durations) / len(durations) if durations else 0.0
