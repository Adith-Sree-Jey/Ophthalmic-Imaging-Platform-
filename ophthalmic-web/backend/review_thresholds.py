from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from database import ReviewThresholdConfig, log_audit_event, normalize_confidence, normalize_module_name


PRIORITY_ORDER = {"routine": 0, "urgent": 1, "critical": 2}


@dataclass
class ReviewDecision:
  flagged: bool
  reasons: list[str]
  priority: str


class ReviewThresholdEngine:
  def __init__(self, db: Session):
    self.db = db

  def evaluate(self, module: str, result: dict) -> ReviewDecision:
    module_name = normalize_module_name(module)
    thresholds = self.get_thresholds(module_name)
    reasons: list[str] = []
    priority = "routine"

    def bump_priority(candidate: str) -> None:
      nonlocal priority
      if PRIORITY_ORDER.get(candidate, 0) > PRIORITY_ORDER.get(priority, 0):
        priority = candidate

    confidence = normalize_confidence(result.get("confidence"))

    if module_name == "cataract":
      if confidence < thresholds.get("confidence", {}).get("threshold_value", 0.75):
        item = thresholds["confidence"]
        reasons.append(item["reason_template"])
        bump_priority(item["priority_level"])
      if (result.get("grade") or "").upper() == "NS4":
        item = thresholds.get("ns4_grade")
        if item:
          reasons.append(item["reason_template"])
          bump_priority(item["priority_level"])
      disagreement = float(result.get("ordinal_head_disagreement") or 0.0)
      if disagreement > thresholds.get("ordinal_head_disagreement", {}).get("threshold_value", 0.2):
        item = thresholds["ordinal_head_disagreement"]
        reasons.append(item["reason_template"])
        bump_priority(item["priority_level"])
      modality_count = int(result.get("modality_count") or len(result.get("attention", {}) or {}) or 1)
      if modality_count < thresholds.get("minimum_modality_count", {}).get("threshold_value", 2.0):
        item = thresholds["minimum_modality_count"]
        reasons.append(item["reason_template"])
        bump_priority(item["priority_level"])

    elif module_name == "glaucoma":
      if confidence < thresholds.get("confidence", {}).get("threshold_value", 0.8):
        item = thresholds["confidence"]
        reasons.append(item["reason_template"])
        bump_priority(item["priority_level"])
      if str(result.get("predicted_class") or "").strip().lower() == "glaucoma":
        item = thresholds.get("predicted_glaucoma")
        if item:
          reasons.append(item["reason_template"])
          bump_priority(item["priority_level"])
      cup_to_disc_ratio = float(result.get("cup_to_disc_ratio") or 0.0)
      if cup_to_disc_ratio > thresholds.get("cup_to_disc_ratio", {}).get("threshold_value", 0.7):
        item = thresholds["cup_to_disc_ratio"]
        reasons.append(item["reason_template"])
        bump_priority(item["priority_level"])
      vessel_quality = str(result.get("vessel_feature_quality") or "").strip().lower()
      if vessel_quality == "poor":
        item = thresholds.get("poor_vessel_quality")
        if item:
          reasons.append(item["reason_template"])
          bump_priority(item["priority_level"])

    elif module_name == "retina":
      vessel_coverage = float(result.get("vessel_coverage") or 0.0)
      if vessel_coverage < thresholds.get("vessel_coverage", {}).get("threshold_value", 0.3):
        item = thresholds["vessel_coverage"]
        reasons.append(item["reason_template"])
        bump_priority(item["priority_level"])
      image_quality_score = float(result.get("image_quality_score") or 0.0)
      if image_quality_score < thresholds.get("image_quality_score", {}).get("threshold_value", 0.6):
        item = thresholds["image_quality_score"]
        reasons.append(item["reason_template"])
        bump_priority(item["priority_level"])

    return ReviewDecision(
      flagged=bool(reasons),
      reasons=reasons,
      priority=priority,
    )

  def get_thresholds(self, module: str) -> dict[str, dict[str, Any]]:
    rows = (
      self.db.query(ReviewThresholdConfig)
      .filter(
        ReviewThresholdConfig.module == normalize_module_name(module),
        ReviewThresholdConfig.is_active == 1,
      )
      .all()
    )
    return {
      row.threshold_key: {
        "id": row.id,
        "threshold_key": row.threshold_key,
        "threshold_value": row.threshold_value,
        "priority_level": row.priority_level,
        "reason_template": row.reason_template,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "updated_by": row.updated_by,
      }
      for row in rows
    }

  def update_thresholds(self, module: str, thresholds: dict, updated_by: int):
    module_name = normalize_module_name(module)
    threshold_key = thresholds["threshold_key"]
    existing = (
      self.db.query(ReviewThresholdConfig)
      .filter(
        ReviewThresholdConfig.module == module_name,
        ReviewThresholdConfig.threshold_key == threshold_key,
        ReviewThresholdConfig.is_active == 1,
      )
      .first()
    )

    before_state = existing
    if existing is not None:
      existing.is_active = False
      self.db.flush()

    row = ReviewThresholdConfig(
      module=module_name,
      threshold_key=threshold_key,
      threshold_value=float(thresholds["threshold_value"]),
      priority_level=str(thresholds.get("priority_level") or "routine"),
      reason_template=str(thresholds.get("reason_template") or threshold_key),
      is_active=True,
      updated_by=updated_by,
    )
    self.db.add(row)
    self.db.flush()
    log_audit_event(
      self.db,
      actor_id=updated_by,
      action="threshold.updated",
      before_state=before_state,
      after_state=row,
    )
    return row
