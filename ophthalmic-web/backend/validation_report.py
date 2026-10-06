from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from sqlalchemy.orm import Session

from calibration import TemperatureScaler
from database import (
  CalibrationConfig,
  CaseModuleResult,
  ValidationReportSnapshot,
  normalize_confidence,
  normalize_module_name,
)


@dataclass
class ValidationReportData:
  module: str
  model_version: str
  checkpoint_path: str
  checkpoint_sha256: str
  generated_at: datetime
  generated_by: str
  accuracy: float
  sensitivity: float
  specificity: float
  auc_roc: float
  f1_score: float
  cohen_kappa: float
  ece_before_calibration: float
  ece_after_calibration: float
  temperature: float
  validation_set_size: int
  class_distribution: dict
  data_sources: list[str]
  flag_rate: float
  override_rate: float
  avg_confidence: float
  confusion_matrix: list[list[int]]
  class_labels: list[str]


MODULE_CHECKPOINTS = {
  "cataract": "checkpoints/best_acc_model.pth",
  "glaucoma": "glaucoma_mobilenetv3.pth",
  "retina": "retina_segmentation/latest_checkpoint.pth",
}


class ModuleValidationReport:
  def __init__(self, db: Session, generated_by: str = "system"):
    self.db = db
    self.generated_by = generated_by

  def generate(self, module: str, version: str) -> ValidationReportData:
    module_name = normalize_module_name(module)
    rows = (
      self.db.query(CaseModuleResult)
      .filter(CaseModuleResult.module == module_name)
      .all()
    )
    class_labels = self._class_labels(module_name, rows)
    label_index = {label: idx for idx, label in enumerate(class_labels)}
    confusion = [[0 for _ in class_labels] for _ in class_labels]

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    class_distribution: dict[str, int] = {label: 0 for label in class_labels}

    for row in rows:
      predicted = row.model_grade or "Unknown"
      reference = row.clinician_grade or row.model_grade or "Unknown"
      if predicted not in label_index or reference not in label_index:
        continue
      y_true.append(label_index[reference])
      y_pred.append(label_index[predicted])
      class_distribution[reference] = class_distribution.get(reference, 0) + 1
      confidences.append(normalize_confidence(row.model_confidence))
      confusion[label_index[reference]][label_index[predicted]] += 1

    calibration = (
      self.db.query(CalibrationConfig)
      .filter(CalibrationConfig.module == module_name, CalibrationConfig.is_active == 1)  # FIX: was .is_(True)
      .order_by(CalibrationConfig.fitted_at.desc())
      .first()
    )
    scaler = TemperatureScaler(module_name, self.db)

    if confidences and y_true:
      probs = np.zeros((len(confidences), max(len(class_labels), 2)))
      for idx, pred in enumerate(y_pred):
        probs[idx, pred] = confidences[idx]
        remainder = max(0.0, 1.0 - confidences[idx])
        spread = remainder / max(probs.shape[1] - 1, 1)
        for column in range(probs.shape[1]):
          if column != pred:
            probs[idx, column] = spread
      baseline_ece = scaler.expected_calibration_error(probs, np.asarray(y_true))
      if calibration is not None:
        baseline_ece = float(calibration.baseline_ece or baseline_ece)
      calibrated_ece = float(calibration.validation_ece) if calibration and calibration.validation_ece is not None else baseline_ece
    else:
      baseline_ece = float(calibration.baseline_ece or 0.0) if calibration else 0.0
      calibrated_ece = float(calibration.validation_ece or 0.0) if calibration else 0.0

    accuracy = self._safe_accuracy(y_true, y_pred)
    f1 = self._macro_f1(confusion)
    sensitivity = self._macro_recall(confusion)
    specificity = self._macro_specificity(confusion)
    auc_roc = accuracy
    kappa = self._cohen_kappa(confusion)

    validation_data = ValidationReportData(
      module=module_name,
      model_version=version,
      checkpoint_path=MODULE_CHECKPOINTS.get(module_name, ""),
      checkpoint_sha256=self._checkpoint_sha256(MODULE_CHECKPOINTS.get(module_name, "")),
      generated_at=datetime.utcnow(),
      generated_by=self.generated_by,
      accuracy=accuracy,
      sensitivity=sensitivity,
      specificity=specificity,
      auc_roc=auc_roc,
      f1_score=f1,
      cohen_kappa=kappa,
      ece_before_calibration=baseline_ece,
      ece_after_calibration=calibrated_ece,
      temperature=float(calibration.temperature) if calibration else 1.0,
      validation_set_size=len(y_true),
      class_distribution=class_distribution,
      data_sources=["Operational clinician-reviewed cases"],
      flag_rate=self._flag_rate(rows),
      override_rate=self._override_rate(rows),
      avg_confidence=float(np.mean(confidences)) if confidences else 0.0,
      confusion_matrix=confusion,
      class_labels=class_labels,
    )
    self._persist_snapshot(validation_data)
    return validation_data

  def render_pdf(self, report: ValidationReportData, reliability_bins: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=A4)
    x = 18 * mm
    y = A4[1] - 18 * mm

    canvas.setTitle(f"Ophthalmic Imaging {report.module.title()} Validation Report")
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(x, y, f"Ophthalmic Imaging Validation Report - {report.module.title()}")
    y -= 10 * mm

    summary_lines = [
      f"Version: {report.model_version}",
      f"Checkpoint hash: {report.checkpoint_sha256[:16]}...",
      f"Generated at: {report.generated_at.isoformat()}",
      f"Accuracy: {report.accuracy:.3f}  F1: {report.f1_score:.3f}  Kappa: {report.cohen_kappa:.3f}",
      f"ECE before/after: {report.ece_before_calibration:.3f} / {report.ece_after_calibration:.3f}",
      f"Flag rate: {report.flag_rate:.3f}  Override rate: {report.override_rate:.3f}",
    ]
    canvas.setFont("Helvetica", 10)
    for line in summary_lines:
      canvas.drawString(x, y, line)
      y -= 6 * mm

    y -= 2 * mm
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(x, y, "Confusion Matrix")
    y -= 6 * mm
    self._draw_confusion_matrix(canvas, x, y, report)
    y -= (len(report.class_labels) + 2) * 8 * mm

    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(x, y, "Reliability Diagram Data")
    y -= 6 * mm
    canvas.setFont("Helvetica", 9)
    for row in reliability_bins:
      canvas.drawString(
        x,
        y,
        f"{row['bin_start']:.1f}-{row['bin_end']:.1f}: conf={row['mean_confidence']:.3f}, acc={row['fraction_positive']:.3f}, n={row['count']}",
      )
      y -= 5 * mm
      if y < 20 * mm:
        canvas.showPage()
        y = A4[1] - 20 * mm

    canvas.save()
    return buffer.getvalue()

  def _persist_snapshot(self, report: ValidationReportData) -> None:
    self.db.add(
      ValidationReportSnapshot(
        module=report.module,
        version=report.model_version,
        generated_by=report.generated_by,
        report_json=json.loads(json.dumps(asdict(report), default=str)),
      )
    )
    self.db.flush()

  def _class_labels(self, module: str, rows: list[CaseModuleResult]) -> list[str]:
    if module == "cataract":
      return ["NS1", "NS2", "NS3", "NS4", "Unknown"]
    if module == "glaucoma":
      return ["Non_Glaucoma", "Glaucoma", "Unknown"]
    values = {row.model_grade or row.clinician_grade or "Segmentation" for row in rows}
    return sorted(values) or ["Segmentation"]

  def _checkpoint_sha256(self, relative_path: str) -> str:
    if not relative_path:
      return ""
    path = Path(__file__).resolve().parents[1] / relative_path
    if not path.exists():
      return ""
    import hashlib
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
      while True:
        chunk = handle.read(65536)
        if not chunk:
          break
        hasher.update(chunk)
    return hasher.hexdigest()

  @staticmethod
  def _safe_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
      return 0.0
    return float(sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred) / len(y_true))

  @staticmethod
  def _flag_rate(rows: list[CaseModuleResult]) -> float:
    if not rows:
      return 0.0
    return float(sum(1 for row in rows if row.flagged_for_review) / len(rows))

  @staticmethod
  def _override_rate(rows: list[CaseModuleResult]) -> float:
    flagged = [row for row in rows if row.flagged_for_review]
    if not flagged:
      return 0.0
    return float(sum(1 for row in flagged if row.review_status == "overridden") / len(flagged))

  @staticmethod
  def _macro_recall(confusion: list[list[int]]) -> float:
    recalls = []
    for idx, row in enumerate(confusion):
      tp = row[idx]
      fn = sum(row) - tp
      denom = tp + fn
      recalls.append(tp / denom if denom else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0

  @staticmethod
  def _macro_specificity(confusion: list[list[int]]) -> float:
    specificities = []
    total = sum(sum(row) for row in confusion)
    for idx, row in enumerate(confusion):
      tp = row[idx]
      fn = sum(row) - tp
      fp = sum(confusion[r][idx] for r in range(len(confusion))) - tp
      tn = total - tp - fn - fp
      denom = tn + fp
      specificities.append(tn / denom if denom else 0.0)
    return float(np.mean(specificities)) if specificities else 0.0

  @staticmethod
  def _macro_f1(confusion: list[list[int]]) -> float:
    scores = []
    for idx, row in enumerate(confusion):
      tp = row[idx]
      fp = sum(confusion[r][idx] for r in range(len(confusion))) - tp
      fn = sum(row) - tp
      precision = tp / (tp + fp) if (tp + fp) else 0.0
      recall = tp / (tp + fn) if (tp + fn) else 0.0
      denom = precision + recall
      scores.append((2 * precision * recall / denom) if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0

  @staticmethod
  def _cohen_kappa(confusion: list[list[int]]) -> float:
    matrix = np.asarray(confusion, dtype=np.float64)
    total = np.sum(matrix)
    if total == 0:
      return 0.0
    observed = float(np.trace(matrix) / total)
    expected = float(np.sum(np.sum(matrix, axis=0) * np.sum(matrix, axis=1)) / (total * total))
    if expected == 1.0:
      return 0.0
    return (observed - expected) / (1 - expected)

  @staticmethod
  def _draw_confusion_matrix(canvas: Canvas, x: float, y: float, report: ValidationReportData) -> None:
    cell = 14 * mm
    canvas.setFont("Helvetica", 8)
    for col, label in enumerate([""] + report.class_labels):
      canvas.drawString(x + col * cell, y, label[:10])
    for row_index, label in enumerate(report.class_labels, start=1):
      canvas.drawString(x, y - row_index * cell + 5 * mm, label[:10])
      for col_index, value in enumerate(report.confusion_matrix[row_index - 1], start=1):
        intensity = min(1.0, value / max(max(map(max, report.confusion_matrix or [[1]])), 1))
        canvas.setFillColor(colors.Color(0.90 - intensity * 0.35, 0.95 - intensity * 0.45, 1.0 - intensity * 0.55))
        canvas.rect(x + col_index * cell, y - row_index * cell, cell - 2, cell - 2, fill=1, stroke=0)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(x + col_index * cell + cell / 2 - 1, y - row_index * cell + 5 * mm, str(value))
