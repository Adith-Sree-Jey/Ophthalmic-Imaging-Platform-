from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from database import CalibrationConfig, normalize_module_name


def _softmax(logits: np.ndarray) -> np.ndarray:
  logits = logits - np.max(logits, axis=-1, keepdims=True)
  exp = np.exp(logits)
  denom = np.sum(exp, axis=-1, keepdims=True)
  return exp / np.clip(denom, 1e-12, None)


def _ensure_2d(values: np.ndarray) -> np.ndarray:
  arr = np.asarray(values, dtype=np.float64)
  if arr.ndim == 1:
    arr = arr[:, None]
  return arr


@dataclass
class CalibrationFitResult:
  temperature: float
  baseline_ece: float
  validation_ece: float
  sample_count: int


class TemperatureScaler:
  def __init__(self, module: str, db: Session | None = None):
    self.module = normalize_module_name(module)
    self.db = db
    self.temperature = 1.0
    self.active_config: CalibrationConfig | None = None
    if db is not None:
      self.active_config = (
        db.query(CalibrationConfig)
        .filter(
          CalibrationConfig.module == self.module,
          CalibrationConfig.is_active == 1,  # FIX: was .is_(True) which generates IS 1 on MSSQL
        )
        .order_by(CalibrationConfig.fitted_at.desc())
        .first()
      )
      if self.active_config is not None:
        self.temperature = float(self.active_config.temperature or 1.0)

  def calibrate(self, logits: np.ndarray) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=np.float64)
    temp = max(float(self.temperature), 1e-3)
    return _softmax(logits_arr / temp)

  def calibrate_probabilities(self, probabilities: dict[str, float] | list[float] | np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(
      list(probabilities.values()) if isinstance(probabilities, dict) else probabilities,
      dtype=np.float64,
    )
    if arr.size == 0:
      return arr, False
    clipped = np.clip(arr, 1e-8, 1.0)
    logits = np.log(clipped)
    calibrated = self.calibrate(logits)
    return calibrated, self.active_config is not None

  def fit(self, logits: np.ndarray, labels: np.ndarray, fitted_by: int | None = None) -> CalibrationFitResult:
    logits_arr = _ensure_2d(np.asarray(logits, dtype=np.float64))
    label_arr = np.asarray(labels, dtype=np.int64)
    if logits_arr.shape[0] != label_arr.shape[0]:
      raise ValueError("Logits and labels must have the same number of rows.")
    if logits_arr.shape[0] == 0:
      raise ValueError("Calibration fit requires at least one sample.")

    baseline_probs = _softmax(logits_arr)
    baseline_ece = self.expected_calibration_error(baseline_probs, label_arr)

    best_temp = 1.0
    best_loss = self._negative_log_likelihood(baseline_probs, label_arr)
    for temp in np.linspace(0.5, 5.0, 91):
      calibrated_probs = _softmax(logits_arr / temp)
      loss = self._negative_log_likelihood(calibrated_probs, label_arr)
      if loss < best_loss:
        best_loss = loss
        best_temp = float(temp)

    self.temperature = best_temp
    calibrated_probs = _softmax(logits_arr / best_temp)
    validation_ece = self.expected_calibration_error(calibrated_probs, label_arr)

    if self.db is not None:
      (
        self.db.query(CalibrationConfig)
        .filter(
          CalibrationConfig.module == self.module,
          CalibrationConfig.is_active == 1,  # FIX: was .is_(True) which generates IS 1 on MSSQL
        )
        .update({"is_active": False}, synchronize_session=False)
      )
      config = CalibrationConfig(
        module=self.module,
        temperature=best_temp,
        fitted_at=datetime.utcnow(),
        fitted_by=fitted_by,
        validation_ece=validation_ece,
        baseline_ece=baseline_ece,
        sample_count=int(label_arr.shape[0]),
        is_active=True,
      )
      self.db.add(config)
      self.db.flush()
      self.active_config = config

    return CalibrationFitResult(
      temperature=best_temp,
      baseline_ece=baseline_ece,
      validation_ece=validation_ece,
      sample_count=int(label_arr.shape[0]),
    )

  def expected_calibration_error(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs_arr = _ensure_2d(probs)
    confidences = np.max(probs_arr, axis=1)
    predictions = np.argmax(probs_arr, axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
      lower = bins[i]
      upper = bins[i + 1]
      if i == n_bins - 1:
        mask = (confidences >= lower) & (confidences <= upper)
      else:
        mask = (confidences >= lower) & (confidences < upper)
      if not np.any(mask):
        continue
      bucket_conf = float(np.mean(confidences[mask]))
      bucket_acc = float(np.mean(correctness[mask]))
      bucket_weight = float(np.mean(mask.astype(np.float64)))
      ece += abs(bucket_conf - bucket_acc) * bucket_weight
    return float(ece)

  def reliability_bins(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    probs_arr = _ensure_2d(probs)
    confidences = np.max(probs_arr, axis=1)
    predictions = np.argmax(probs_arr, axis=1)
    correctness = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
      lower = float(bins[i])
      upper = float(bins[i + 1])
      if i == n_bins - 1:
        mask = (confidences >= lower) & (confidences <= upper)
      else:
        mask = (confidences >= lower) & (confidences < upper)
      count = int(np.sum(mask))
      rows.append(
        {
          "bin_start": lower,
          "bin_end": upper,
          "mean_confidence": float(np.mean(confidences[mask])) if count else 0.0,
          "fraction_positive": float(np.mean(correctness[mask])) if count else 0.0,
          "count": count,
        }
      )
    return rows

  @staticmethod
  def _negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    idx = np.arange(labels.shape[0])
    chosen = np.clip(probs[idx, labels], 1e-12, 1.0)
    return float(-np.mean(np.log(chosen)))