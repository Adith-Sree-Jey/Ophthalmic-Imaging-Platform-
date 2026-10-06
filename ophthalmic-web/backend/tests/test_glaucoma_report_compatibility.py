from __future__ import annotations

import glaucoma_service
import main


def test_file_report_path_supports_legacy_inference(monkeypatch):
    monkeypatch.setattr(glaucoma_service, "predict_glaucoma_file", lambda image_path: {
        "prediction": "normal",
        "predicted_class": "normal",
        "confidence": 0.91,
        "probabilities": {"normal": 0.91, "glaucoma": 0.09},
        "model_mode": "legacy",
    })

    class FakeReport:
        def to_dict(self):
            return {"findings": "Legacy report generated"}

    monkeypatch.setattr(
        glaucoma_service,
        "build_glaucoma_report",
        lambda _result, backend="local": FakeReport(),
    )
    result = glaucoma_service.generate_report_from_file(
        image_path="legacy-image.png",
        patient_info={"patient_id": "P-1", "eye_side": "OD"},
    )
    assert result["model_mode"] == "legacy"
    assert result["report"]["findings"] == "Legacy report generated"


def test_glaucoma_prediction_payload_keeps_gradcam_and_overlay(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "predict_glaucoma_file", lambda _path: {
        "predicted_class": "glaucoma",
        "confidence": 0.88,
        "probabilities": {"glaucoma": 0.88, "normal": 0.12},
        "gradcam_b64": "gradcam-data",
        "overlay_b64": "overlay-data",
    })
    monkeypatch.setattr(
        main,
        "_apply_probability_calibration",
        lambda _module, probabilities: (probabilities, False),
    )
    monkeypatch.setattr(
        main,
        "_evaluate_review_decision",
        lambda _module, _payload: (False, None, "routine"),
    )

    def run_inline(_job_id, work, _cleanup_paths):
        captured.update(work())

    monkeypatch.setattr(main, "_run_background_job", run_inline)
    main.run_glaucoma_prediction_job(
        "job-images",
        "input.png",
        "CASE-1",
        "Patient",
        "",
        "OD",
        "doctor_a",
        1,
    )
    assert captured["gradcam_b64"] == "gradcam-data"
    assert captured["overlay_b64"] == "overlay-data"
