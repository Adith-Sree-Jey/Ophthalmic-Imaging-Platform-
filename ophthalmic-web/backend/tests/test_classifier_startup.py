from pathlib import Path

import pytest

from classifier import ClassifierService


def test_missing_configured_checkpoint_fails_clearly(monkeypatch, tmp_path: Path):
    missing = tmp_path / "best_qwk_model.pth"
    monkeypatch.setenv("CATARACT_CHECKPOINT_PATH", str(missing))

    service = ClassifierService()

    assert service.availability_error is not None
    assert "CATARACT_CHECKPOINT_PATH" in service.availability_error
    assert "fake weights" in service.availability_error
    with pytest.raises(RuntimeError, match="CATARACT_CHECKPOINT_PATH"):
        service._ensure_ready()

