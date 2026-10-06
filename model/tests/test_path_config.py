from pathlib import Path

from model.path_config import resolve_metadata_image_path


def test_legacy_metadata_paths_rebase_to_configured_roots(tmp_path, monkeypatch):
    data_root = tmp_path / "clinical-data"
    cropped_root = tmp_path / "clinical-data-cropped"
    monkeypatch.setenv("OPHTHALMIC_CATARACT_DATA_ROOT", str(data_root))
    monkeypatch.setenv("OPHTHALMIC_CROPPED_DATA_ROOT", str(cropped_root))
    csv_path = tmp_path / "metadata" / "train.csv"

    regular = resolve_metadata_image_path(
        r"C:\legacy\Data\NS1\anterior_segment\sample.jpg", csv_path
    )
    cropped = resolve_metadata_image_path(
        r"C:\legacy\Data_Cropped\NS1\red_glow\sample.jpg", csv_path
    )

    assert regular == data_root / "NS1" / "anterior_segment" / "sample.jpg"
    assert cropped == cropped_root / "NS1" / "red_glow" / "sample.jpg"


def test_relative_metadata_path_defaults_beside_csv(tmp_path):
    csv_path = tmp_path / "metadata" / "train.csv"
    image_path = csv_path.parent / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"test")

    resolved = resolve_metadata_image_path("images/sample.jpg", csv_path)

    assert resolved == image_path
