"""Portable path resolution for legacy multimodal metadata manifests."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "Data"
DEFAULT_CROPPED_DATA_ROOT = REPO_ROOT / "Data_Cropped"


def _configured_root(env_name: str, default: Path) -> Path:
    configured = Path(os.getenv(env_name, str(default))).expanduser()
    return configured if configured.is_absolute() else REPO_ROOT / configured


def resolve_metadata_image_path(raw_path: str, csv_path: str | Path) -> Path:
    """Resolve a metadata image path without rewriting the source CSV.

    Existing valid paths are preserved. Legacy absolute Windows paths are
    rebased at their ``Data`` or ``Data_Cropped`` segment onto configurable
    roots. Relative paths are resolved first beside the CSV, then from the
    repository root.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Metadata image path is empty or non-string.")

    expanded = Path(os.path.expandvars(raw_path)).expanduser()
    if expanded.exists():
        return expanded

    # PurePosixPath after slash normalization handles legacy Windows paths on
    # every host without depending on the host's path flavour.
    normalized_parts = PurePosixPath(raw_path.replace("\\", "/")).parts
    lowered = [part.lower() for part in normalized_parts]
    for marker, env_name, default in (
        ("data_cropped", "OPHTHALMIC_CROPPED_DATA_ROOT", DEFAULT_CROPPED_DATA_ROOT),
        ("data", "OPHTHALMIC_CATARACT_DATA_ROOT", DEFAULT_DATA_ROOT),
    ):
        if marker in lowered:
            marker_index = lowered.index(marker)
            root = _configured_root(env_name, default)
            return root.joinpath(*normalized_parts[marker_index + 1 :])

    if not expanded.is_absolute():
        beside_csv = Path(csv_path).expanduser().resolve().parent / expanded
        if beside_csv.exists():
            return beside_csv
        return REPO_ROOT / expanded

    # Keep an unrecognized absolute path intact so the eventual load error is
    # honest; do not guess how an unknown dataset layout should be mapped.
    return expanded
