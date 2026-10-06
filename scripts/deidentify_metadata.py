"""Create an explicitly configured, non-overwriting de-identified CSV export.

This script does not modify source metadata or image files. It must not be run
until a data owner confirms the identifier and path columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identifier-column", action="append", default=[])
    parser.add_argument("--path-column", action="append", default=[])
    parser.add_argument("--salt-env", default="DEID_HMAC_KEY")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement; without it the script only validates the plan.",
    )
    return parser.parse_args()


def token(value: str, key: bytes) -> str:
    digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:24]


def deidentified_path(value: str, key: bytes) -> str:
    normalized = PurePosixPath(value.replace("\\", "/"))
    suffix = "".join(normalized.suffixes)
    return f"deidentified/{token(value, key)}{suffix}"


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    selected = set(args.identifier_column) | set(args.path_column)

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")
    if source == destination:
        raise ValueError("Output must differ from input; in-place edits are prohibited.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if not selected:
        raise ValueError("Select at least one explicitly approved column.")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(selected - set(fieldnames))
        if missing:
            raise ValueError(f"Selected columns are absent from the CSV: {missing}")
        if not args.execute:
            print("Plan validated; no output written. Re-run with --execute after approval.")
            print(f"Selected identifier columns: {sorted(args.identifier_column)}")
            print(f"Selected path columns: {sorted(args.path_column)}")
            return 0

        secret = os.getenv(args.salt_env, "")
        if len(secret) < 32:
            raise ValueError(
                f"{args.salt_env} must contain an authorized secret of at least 32 characters."
            )
        key = secret.encode("utf-8")
        rows = list(reader)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for column in args.identifier_column:
                if row[column]:
                    row[column] = f"subject_{token(row[column], key)}"
            for column in args.path_column:
                if row[column]:
                    row[column] = deidentified_path(row[column], key)
            writer.writerow(row)

    print(f"Wrote de-identified metadata export with {len(rows)} rows.")
    print("No source metadata or image files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

