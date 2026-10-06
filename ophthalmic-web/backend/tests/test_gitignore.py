"""Guard test for the repository .gitignore (Finding 5).

A previous edit wrote the trailing patterns in UTF-16, leaving 52 NUL bytes in
the file. The leading '*' followed by a NUL made git treat the line as a
match-everything wildcard, silently un-tracking the entire repository.

This test fails if any NUL bytes reappear or if the file stops being valid UTF-8.
"""
from __future__ import annotations

from pathlib import Path

# backend/tests/ -> backend -> ophthalmic-web -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
GITIGNORE = REPO_ROOT / ".gitignore"


def test_gitignore_has_no_null_bytes():
    assert GITIGNORE.exists(), ".gitignore is missing"
    raw = GITIGNORE.read_bytes()
    assert b"\x00" not in raw, ".gitignore contains NUL bytes (UTF-16 corruption)"


def test_gitignore_is_valid_utf8_and_keeps_pth_rule():
    assert GITIGNORE.exists(), ".gitignore is missing"
    text = GITIGNORE.read_text(encoding="utf-8")  # raises if not valid UTF-8
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    assert lines.count("*.pth") == 1
    # No line may be a bare '*' (match-everything) wildcard.
    assert "*" not in lines
