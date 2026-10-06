import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if "ADMIN_USERNAME" not in os.environ or "ADMIN_PASSWORD" not in os.environ:
    load_dotenv(BACKEND_DIR / ".env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth import hash_password
from database import SessionLocal, User


def _is_placeholder(value: str) -> bool:
    return value in {"your_admin_username", "your_strong_password_here"} or value.startswith("REPLACE_ME")


def main():
    username = os.environ.get("ADMIN_USERNAME")
    password = os.environ.get("ADMIN_PASSWORD")

    if not username or not password or _is_placeholder(username) or _is_placeholder(password):
        sys.exit(
            "FATAL: Set ADMIN_USERNAME and ADMIN_PASSWORD in your .env file "
            "before running this script."
        )

    if len(password) < 12:
        sys.exit("FATAL: ADMIN_PASSWORD must be at least 12 characters.")

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            print(f"User '{username}' already exists. Skipping.")
            return
        user = User(
            username=username,
            hashed_password=hash_password(password),
            role="admin",
            is_active=True,
        )
        db.add(user)
        db.commit()
        print(f"Admin user '{username}' created successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
