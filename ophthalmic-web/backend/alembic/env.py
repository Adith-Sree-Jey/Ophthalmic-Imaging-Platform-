from __future__ import annotations

import os
from logging.config import fileConfig
from urllib.parse import quote_plus

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

from database import Base

# ---------------------------------------------------------------------------
# Build the connection string from the SAME environment variables the app uses
# (see database.py). This guarantees migrations run against the application
# database instead of a hardcoded target, and keeps credentials out of source.
# ---------------------------------------------------------------------------
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(ENV_PATH)

SERVER = os.getenv("MSSQL_SERVER", r"localhost\SQLEXPRESS")
DATABASE = os.getenv("MSSQL_DATABASE")
DRIVER = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
USERNAME = os.getenv("MSSQL_USERNAME", "")
PASSWORD = os.getenv("MSSQL_PASSWORD", "")
ENCRYPT = os.getenv("MSSQL_ENCRYPT", "no")

if not DATABASE:
    raise RuntimeError(f"MSSQL_DATABASE must be set in {ENV_PATH}.")

if USERNAME:
    _auth = f"UID={USERNAME};PWD={PASSWORD}"
else:
    _auth = "Trusted_Connection=yes"

_odbc_connect = quote_plus(
    ";".join(
        [
            f"DRIVER={{{DRIVER}}}",
            f"SERVER={SERVER}",
            f"DATABASE={DATABASE}",
            _auth,
            f"Encrypt={ENCRYPT}",
            "TrustServerCertificate=yes",
        ]
    )
)

connection_url = f"mssql+pyodbc:///?odbc_connect={_odbc_connect}"

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=connection_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(connection_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
