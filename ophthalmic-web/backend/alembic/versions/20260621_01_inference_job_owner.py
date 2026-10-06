"""Add ownership to inference jobs.

Revision ID: 20260621_01
Revises: 20260505_01
Create Date: 2026-06-21
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260621_01"
down_revision = "20260505_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inference_jobs", sa.Column("created_by", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_inference_jobs_created_by_users",
        "inference_jobs",
        "users",
        ["created_by"],
        ["id"],
    )
    op.create_index("ix_inference_jobs_created_by", "inference_jobs", ["created_by"])


def downgrade() -> None:
    op.drop_index("ix_inference_jobs_created_by", table_name="inference_jobs")
    op.drop_constraint(
        "fk_inference_jobs_created_by_users",
        "inference_jobs",
        type_="foreignkey",
    )
    op.drop_column("inference_jobs", "created_by")
