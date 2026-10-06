"""Add clinician review workflow, calibration, thresholds, and validation tables.

Revision ID: 20260505_01
Revises:
Create Date: 2026-05-05 22:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260505_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("patient_id", sa.Integer(), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("GETUTCDATE()"), nullable=False),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("module_types", sa.JSON(), nullable=False),
        sa.Column("assigned_reviewer", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("priority", sa.String(length=20), nullable=False, server_default="routine"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("report_status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("report_version_hash", sa.String(length=128), nullable=True),
        sa.Column("report_rejection_reason", sa.Text(), nullable=True),
        sa.Column("report_approved_at", sa.DateTime(), nullable=True),
        sa.Column("report_approved_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_cases_patient_id", "cases", ["patient_id"])
    op.create_index("ix_cases_status", "cases", ["status"])
    op.create_index("ix_cases_assigned_reviewer", "cases", ["assigned_reviewer"])
    op.create_index("ix_cases_priority", "cases", ["priority"])

    op.create_table(
        "case_module_results",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("module", sa.String(length=30), nullable=False),
        sa.Column("job_id", sa.String(length=36), sa.ForeignKey("inference_jobs.id"), nullable=False),
        sa.Column("model_prediction", sa.JSON(), nullable=False),
        sa.Column("model_confidence", sa.Float(), nullable=False),
        sa.Column("model_grade", sa.String(length=100), nullable=True),
        sa.Column("review_status", sa.String(length=20), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("clinician_grade", sa.String(length=100), nullable=True),
        sa.Column("clinician_notes", sa.Text(), nullable=True),
        sa.Column("flagged_for_review", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("flag_reason", sa.Text(), nullable=True),
    )
    op.create_index("ix_case_module_results_case_id", "case_module_results", ["case_id"])
    op.create_index("ix_case_module_results_module", "case_module_results", ["module"])
    op.create_index("ix_case_module_results_reviewer_id", "case_module_results", ["reviewer_id"])
    op.create_index("ix_case_module_results_case_module", "case_module_results", ["case_id", "module"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.text("GETUTCDATE()"), nullable=False),
        sa.Column("actor_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("cases.id"), nullable=True),
        sa.Column("module_result_id", sa.String(length=36), sa.ForeignKey("case_module_results.id"), nullable=True),
        sa.Column("before_state", sa.JSON(), nullable=True),
        sa.Column("after_state", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(length=100), nullable=True),
        sa.Column("session_id", sa.String(length=100), nullable=True),
    )
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index("ix_audit_log_case_id", "audit_log", ["case_id"])

    op.create_table(
        "calibration_configs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("module", sa.String(length=30), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("fitted_at", sa.DateTime(), server_default=sa.text("GETUTCDATE()"), nullable=False),
        sa.Column("fitted_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("validation_ece", sa.Float(), nullable=True),
        sa.Column("baseline_ece", sa.Float(), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
    )
    op.create_index("ix_calibration_configs_module", "calibration_configs", ["module"])

    op.create_table(
        "review_threshold_configs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("module", sa.String(length=30), nullable=False),
        sa.Column("threshold_key", sa.String(length=100), nullable=False),
        sa.Column("threshold_value", sa.Float(), nullable=False),
        sa.Column("priority_level", sa.String(length=20), nullable=False),
        sa.Column("reason_template", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("GETUTCDATE()"), nullable=False),
        sa.Column("updated_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_review_threshold_configs_module", "review_threshold_configs", ["module"])
    op.create_index("ix_review_threshold_configs_threshold_key", "review_threshold_configs", ["threshold_key"])

    op.create_table(
        "validation_reports",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("module", sa.String(length=30), nullable=False),
        sa.Column("version", sa.String(length=100), nullable=False),
        sa.Column("generated_at", sa.DateTime(), server_default=sa.text("GETUTCDATE()"), nullable=False),
        sa.Column("generated_by", sa.String(length=100), nullable=True),
        sa.Column("report_json", sa.JSON(), nullable=False),
    )
    op.create_index("ix_validation_reports_module", "validation_reports", ["module"])

def downgrade() -> None:
    op.drop_index("ix_validation_reports_module", table_name="validation_reports")
    op.drop_table("validation_reports")
    op.drop_index("ix_review_threshold_configs_threshold_key", table_name="review_threshold_configs")
    op.drop_index("ix_review_threshold_configs_module", table_name="review_threshold_configs")
    op.drop_table("review_threshold_configs")
    op.drop_index("ix_calibration_configs_module", table_name="calibration_configs")
    op.drop_table("calibration_configs")
    op.drop_index("ix_audit_log_case_id", table_name="audit_log")
    op.drop_index("ix_audit_log_action", table_name="audit_log")
    op.drop_index("ix_audit_log_timestamp", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_case_module_results_case_module", table_name="case_module_results")
    op.drop_index("ix_case_module_results_reviewer_id", table_name="case_module_results")
    op.drop_index("ix_case_module_results_module", table_name="case_module_results")
    op.drop_index("ix_case_module_results_case_id", table_name="case_module_results")
    op.drop_table("case_module_results")
    op.drop_index("ix_cases_priority", table_name="cases")
    op.drop_index("ix_cases_assigned_reviewer", table_name="cases")
    op.drop_index("ix_cases_status", table_name="cases")
    op.drop_index("ix_cases_patient_id", table_name="cases")
    op.drop_table("cases")
