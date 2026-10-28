"""Shared settings for the churn pipeline.

Everything the pipeline needs to know lives here so the other modules
stay small: where files go, which columns we expect, and where the
quality gates are drawn. Tune the thresholds here, not in the code.
"""

from pathlib import Path

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# Columns a daily file must have. Anything missing fails validation.
REQUIRED_COLUMNS = [
    "user_id",
    "signup_date",
    "last_active_date",
    "sessions_30d",
    "avg_session_min",
    "support_tickets",
    "plan",
    "monthly_spend",
    "churned",
]

NUMERIC_COLUMNS = [
    "sessions_30d",
    "avg_session_min",
    "support_tickets",
    "monthly_spend",
]

CATEGORICAL_COLUMNS = ["plan"]

# Quality gates. If any CRITICAL check fails, training does not run.
# Thresholds are deliberately simple: this is a gate, not a research paper.
MIN_ROWS = 100            # too little data to train on
MAX_NULL_RATE = 0.05      # per column, after cleaning
MIN_CHURN_RATE = 0.01     # a dataset with ~no churn teaches nothing
MAX_CHURN_RATE = 0.90     # ...and one with ~all churn is broken
VALID_PLANS = {"basic", "pro", "premium"}
