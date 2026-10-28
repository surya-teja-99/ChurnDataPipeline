"""Validation: the quality gate between cleaning and training.

Every check returns a small dict: name, status (pass/fail/warn), detail.
The gate is simple on purpose:

  - One failed CRITICAL check -> the whole run fails, training is skipped.
  - Warnings are reported but never block training.

This is the file to edit when the team agrees on a new rule: add one
function, add it to CHECKS, and the report picks it up automatically.
"""

import pandas as pd

from src import config


def _check(name, passed, detail, critical=True):
    return {
        "name": name,
        "status": "pass" if passed else ("fail" if critical else "warn"),
        "detail": detail,
        "critical": critical,
    }


def check_columns(df):
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    return _check(
        "required_columns",
        not missing,
        "all required columns present" if not missing else f"missing: {missing}",
    )


def check_row_count(df):
    return _check(
        "row_count",
        len(df) >= config.MIN_ROWS,
        f"{len(df)} rows (minimum {config.MIN_ROWS})",
    )


def check_nulls(df):
    worst_col, worst_rate = None, 0.0
    for col in df.columns:
        rate = float(df[col].isna().mean()) if len(df) else 0.0
        if rate > worst_rate:
            worst_col, worst_rate = col, rate
    return _check(
        "null_rate",
        worst_rate <= config.MAX_NULL_RATE,
        ("all columns 0% null" if worst_rate == 0 else
         f"worst column {worst_col}: {worst_rate:.1%} null "
         f"(max {config.MAX_NULL_RATE:.0%})"),
    )


def check_duplicate_users(df):
    dupes = int(df["user_id"].duplicated().sum()) if "user_id" in df.columns else 0
    return _check(
        "duplicate_user_ids",
        dupes == 0,
        f"{dupes} duplicate user_id values",
    )


def check_labels(df):
    if "churned" not in df.columns:
        return _check("label_values", False, "churned column missing")
    bad = df[~df["churned"].isin([0, 1])]
    return _check(
        "label_values",
        len(bad) == 0,
        "churned is all 0/1" if len(bad) == 0 else f"{len(bad)} bad label values",
    )


def check_plans(df):
    if "plan" not in df.columns:
        return _check("plan_values", False, "plan column missing")
    bad = set(df["plan"].dropna().unique()) - config.VALID_PLANS
    return _check(
        "plan_values",
        not bad,
        "plans all valid" if not bad else f"unexpected plans: {sorted(bad)}",
    )


def check_negatives(df):
    bad_cols = []
    for col in config.NUMERIC_COLUMNS:
        if col in df.columns and (df[col] < 0).any():
            bad_cols.append(col)
    return _check(
        "no_negatives",
        not bad_cols,
        "no negative numerics" if not bad_cols else f"negatives in: {bad_cols}",
    )


def check_churn_rate(df):
    if "churned" not in df.columns or len(df) == 0:
        return _check("churn_rate", False, "cannot compute churn rate")
    rate = float(df["churned"].mean())
    return _check(
        "churn_rate",
        config.MIN_CHURN_RATE <= rate <= config.MAX_CHURN_RATE,
        f"churn rate {rate:.1%} (allowed "
        f"{config.MIN_CHURN_RATE:.0%}–{config.MAX_CHURN_RATE:.0%})",
    )


def check_cleaning_loss(clean_stats):
    """Warn (don't fail) if cleaning threw away a lot of the file."""
    rows_in = clean_stats.get("rows_in", 0)
    rows_out = clean_stats.get("rows_out", 0)
    if not rows_in:
        return _check("cleaning_loss", False, "no rows ingested")
    loss = 1 - rows_out / rows_in
    return _check(
        "cleaning_loss",
        True,
        f"cleaning kept {rows_out}/{rows_in} rows ({loss:.0%} dropped)",
        critical=False,
    )


CHECKS = [
    check_columns,
    check_row_count,
    check_nulls,
    check_duplicate_users,
    check_labels,
    check_plans,
    check_negatives,
    check_churn_rate,
]

# Checks that must run on the RAW frame: cleaning imputes nulls and
# normalises values, so running these after cleaning would hide problems.
RAW_CHECKS = [check_columns, check_nulls]

# Checks that run on the CLEANED frame.
CLEAN_CHECKS = [check_row_count, check_duplicate_users, check_labels,
                check_plans, check_negatives, check_churn_rate]


def validate_raw(df: pd.DataFrame) -> dict:
    """Raw-file gate: missing columns and heavy nulls fail before cleaning."""
    results = [fn(df) for fn in RAW_CHECKS]
    failed = [r for r in results if r["status"] == "fail"]
    return {"passed": not failed, "checks": results}


def validate(df: pd.DataFrame, clean_stats: dict | None = None) -> dict:
    """Run every check. passed is False if any critical check fails."""
    results = [fn(df) for fn in CLEAN_CHECKS]
    if clean_stats is not None:
        results.append(check_cleaning_loss(clean_stats))
    failed = [r for r in results if r["status"] == "fail"]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "checks": results,
    }
