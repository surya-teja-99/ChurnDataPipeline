"""Cleaning: turn the raw daily file into something model-ready.

Rules, in order:
  1. Drop rows with no user_id (can't track a user without one).
  2. Drop exact duplicate rows.
  3. Normalise text: strip whitespace, lowercase plan names.
  4. Parse dates; rows with unparseable dates are dropped.
  5. Fill missing numerics with the column median, missing plans with
     the mode. Median beats mean here because spend is skewed.
  6. Clip impossible values: sessions, tickets and spend can't be
     negative, so negative values become 0.

Cleaning never invents labels: a row with a missing churned value is
dropped, not filled. Everything removed is counted and returned so the
validation report can say what happened.
"""

import pandas as pd

from src import config


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_in": len(df)}
    out = df.copy()

    out = out.dropna(subset=["user_id"])
    stats["dropped_no_user_id"] = stats["rows_in"] - len(out)

    before = len(out)
    out = out.drop_duplicates()
    stats["dropped_duplicates"] = before - len(out)

    for col in config.CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.lower()

    for col in ["signup_date", "last_active_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["signup_date", "last_active_date"])
    stats["dropped_bad_dates"] = before - len(out)

    # Labels are sacred: never impute churned.
    before = len(out)
    out = out.dropna(subset=["churned"])
    stats["dropped_no_label"] = before - len(out)

    for col in config.NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            median = out[col].median()
            out[col] = out[col].fillna(median).clip(lower=0)

    for col in config.CATEGORICAL_COLUMNS:
        if col in out.columns:
            mode = out[col].mode()
            fill = mode.iloc[0] if not mode.empty else "unknown"
            out[col] = out[col].fillna(fill)

    stats["rows_out"] = len(out)
    return out, stats
