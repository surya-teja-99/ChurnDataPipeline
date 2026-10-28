"""Ingestion: find and load the daily CSV.

The convention is one file per day: data/raw/churn_YYYY-MM-DD.csv.
run_daily.py defaults to today's file; you can also point at any file
with --file. Ingestion does no cleaning, it just loads and reports
what it found so the later stages have something honest to work with.
"""

from pathlib import Path

import pandas as pd

from src import config


class IngestError(Exception):
    """Raised when there is nothing usable to ingest."""


def daily_file_for(date_str: str) -> Path:
    return config.RAW_DIR / f"churn_{date_str}.csv"


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise IngestError(f"no file at {path}")
    if path.stat().st_size == 0:
        raise IngestError(f"file is empty: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise IngestError(f"could not parse {path}: {exc}") from exc
    if df.empty:
        raise IngestError(f"file has no rows: {path}")
    # Normalise headers: " User_ID " and "user_id" should be the same column.
    df.columns = [c.strip().lower() for c in df.columns]
    return df
