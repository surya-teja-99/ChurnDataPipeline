"""Storage: write processed data and reports to disk.

Processed files are dated (churn_YYYY-MM-DD.csv) so every daily run is
kept, not overwritten. Reports are JSON so they are easy to read and
easy to feed into a dashboard later.
"""

import json
from pathlib import Path

import pandas as pd

from src import config


def save_processed(df: pd.DataFrame, date_str: str) -> Path:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / f"churn_{date_str}.csv"
    df.to_csv(path, index=False)
    return path


def save_report(report: dict, date_str: str) -> Path:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / f"validation_{date_str}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def save_metrics(metrics: dict, date_str: str) -> Path:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODEL_DIR / f"metrics_{date_str}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path
