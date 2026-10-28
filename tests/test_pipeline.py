"""Tests for the churn pipeline.

Run from the project root:  python -m unittest discover -s tests -v
The pipeline tests use temporary directories so they never touch the
real data/ folder.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.clean import clean
from src.pipeline import run
from src.validate import validate, validate_raw


def good_frame(n=150):
    return pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "signup_date": ["2025-01-01"] * n,
        "last_active_date": ["2026-10-01"] * n,
        "sessions_30d": [10 + (i % 20) for i in range(n)],
        "avg_session_min": [20.0 + (i % 10) for i in range(n)],
        "support_tickets": [i % 3 for i in range(n)],
        "plan": [["basic", "pro", "premium"][i % 3] for i in range(n)],
        "monthly_spend": [15.0 + (i % 30) for i in range(n)],
        "churned": [1 if i % 5 == 0 else 0 for i in range(n)],
    })


class TestClean(unittest.TestCase):
    def test_drops_duplicates_and_fills_nulls(self):
        df = good_frame(10)
        df.loc[0, "avg_session_min"] = None
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)
        cleaned, stats = clean(df)
        self.assertEqual(len(cleaned), 10)
        self.assertFalse(cleaned["avg_session_min"].isna().any())
        self.assertEqual(stats["dropped_duplicates"], 2)

    def test_never_imputes_labels(self):
        df = good_frame(10)
        df.loc[0, "churned"] = None
        cleaned, _ = clean(df)
        self.assertEqual(len(cleaned), 9)


class TestValidate(unittest.TestCase):
    def test_good_data_passes(self):
        report = validate(good_frame())
        self.assertTrue(report["passed"])

    def test_missing_column_fails_raw_gate(self):
        df = good_frame().drop(columns=["monthly_spend"])
        self.assertFalse(validate_raw(df)["passed"])

    def test_heavy_nulls_fail_raw_gate(self):
        df = good_frame()
        df.loc[:60, "avg_session_min"] = None
        self.assertFalse(validate_raw(df)["passed"])

    def test_duplicate_users_fail(self):
        df = good_frame()
        df.loc[1, "user_id"] = df.loc[0, "user_id"]
        report = validate(df)
        self.assertFalse(report["passed"])

    def test_too_few_rows_fails(self):
        report = validate(good_frame(20))
        self.assertFalse(report["passed"])


class TestPipelineGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        config.RAW_DIR = root / "raw"
        config.PROCESSED_DIR = root / "processed"
        config.MODEL_DIR = root / "models"
        config.REPORT_DIR = root / "reports"
        for d in (config.RAW_DIR, config.PROCESSED_DIR,
                  config.MODEL_DIR, config.REPORT_DIR):
            d.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_trains_on_good_data(self):
        f = config.RAW_DIR / "churn_2026-10-28.csv"
        good_frame(300).to_csv(f, index=False)
        result = run("2026-10-28")
        self.assertTrue(result["ok"])
        self.assertIn("metrics", result)
        self.assertTrue(list(config.MODEL_DIR.glob("*.pkl")))

    def test_skips_training_on_bad_data(self):
        f = config.RAW_DIR / "churn_2026-10-28.csv"
        good_frame(300).drop(columns=["monthly_spend"]).to_csv(f, index=False)
        result = run("2026-10-28")
        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "validate")
        self.assertFalse(list(config.MODEL_DIR.glob("*.pkl")))
        # Processed data and report are still stored for debugging.
        self.assertTrue(list(config.PROCESSED_DIR.glob("*.csv")))
        self.assertTrue(list(config.REPORT_DIR.glob("*.json")))


if __name__ == "__main__":
    unittest.main()
