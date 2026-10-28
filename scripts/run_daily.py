"""Daily entry point.

Usage:
    python scripts/run_daily.py                  # today's file
    python scripts/run_daily.py --date 2026-10-27
    python scripts/run_daily.py --file data/raw/churn_sample.csv

Exit codes: 0 = trained, 2 = validation gate blocked training,
1 = ingest error (missing/unreadable file).
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Run the daily churn pipeline.")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Run date YYYY-MM-DD (default: today)")
    parser.add_argument("--file", default=None,
                        help="Run against a specific CSV instead of the daily file")
    args = parser.parse_args()

    result = run(args.date, input_file=args.file)
    if result["ok"]:
        return 0
    return 1 if result["stage"] == "ingest" else 2


if __name__ == "__main__":
    sys.exit(main())
