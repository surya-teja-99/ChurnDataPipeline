"""Generate a realistic daily churn CSV for testing the pipeline.

The data is synthetic but shaped like real user activity: most users
are fine, a minority churn, and churn correlates with low sessions and
lots of support tickets. A little mess is included on purpose (a few
nulls, a couple of duplicate rows, mixed-case plan names) so cleaning
actually has something to do.

Usage:
    python scripts/generate_sample.py --rows 500 --out data/raw/churn_sample.csv
    python scripts/generate_sample.py --bad --out data/raw/churn_bad.csv
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PLANS = ["basic", "pro", "premium"]


def make_good(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = [f"U{10000 + i}" for i in range(rows)]
    plans = rng.choice(PLANS, size=rows, p=[0.5, 0.35, 0.15])
    tenure = rng.integers(30, 900, size=rows)
    today = date.today()
    signup = [today - timedelta(days=int(t)) for t in tenure]
    last_active = [s + timedelta(days=int(rng.integers(0, 40))) for s in signup]
    sessions = rng.poisson(18, size=rows)
    tickets = rng.poisson(1.2, size=rows)
    # Churn risk grows with tickets and shrinks with sessions.
    risk = 1 / (1 + np.exp(-(-2.0 + 0.8 * tickets - 0.08 * sessions)))
    churned = (rng.random(rows) < risk).astype(int)
    spend = np.where(
        plans == "premium", rng.normal(49, 8, rows),
        np.where(plans == "pro", rng.normal(29, 6, rows), rng.normal(12, 4, rows)),
    ).clip(min=1).round(2)

    df = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": [d.isoformat() for d in signup],
        "last_active_date": [d.isoformat() for d in last_active],
        "sessions_30d": sessions,
        "avg_session_min": rng.normal(24, 9, rows).clip(min=1).round(1),
        "support_tickets": tickets,
        "plan": plans,
        "monthly_spend": spend,
        "churned": churned,
    })

    # A little realistic mess.
    for col in ["avg_session_min", "monthly_spend"]:
        idx = rng.choice(rows, size=max(1, rows // 50), replace=False)
        df.loc[idx, col] = np.nan
    df.loc[rng.choice(rows, size=3, replace=False), "plan"] = "Pro"  # mixed case
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_bad(rows: int, seed: int = 7) -> pd.DataFrame:
    """A file the gate must reject: missing column, heavy nulls."""
    df = make_good(rows, seed)
    df = df.drop(columns=["monthly_spend"])
    idx = np.random.default_rng(seed).choice(
        rows, size=int(rows * 0.4), replace=False)
    df.loc[idx, "avg_session_min"] = np.nan
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bad", action="store_true",
                        help="Generate a broken file the gate should reject")
    args = parser.parse_args()

    df = make_bad(args.rows) if args.bad else make_good(args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
