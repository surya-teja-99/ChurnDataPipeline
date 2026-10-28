# Churn ML Data Pipeline

A simple daily ETL pipeline for a churn prediction model. Each day a new
CSV of user activity arrives, and the pipeline:

1. **Ingests** it from `data/raw/churn_YYYY-MM-DD.csv`
2. **Cleans** it (dedupes, fixes casing, imputes missing values, drops
   rows with no user or no label)
3. **Validates** it against a set of quality checks
4. **Stores** the processed data and a validation report
5. **Trains** a churn model — but only if validation passes

The whole point is step 5's condition: a bad day of data should never
silently retrain the model. If a check fails, training is skipped and
the report says exactly which check failed.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Make a sample daily file and run the pipeline on it
python scripts/generate_sample.py --rows 500 --out data/raw/churn_sample.csv
python scripts/run_daily.py --file data/raw/churn_sample.csv
```

To run it like a real daily job, name the file by date and skip `--file`:

```bash
python scripts/generate_sample.py --out data/raw/churn_2026-10-28.csv
python scripts/run_daily.py --date 2026-10-28
```

See the gate in action with a broken file:

```bash
python scripts/generate_sample.py --bad --out data/raw/churn_bad.csv
python scripts/run_daily.py --file data/raw/churn_bad.csv   # training skipped
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## The quality gate

| Check | Rule |
|---|---|
| required_columns | all 9 expected columns present |
| row_count | at least 100 rows |
| null_rate | no column more than 5% null after cleaning |
| duplicate_user_ids | zero duplicates |
| label_values | `churned` is only 0 or 1 |
| plan_values | plans are basic / pro / premium |
| no_negatives | sessions, tickets, spend are never negative |
| churn_rate | between 1% and 90% |

Any failed check blocks training. Cleaning loss (how much of the file
was dropped) is reported as a warning only.

Two checks run on the **raw** file before cleaning — required columns
and null rate — because cleaning imputes nulls and would otherwise
hide a bad file. The rest run on the cleaned data.

Thresholds live in `src/config.py` — change them there, not in the code.

## Project layout

```
src/
  config.py    settings + gate thresholds
  ingest.py    load the daily CSV
  clean.py     dedupe, impute, normalise
  validate.py  the quality gate
  store.py     write processed data + reports
  train.py     logistic regression churn model
  pipeline.py  ties it together, enforces the gate
scripts/
  generate_sample.py   synthetic daily data (good + broken variants)
  run_daily.py         CLI entry point
tests/
  test_pipeline.py     unit + gate tests
```

## Exit codes

- `0` — validation passed, model trained
- `2` — validation failed, training skipped (processed data + report saved)
- `1` — ingest error (missing or unreadable file)
