"""Pipeline orchestration: ingest -> clean -> validate -> store -> train?

The one rule that matters lives here: training runs ONLY if validation
passes. A failed gate still stores the processed data and the report,
so a bad day is visible and debuggable instead of silently skipped.
"""

from src import config
from src.clean import clean
from src.ingest import IngestError, daily_file_for, load_csv
from src.store import save_metrics, save_processed, save_report
from src.train import train
from src.validate import validate, validate_raw


def _print_checks(checks):
    for check in checks:
        mark = {"pass": "ok  ", "fail": "FAIL", "warn": "warn"}.get(check["status"], "?")
        print(f"[validate] {mark} {check['name']}: {check['detail']}")


def run(date_str: str, input_file=None) -> dict:
    path = input_file or daily_file_for(date_str)
    print(f"[ingest]  loading {path}")
    try:
        raw = load_csv(path)
    except IngestError as exc:
        print(f"[ingest]  FAILED: {exc}")
        return {"ok": False, "stage": "ingest", "error": str(exc)}
    print(f"[ingest]  {len(raw)} rows, columns: {list(raw.columns)}")

    # Raw gate first: cleaning would impute nulls and hide a bad file.
    raw_report = validate_raw(raw)
    _print_checks(raw_report["checks"])

    cleaned, clean_stats = clean(raw)
    print(f"[clean]   {clean_stats['rows_in']} -> {clean_stats['rows_out']} rows "
          f"({clean_stats['dropped_duplicates']} dupes dropped)")

    report = validate(cleaned, clean_stats)
    _print_checks(report["checks"])
    all_checks = raw_report["checks"] + report["checks"]
    passed = raw_report["passed"] and report["passed"]
    report = {"passed": passed,
              "failed_count": sum(1 for c in all_checks if c["status"] == "fail"),
              "checks": all_checks}

    processed_path = save_processed(cleaned, date_str)
    report_path = save_report(
        {"date": date_str, "clean_stats": clean_stats, **report}, date_str
    )
    print(f"[store]   processed -> {processed_path}")
    print(f"[store]   report    -> {report_path}")

    if not passed:
        print("[gate]    VALIDATION FAILED - training skipped.")
        return {"ok": False, "stage": "validate", "report": report}

    print("[gate]    validation passed - training model...")
    metrics = train(cleaned, date_str)
    metrics_path = save_metrics(metrics, date_str)
    print(f"[train]   accuracy={metrics['accuracy']} f1={metrics['f1']} "
          f"(model -> {metrics['model_path']})")
    print(f"[train]   metrics -> {metrics_path}")
    return {"ok": True, "stage": "train", "report": report, "metrics": metrics}
