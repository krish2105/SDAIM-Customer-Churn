"""Raw dataset integrity validation.

The raw file under ``data/raw`` is treated as immutable evidence: this module
reads it, reports on it and never writes to it. Validation failures are raised
loudly; nothing is silently repaired or substituted.

CLI::

    python -m src.data_validation
    python -m src.data_validation --path data/raw/Telco-Customer-Churn.csv --strict-sha
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import config


class DatasetValidationError(RuntimeError):
    """Raised when the raw dataset violates the documented contract."""


@dataclass
class ValidationResult:
    """Outcome of a full validation pass."""

    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if not check["passed"]]

    def add(self, name: str, passed: bool, expected: Any, observed: Any, detail: str = "") -> None:
        self.checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "expected": expected,
                "observed": observed,
                "detail": detail,
            }
        )
        if not passed:
            self.passed = False


def git_blob_sha(path: Path) -> str | None:
    """Return ``git hash-object`` for *path*, or ``None`` when Git is absent."""
    try:
        completed = subprocess.run(
            ["git", "hash-object", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def load_raw_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV with every column as text, preserving it byte-for-byte.

    Reading as ``str`` is deliberate: it keeps blank ``TotalCharges`` values
    visible as blanks so validation can report them, instead of letting pandas
    infer a type and hide the issue.
    """
    dataset_path = path or config.RAW_DATASET_PATH
    if not dataset_path.is_file():
        raise DatasetValidationError(
            f"Raw dataset not found at {dataset_path}. "
            "Place the official IBM Telco-Customer-Churn.csv there; do not substitute another file."
        )
    return pd.read_csv(dataset_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_dataset(path: Path | None = None, *, strict_sha: bool = False) -> ValidationResult:
    """Run every documented integrity check against the raw dataset."""
    dataset_path = path or config.RAW_DATASET_PATH
    result = ValidationResult(passed=True)

    # 1. File existence -----------------------------------------------------
    exists = dataset_path.is_file()
    result.add("file_exists", exists, True, exists, str(dataset_path))
    if not exists:
        raise DatasetValidationError(f"Raw dataset not found at {dataset_path}")

    frame = load_raw_dataframe(dataset_path)

    # 2. Exact columns and order -------------------------------------------
    observed_columns = list(frame.columns)
    result.add(
        "columns_exact_and_ordered",
        observed_columns == config.EXPECTED_COLUMNS,
        config.EXPECTED_COLUMNS,
        observed_columns,
        "Column names and their order must match the IBM data dictionary exactly.",
    )

    # 3. Row count ----------------------------------------------------------
    result.add(
        "row_count",
        len(frame) == config.EXPECTED_ROWS,
        config.EXPECTED_ROWS,
        int(len(frame)),
    )

    # 4. Duplicate full rows ------------------------------------------------
    duplicate_rows = int(frame.duplicated().sum())
    result.add("no_duplicate_full_rows", duplicate_rows == 0, 0, duplicate_rows)

    # 5. Unique identifier --------------------------------------------------
    if config.ID_COLUMN in frame.columns:
        duplicate_ids = int(frame[config.ID_COLUMN].duplicated().sum())
        result.add("customer_id_unique", duplicate_ids == 0, 0, duplicate_ids)
    else:
        result.add("customer_id_unique", False, 0, "column missing")

    # 6. Target domain ------------------------------------------------------
    if config.TARGET_COLUMN in frame.columns:
        target_values = sorted(frame[config.TARGET_COLUMN].unique().tolist())
        result.add(
            "target_domain",
            target_values == ["No", "Yes"],
            ["No", "Yes"],
            target_values,
        )
    else:
        result.add("target_domain", False, ["No", "Yes"], "column missing")

    # 7. SeniorCitizen domain ----------------------------------------------
    if "SeniorCitizen" in frame.columns:
        senior_values = sorted(frame["SeniorCitizen"].unique().tolist())
        result.add(
            "senior_citizen_domain",
            set(senior_values).issubset({"0", "1"}),
            ["0", "1"],
            senior_values,
        )
    else:
        result.add("senior_citizen_domain", False, ["0", "1"], "column missing")

    # 8. Numeric conversion viability --------------------------------------
    numeric_report: dict[str, dict[str, Any]] = {}
    for column in config.NUMERIC_FEATURES:
        if column not in frame.columns:
            result.add(f"numeric_convertible__{column}", False, "convertible", "column missing")
            continue
        coerced = pd.to_numeric(frame[column].str.strip(), errors="coerce")
        blank_count = int((frame[column].str.strip() == "").sum())
        non_blank_uncoercible = int(coerced.isna().sum() - blank_count)
        numeric_report[column] = {
            "blank_values": blank_count,
            "non_blank_uncoercible": non_blank_uncoercible,
            "min": None if coerced.dropna().empty else float(coerced.min()),
            "max": None if coerced.dropna().empty else float(coerced.max()),
            "median": None if coerced.dropna().empty else float(coerced.median()),
        }
        # Blanks are documented and expected in TotalCharges; values that are
        # non-blank yet still uncoercible would indicate genuine corruption.
        result.add(
            f"numeric_convertible__{column}",
            non_blank_uncoercible == 0,
            0,
            non_blank_uncoercible,
            "Non-blank values that could not be parsed as numeric.",
        )

        # 9. Impossible negative values ------------------------------------
        negatives = int((coerced.dropna() < 0).sum())
        result.add(f"no_negative_values__{column}", negatives == 0, 0, negatives)

    # 10. Unexpected categories --------------------------------------------
    category_report: dict[str, list[str]] = {}
    for column, expected in config.EXPECTED_CATEGORIES.items():
        if column not in frame.columns:
            result.add(f"categories__{column}", False, expected, "column missing")
            continue
        observed = sorted(frame[column].unique().tolist())
        category_report[column] = observed
        unexpected = sorted(set(observed) - set(expected))
        result.add(
            f"categories__{column}",
            not unexpected,
            expected,
            observed,
            f"Unexpected categories: {unexpected}" if unexpected else "",
        )

    # 11. Blank and missing-value summary ----------------------------------
    blank_summary = {
        column: int((frame[column].astype(str).str.strip() == "").sum()) for column in frame.columns
    }
    null_summary = {column: int(frame[column].isna().sum()) for column in frame.columns}

    result.summary = {
        "path": str(dataset_path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "git_blob_sha": git_blob_sha(dataset_path),
        "expected_git_blob_sha": config.EXPECTED_GIT_BLOB_SHA,
        "target_distribution": (
            frame[config.TARGET_COLUMN].value_counts().to_dict()
            if config.TARGET_COLUMN in frame.columns
            else {}
        ),
        "blank_string_counts": blank_summary,
        "null_counts": null_summary,
        "numeric_columns": numeric_report,
        "observed_categories": category_report,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # 12. Git blob SHA ------------------------------------------------------
    observed_sha = result.summary["git_blob_sha"]
    if observed_sha is None:
        result.add(
            "git_blob_sha",
            not strict_sha,
            config.EXPECTED_GIT_BLOB_SHA,
            "git unavailable",
            "Git was not available, so the blob SHA could not be verified.",
        )
    else:
        result.add(
            "git_blob_sha",
            observed_sha == config.EXPECTED_GIT_BLOB_SHA,
            config.EXPECTED_GIT_BLOB_SHA,
            observed_sha,
            "Identifies the exact official file revision.",
        )

    return result


def write_report(result: ValidationResult, path: Path | None = None) -> Path:
    """Persist the validation report as JSON."""
    config.ensure_output_dirs()
    report_path = path or (config.TABLES_DIR / "data_validation_report.json")
    payload = {
        "passed": result.passed,
        "summary": result.summary,
        "checks": result.checks,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the raw IBM Telco churn dataset.")
    parser.add_argument("--path", type=Path, default=config.RAW_DATASET_PATH)
    parser.add_argument(
        "--strict-sha",
        action="store_true",
        help="Fail when the Git blob SHA cannot be computed (Git unavailable).",
    )
    args = parser.parse_args(argv)

    try:
        result = validate_dataset(args.path, strict_sha=args.strict_sha)
    except DatasetValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    report_path = write_report(result)

    print("=" * 72)
    print("IBM Telco Customer Churn — raw dataset validation")
    print("=" * 72)
    print(f"File            : {result.summary['path']}")
    print(f"Rows / columns  : {result.summary['rows']} / {result.summary['columns']}")
    print(f"Git blob SHA    : {result.summary['git_blob_sha']}")
    print(f"Expected SHA    : {result.summary['expected_git_blob_sha']}")
    print(f"Target counts   : {result.summary['target_distribution']}")
    blanks = {k: v for k, v in result.summary["blank_string_counts"].items() if v}
    print(f"Blank strings   : {blanks or 'none'}")
    print(f"Checks executed : {len(result.checks)}")
    print(f"Report written  : {report_path}")

    if not result.passed:
        print("-" * 72, file=sys.stderr)
        print("VALIDATION FAILED. Offending checks:", file=sys.stderr)
        for failure in result.failures:
            print(
                f"  - {failure['check']}: expected {failure['expected']!r}, "
                f"observed {failure['observed']!r} {failure['detail']}",
                file=sys.stderr,
            )
        print(
            "Do not repair the raw file in place. Re-download the official dataset "
            "from the URL in SOURCE_MANIFEST.json.",
            file=sys.stderr,
        )
        return 1

    print("RESULT          : PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
