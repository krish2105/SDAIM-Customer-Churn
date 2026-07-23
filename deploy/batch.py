"""Batch scoring and retention work-queue construction (H2-3).

Closes gap G7. Scoring one customer at a time makes the tool a calculator; a
retention manager needs a ranked book of accounts. This module turns an uploaded
CSV into a prioritised queue.

Validation is strict and up front: the file is checked against the persisted
feature schema and rejected with per-column errors *before* any scoring starts,
rather than failing part-way through with a partial result the user might trust.

Nothing here writes to disk. Uploaded customer data is scored in memory and
discarded when the session ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Upper bound on rows accepted in one upload. Chosen so a scoring pass stays
#: comfortably inside a Space's request budget; exceeding it is a clear error
#: rather than a slow, silent degradation.
MAX_ROWS = 5_000


@dataclass
class ValidationResult:
    """Outcome of checking an uploaded frame against the schema."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    identifier_column: str | None = None


def validate_batch(frame: pd.DataFrame, schema: dict[str, Any]) -> ValidationResult:
    """Check an uploaded frame against the model input contract.

    Errors block scoring. Warnings do not: an unseen category is absorbed by
    ``handle_unknown="ignore"`` and is worth flagging, not refusing.
    """
    errors: list[str] = []
    warnings: list[str] = []
    required = list(schema["feature_order"])

    if frame.empty:
        return ValidationResult(ok=False, errors=["The uploaded file contains no rows."])

    if len(frame) > MAX_ROWS:
        errors.append(
            f"The file contains {len(frame):,} rows. The maximum accepted is {MAX_ROWS:,}. "
            "Split the file and upload it in parts."
        )

    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            "The file must contain every model predictor, named exactly as in the schema."
        )

    # An identifier is optional but makes the queue actionable. It is carried
    # through for traceability and never used as a predictor.
    identifier = schema.get("excluded_identifier")
    identifier_column = identifier if identifier in frame.columns else None

    extra = [
        column
        for column in frame.columns
        if column not in required and column != identifier_column
    ]
    if extra:
        warnings.append(
            f"Ignoring {len(extra)} column(s) not used by the model: "
            f"{', '.join(extra[:6])}{' …' if len(extra) > 6 else ''}."
        )

    if missing:
        # Further checks would be misleading against an incomplete frame.
        return ValidationResult(ok=False, errors=errors, warnings=warnings,
                                identifier_column=identifier_column)

    # Numeric columns must be coercible and non-negative.
    for column in schema["numeric_features"]:
        coerced = pd.to_numeric(
            frame[column].astype(str).str.strip().replace({"": None}), errors="coerce"
        )
        unparseable = int(coerced.isna().sum())
        if unparseable:
            rows = (coerced.isna().to_numpy().nonzero()[0][:5] + 2).tolist()
            warnings.append(
                f"'{column}': {unparseable} value(s) could not be read as a number "
                f"(first affected file row(s): {rows}). These will be median-imputed by the "
                "pipeline, exactly as during training."
            )
        negatives = int((coerced.dropna() < 0).sum())
        if negatives:
            errors.append(
                f"'{column}': {negatives} negative value(s). This field cannot be below zero."
            )

    # Unseen categories are tolerated by the encoder but are worth surfacing.
    by_name = {feature["name"]: feature for feature in schema["features"]}
    for column in schema["categorical_features"]:
        known = set(by_name[column]["categories"])
        observed = set(frame[column].astype(str).str.strip().unique())
        unknown = sorted(observed - known)
        if unknown:
            warnings.append(
                f"'{column}': unrecognised value(s) {unknown[:4]}"
                f"{' …' if len(unknown) > 4 else ''}. These were not seen during training and "
                "will contribute nothing to the score."
            )

    return ValidationResult(
        ok=not errors, errors=errors, warnings=warnings, identifier_column=identifier_column
    )


def prepare_batch(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Coerce an uploaded frame into exactly the training schema and dtypes."""
    prepared = frame.loc[:, list(schema["feature_order"])].copy()
    for column in schema["categorical_features"]:
        prepared[column] = prepared[column].astype(str).str.strip()
    for column in schema["numeric_features"]:
        prepared[column] = pd.to_numeric(
            prepared[column].astype(str).str.strip().replace({"": None}), errors="coerce"
        ).astype(float)
    return prepared


def score_batch(
    pipeline,
    frame: pd.DataFrame,
    schema: dict[str, Any],
    threshold: float,
    identifier_column: str | None = None,
    risk_bands: tuple[float, float] = (0.40, 0.70),
) -> pd.DataFrame:
    """Score every row and return a queue ranked by descending risk."""
    prepared = prepare_batch(frame, schema)
    probabilities = pipeline.predict_proba(prepared)[:, 1]
    low_max, medium_max = risk_bands

    queue = pd.DataFrame(
        {
            "Churn probability": probabilities,
            "Predicted class": np.where(probabilities >= threshold, "Likely to churn",
                                        "Not likely to churn"),
            "Risk band": np.select(
                [probabilities < low_max, probabilities < medium_max],
                ["Low", "Medium"],
                default="High",
            ),
        }
    )

    if identifier_column and identifier_column in frame.columns:
        queue.insert(0, "Customer", frame[identifier_column].astype(str).to_numpy())
    else:
        queue.insert(0, "Customer", [f"Row {i + 2}" for i in range(len(frame))])

    # Context columns a specialist needs to triage without reopening the source file.
    for column in ("Contract", "tenure", "MonthlyCharges"):
        if column in prepared.columns:
            queue[column] = prepared[column].to_numpy()

    queue = queue.sort_values("Churn probability", ascending=False).reset_index(drop=True)
    queue.insert(0, "Priority", range(1, len(queue) + 1))
    return queue


def queue_summary(queue: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Headline counts for the work-queue header."""
    total = len(queue)
    flagged = int((queue["Predicted class"] == "Likely to churn").sum())
    bands = queue["Risk band"].value_counts().to_dict()
    return {
        "total": total,
        "flagged": flagged,
        "flagged_share": flagged / total if total else 0.0,
        "high": int(bands.get("High", 0)),
        "medium": int(bands.get("Medium", 0)),
        "low": int(bands.get("Low", 0)),
        "threshold": threshold,
        "mean_probability": float(queue["Churn probability"].mean()) if total else 0.0,
    }
