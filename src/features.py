"""Feature engineering (H1-6).

Closes a genuine gap: version 1.1.0 used the 19 raw columns unchanged, while
Task 1.2 of the brief lists feature engineering explicitly.

Leakage position, stated precisely
----------------------------------
Every feature here is a **stateless, row-wise function of that row's own
values**. Nothing is aggregated across rows, nothing is fitted, and no target
information is touched. A transformation that learns no parameter from the data
cannot leak between splits — the leakage rule concerns *fitted* statistics such
as an imputer's median or an encoder's category list, and those all remain
inside the pipeline where they were.

The engineering step is nevertheless placed **inside the pipeline** rather than
applied beforehand, for two reasons that matter more than the leakage argument:

1. The application passes the 19 raw columns and gets a prediction. It does not
   need to know these features exist, and cannot compute them inconsistently.
2. The saved artifact stays self-contained — the transformation is inseparable
   from the model, exactly as the preprocessing already is.

Design constraint
-----------------
Only features a retention team could actually explain are included. A feature
nobody can interpret buys accuracy at the cost of the governance story that is
this project's strongest asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

#: Service columns counted towards the "how invested is this customer" signal.
SERVICE_COLUMNS: list[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

#: Protective add-ons. Separated from streaming because the EDA showed support
#: and security behave very differently from entertainment extras.
PROTECTIVE_ADDONS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

ENGINEERED_NUMERIC: list[str] = [
    "AvgMonthlySpend",
    "ChargesTrend",
    "NumServices",
    "NumProtectiveAddons",
]

ENGINEERED_CATEGORICAL: list[str] = [
    "TenureBucket",
    "IsNewCustomer",
    "HasNoProtection",
]

#: Tenure bands. Boundaries follow the EDA finding that churn concentrates in
#: the first year, not an arbitrary equal-width split.
TENURE_BINS: list[float] = [-0.01, 6, 12, 24, 48, np.inf]
TENURE_LABELS: list[str] = ["0-6 months", "7-12 months", "1-2 years", "2-4 years", "4+ years"]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Pure, row-wise, and safe to call anywhere.

    Args:
        frame: A DataFrame containing at least the raw predictor columns.

    Returns:
        A copy with the engineered columns appended. The input is never mutated,
        because scikit-learn may call this on a slice during cross-validation.
    """
    out = frame.copy()

    tenure = pd.to_numeric(out["tenure"], errors="coerce")
    monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(out["TotalCharges"], errors="coerce")

    # --- Average spend per month of tenure ---------------------------------
    # A customer's realised average bill, which differs from their current bill
    # whenever pricing or their package has changed. Tenure is floored at 1 so a
    # brand-new customer yields their first bill rather than a division by zero.
    #
    # The 11 documented zero-tenure customers have a blank TotalCharges, so this
    # is NaN for them — deliberately. Filling it here would bake a domain
    # assumption into the feature; leaving it lets the pipeline's median imputer
    # handle it exactly as it already handles TotalCharges itself, keeping every
    # fitted statistic inside the pipeline where the leakage guarantee lives.
    safe_tenure = tenure.clip(lower=1)
    out["AvgMonthlySpend"] = total / safe_tenure

    # --- Direction of travel on price --------------------------------------
    # Above 1 means the customer now pays more than their historical average —
    # a recent increase or upsell. Below 1 means the reverse. This is the single
    # feature the raw columns cannot express: they hold levels, not change.
    # Falls back to 1.0 (no change) when the average is unknown or zero, which
    # is the neutral value rather than a guess in either direction.
    out["ChargesTrend"] = np.where(
        out["AvgMonthlySpend"] > 0, monthly / out["AvgMonthlySpend"], 1.0
    )

    # --- How embedded is the customer --------------------------------------
    # "No internet service" and "No phone service" are absences, not services,
    # so only an explicit "Yes" counts.
    service_frame = out[SERVICE_COLUMNS].astype(str)
    out["NumServices"] = (service_frame == "Yes").sum(axis=1)

    protective = out[PROTECTIVE_ADDONS].astype(str)
    out["NumProtectiveAddons"] = (protective == "Yes").sum(axis=1)

    # --- Tenure band --------------------------------------------------------
    # The linear model sees tenure only as a straight line. Banding lets it
    # capture the sharp early-life effect the EDA found (churners average 17.98
    # months against 37.57 for retained customers).
    out["TenureBucket"] = pd.cut(
        tenure.fillna(0), bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)

    out["IsNewCustomer"] = np.where(tenure.fillna(0) <= 6, "Yes", "No")

    # --- No protection at all ----------------------------------------------
    # The EDA showed customers without technical support churn at 41.64%
    # against 15.17% with it. This flags the compound case.
    out["HasNoProtection"] = np.where(out["NumProtectiveAddons"] == 0, "Yes", "No")

    return out


def engineered_feature_columns() -> tuple[list[str], list[str]]:
    """``(numeric, categorical)`` engineered column names."""
    return list(ENGINEERED_NUMERIC), list(ENGINEERED_CATEGORICAL)


def all_numeric_features() -> list[str]:
    return list(config.NUMERIC_FEATURES) + ENGINEERED_NUMERIC


def all_categorical_features() -> list[str]:
    return list(config.CATEGORICAL_FEATURES) + ENGINEERED_CATEGORICAL


def describe_features() -> list[dict[str, str]]:
    """Documentation of each engineered feature and its rationale."""
    return [
        {
            "name": "AvgMonthlySpend",
            "kind": "numeric",
            "definition": "TotalCharges / max(tenure, 1)",
            "rationale": (
                "The customer's realised average bill. Differs from MonthlyCharges "
                "whenever pricing or the package has changed."
            ),
        },
        {
            "name": "ChargesTrend",
            "kind": "numeric",
            "definition": "MonthlyCharges / AvgMonthlySpend",
            "rationale": (
                "Above 1 means the customer now pays more than their historical "
                "average. The raw columns hold levels; this is the only feature "
                "expressing change."
            ),
        },
        {
            "name": "NumServices",
            "kind": "numeric",
            "definition": "Count of the 8 service columns equal to 'Yes'",
            "rationale": "How embedded the customer is in the product set.",
        },
        {
            "name": "NumProtectiveAddons",
            "kind": "numeric",
            "definition": "Count of security, backup, device protection and tech support",
            "rationale": (
                "The EDA showed protective add-ons behave very differently from "
                "streaming extras, so they are counted separately."
            ),
        },
        {
            "name": "TenureBucket",
            "kind": "categorical",
            "definition": "tenure banded at 6, 12, 24 and 48 months",
            "rationale": (
                "A linear model sees tenure as a straight line. Banding lets it "
                "capture the sharp early-life churn effect found in the EDA."
            ),
        },
        {
            "name": "IsNewCustomer",
            "kind": "categorical",
            "definition": "tenure <= 6 months",
            "rationale": "Isolates the highest-risk early-life window.",
        },
        {
            "name": "HasNoProtection",
            "kind": "categorical",
            "definition": "NumProtectiveAddons == 0",
            "rationale": (
                "Customers without technical support churn at 41.64% against "
                "15.17% with it; this flags the compound case."
            ),
        },
    ]
