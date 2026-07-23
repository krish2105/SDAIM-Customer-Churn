"""Central configuration: paths, dataset contract and modelling constants.

Every other module imports from here so that a single change propagates
consistently across validation, EDA, training, evaluation and the tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Paths (all derived from the repository root, never from the process cwd)
# --------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
RAW_DATASET_PATH: Final[Path] = RAW_DATA_DIR / "Telco-Customer-Churn.csv"

REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
TABLES_DIR: Final[Path] = REPORTS_DIR / "tables"

DEPLOY_DIR: Final[Path] = PROJECT_ROOT / "deploy"
ARTIFACTS_DIR: Final[Path] = DEPLOY_DIR / "artifacts"
MODEL_PATH: Final[Path] = ARTIFACTS_DIR / "model_pipeline.joblib"
METADATA_PATH: Final[Path] = ARTIFACTS_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH: Final[Path] = ARTIFACTS_DIR / "feature_schema.json"
MODEL_CARD_PATH: Final[Path] = ARTIFACTS_DIR / "model_card.md"
#: Descriptive reference rates used only to draw context charts in the app.
#: Computed on the training split (rates) and the held-out test split (score
#: histogram). Never used to make a prediction.
REFERENCE_RATES_PATH: Final[Path] = ARTIFACTS_DIR / "reference_rates.json"

SOURCE_MANIFEST_PATH: Final[Path] = PROJECT_ROOT / "SOURCE_MANIFEST.json"

# --------------------------------------------------------------------------
# Dataset contract (IBM Telco Customer Churn, official sample)
# --------------------------------------------------------------------------

PROJECT_NAME: Final[str] = (
    "Customer Churn Intelligence and Retention Decision-Support Platform"
)
MODEL_VERSION: Final[str] = "1.0.0"

EXPECTED_COLUMNS: Final[list[str]] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

EXPECTED_ROWS: Final[int] = 7043
EXPECTED_COLUMN_COUNT: Final[int] = 21
EXPECTED_GIT_BLOB_SHA: Final[str] = "3de7a612d1609f25f21a455bda77948729369002"

TARGET_COLUMN: Final[str] = "Churn"
TARGET_MAPPING: Final[dict[str, int]] = {"No": 0, "Yes": 1}
POSITIVE_CLASS: Final[int] = 1
ID_COLUMN: Final[str] = "customerID"

#: Numeric predictors. ``SeniorCitizen`` is deliberately excluded because the
#: data dictionary defines it as a binary category, not a continuous magnitude.
NUMERIC_FEATURES: Final[list[str]] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

#: Model input column order. The Streamlit app builds its one-row frame from
#: this list so the deployed schema can never drift from the trained schema.
FEATURE_COLUMNS: Final[list[str]] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

#: Expected category domains, taken from the IBM data dictionary and confirmed
#: against the validated raw file.
EXPECTED_CATEGORIES: Final[dict[str, list[str]]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

#: Service add-ons that are only meaningful when internet service is present.
INTERNET_DEPENDENT_FEATURES: Final[list[str]] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
NO_INTERNET_SENTINEL: Final[str] = "No internet service"
NO_PHONE_SENTINEL: Final[str] = "No phone service"

# --------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
DECISION_THRESHOLD: Final[float] = 0.5

#: Risk tiers are communication bands for human triage. They are NOT
#: independently validated business thresholds.
RISK_LOW_MAX: Final[float] = 0.40
RISK_MEDIUM_MAX: Final[float] = 0.70


def ensure_output_dirs() -> None:
    """Create every generated-output directory. Never touches ``data/raw``."""
    for directory in (PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def risk_tier(probability: float) -> str:
    """Map a churn probability to its communication band."""
    if probability < RISK_LOW_MAX:
        return "Low"
    if probability < RISK_MEDIUM_MAX:
        return "Medium"
    return "High"
