#!/usr/bin/env python3
"""Generate and execute ``notebooks/01_eda_and_modeling.ipynb``.

The notebook is a readable narrative over the same functions the scripts use;
it deliberately does not reimplement any analysis. Generating it from code keeps
it consistent with ``src/`` and guarantees every cell in the committed notebook
has actually run.

Usage::

    python scripts/build_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_eda_and_modeling.ipynb"


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip("\n"))


def build() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        markdown(
            """
# Customer Churn Intelligence — EDA and Modelling

**Project:** Customer Churn Intelligence and Retention Decision-Support Platform
**Dataset:** IBM Telco Customer Churn sample (7,043 × 21), Git blob SHA
`3de7a612d1609f25f21a455bda77948729369002`

> The dataset represents a **fictional** telecommunications company. It is not actual
> observed commercial customer data.

This notebook is a readable walk-through of the analysis. The reproducible source of
truth is `src/`: every function called here is the same one the scripts run, so the
notebook and the report can never drift apart. Nothing is reimplemented below.

Run order for the equivalent scripts:

```bash
make validate   # python -m src.data_validation
make eda        # python -m src.eda
make train      # python -m src.train
```
"""
        ),
        code(
            """
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src import config, eda, evaluate, schemas, train
from src.data_validation import validate_dataset

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)

print("Project root:", PROJECT_ROOT)
print("Raw dataset :", config.RAW_DATASET_PATH.name)
"""
        ),
        markdown(
            """
## 1. Data validation

Training refuses to start on a file that fails this contract. The raw file is never
modified: it is read, checked and reported on.
"""
        ),
        code(
            """
result = validate_dataset()

print("Validation passed:", result.passed)
print("Rows / columns   :", result.summary["rows"], "/", result.summary["columns"])
print("Git blob SHA     :", result.summary["git_blob_sha"])
print("Expected SHA     :", result.summary["expected_git_blob_sha"])
print("Target counts    :", result.summary["target_distribution"])
print("Checks executed  :", len(result.checks))

blank = {k: v for k, v in result.summary["blank_string_counts"].items() if v}
print("Blank strings    :", blank)
"""
        ),
        markdown(
            """
## 2. Loading the data for analysis

Only the documented type corrections are applied: `TotalCharges` is stripped and coerced
to numeric, and `SeniorCitizen` is kept as a category. **No imputation happens here** —
that belongs inside the pipeline so it can be fitted on the training split alone.
"""
        ),
        code(
            """
frame = eda.prepare_analysis_frame()
print(frame.shape)
frame.head()
"""
        ),
        code(
            """
frame.dtypes.to_frame("dtype")
"""
        ),
        markdown(
            """
## 3. Target balance

The class imbalance is the single most important fact for metric selection.
"""
        ),
        code(
            """
target_table = eda.target_distribution(frame)
target_table
"""
        ),
        markdown(
            """
A model predicting "no churn" for everyone would score the majority-class share as
accuracy while identifying nobody at risk. Accuracy alone is therefore not an acceptable
selection metric; recall, F1 and ROC-AUC are used instead.

## 4. Missing values
"""
        ),
        code(
            """
missing = eda.missing_value_summary(frame)
missing[missing["missing_after_coercion"] > 0]
"""
        ),
        code(
            """
# Every blank TotalCharges belongs to a customer with tenure 0 — the blank is
# structurally meaningful, not random.
blank_rows = frame[frame["TotalCharges"].isna()]
print("Rows with missing TotalCharges:", len(blank_rows))
print("Their tenure values:", sorted(blank_rows["tenure"].unique()))
"""
        ),
        markdown(
            """
## 5. Numeric distributions by churn status
"""
        ),
        code(
            """
tenure_stats = eda.numeric_distribution_by_churn(frame, "tenure", "03_tenure_by_churn.png")
tenure_stats
"""
        ),
        code(
            """
monthly_stats = eda.numeric_distribution_by_churn(
    frame, "MonthlyCharges", "04_monthlycharges_by_churn.png"
)
total_stats = eda.numeric_distribution_by_churn(
    frame, "TotalCharges", "05_totalcharges_by_churn.png"
)
monthly_stats
"""
        ),
        markdown(
            """
## 6. Churn rate by categorical predictor
"""
        ),
        code(
            """
contract = eda.churn_rate_by_category(frame, "Contract", "06_churn_rate_by_contract.png")
contract
"""
        ),
        code(
            """
internet = eda.churn_rate_by_category(
    frame, "InternetService", "07_churn_rate_by_internetservice.png"
)
payment = eda.churn_rate_by_category(
    frame, "PaymentMethod", "08_churn_rate_by_paymentmethod.png"
)
tech = eda.churn_rate_by_category(frame, "TechSupport", "09_churn_rate_by_techsupport.png")
security = eda.churn_rate_by_category(
    frame, "OnlineSecurity", "10_churn_rate_by_onlinesecurity.png"
)
payment
"""
        ),
        markdown(
            """
Note the confound: the `No internet service` level appears in every add-on column, so the
add-on comparisons partly restate the internet-service split rather than measuring the
add-on itself. These are associations within the sample, not causal effects.

## 7. Numeric correlations
"""
        ),
        code(
            """
correlation = eda.correlation_matrix(frame)
correlation
"""
        ),
        markdown(
            """
`tenure` and `TotalCharges` are strongly related because total charges accumulate with
time. Tree ensembles tolerate this; the linear model uses scaling and regularisation, and
the relationship is recorded as a known limitation.

## 8. Preprocessing and the leakage-safe split

The split happens **once, before any transformer is fitted**. Every imputer, encoder and
scaler lives inside the `Pipeline`, so cross-validation refits them per fold and no test
information can leak into training.
"""
        ),
        code(
            """
from sklearn.model_selection import train_test_split

model_frame = train.load_model_frame()
X, y = train.split_features_target(model_frame)

print("Feature matrix :", X.shape)
print("customerID in X:", config.ID_COLUMN in X.columns)
print("Churn in X     :", config.TARGET_COLUMN in X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y,
)
print(f"Train {X_train.shape[0]:,} rows (churn {y_train.mean():.4f})")
print(f"Test  {X_test.shape[0]:,} rows (churn {y_test.mean():.4f})")
"""
        ),
        code(
            """
candidates = train.build_candidates()
candidates["Logistic Regression"]
"""
        ),
        markdown(
            """
## 9. Model selection by cross-validation

The selection rule is fixed in `src/train.py` **before** any test-set result exists:
mean CV ROC-AUC first, then CV F1, then CV recall if the earlier gaps fall inside a 0.01
tolerance. Recall breaks the final tie because a missed churner costs more than an
unnecessary review.
"""
        ),
        code(
            """
cv_results = {
    name: train.cross_validate_model(pipeline, X_train, y_train)
    for name, pipeline in candidates.items()
}

pd.DataFrame(cv_results).T[
    ["cv_roc_auc_mean", "cv_roc_auc_std", "cv_f1_mean", "cv_recall_mean", "cv_precision_mean"]
].round(4)
"""
        ),
        code(
            """
baseline_cv = train.cross_validate_model(train.build_baseline(), X_train, y_train)
print(f"Dummy baseline CV ROC-AUC: {baseline_cv['cv_roc_auc_mean']:.4f} (reference only)")

selected_model, justification = train.select_model(cv_results)
print("\\nSELECTED:", selected_model)
print(justification)
"""
        ),
        markdown(
            """
## 10. Held-out evaluation

The test set is scored **once**, after selection.
"""
        ),
        code(
            """
test_metrics = {}
for name, pipeline in candidates.items():
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    predictions = (proba >= config.DECISION_THRESHOLD).astype(int)
    test_metrics[name] = evaluate.compute_metrics(y_test, predictions, proba)

pd.DataFrame(test_metrics).T.round(4)
"""
        ),
        code(
            """
from sklearn.metrics import confusion_matrix

selected_pipeline = candidates[selected_model]
proba = selected_pipeline.predict_proba(X_test)[:, 1]
predictions = (proba >= config.DECISION_THRESHOLD).astype(int)
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

pd.DataFrame(
    matrix,
    index=["actual: retained", "actual: churn"],
    columns=["predicted: retained", "predicted: churn"],
)
"""
        ),
        markdown(
            """
**Reading the errors.**

- A **false negative** is a customer who churned but was never flagged. No retention
  review happens, so the opportunity is lost silently. This is the costly error.
- A **false positive** is a customer flagged for review who would have stayed. The cost is
  specialist time and the risk of an unnecessary contact, which is recoverable.

This asymmetry is why recall carries weight in selection and why accuracy alone would be
misleading on an imbalanced target.

## 11. Artifacts

`make train` writes the complete pipeline plus metadata, feature schema, reference rates
and model card into `deploy/artifacts/`. The application loads exactly that artifact.
"""
        ),
        code(
            """
import json

metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
print("Model      :", metadata["model_name"], "v" + metadata["model_version"])
print("Trained    :", metadata["training_timestamp_utc"])
print("Dataset SHA:", metadata["dataset_git_blob_sha"])
print("Metrics    :", json.dumps(metadata["metrics"], indent=2))
"""
        ),
        code(
            """
schema = schemas.load_feature_schema()
print("Predictors :", len(schema["features"]))
print("Excluded id:", schema["excluded_identifier"])
print("Order match:", schema["feature_order"] == config.FEATURE_COLUMNS)
"""
        ),
        markdown(
            """
## 12. Conclusions carried forward

1. The target is imbalanced, so recall, F1 and ROC-AUC — not accuracy — govern selection.
2. `TotalCharges` needs coercion and in-pipeline imputation; the raw file stays untouched.
3. `SeniorCitizen` is a binary category, and `customerID` is excluded from the features.
4. Contract term, internet service, tenure and payment method are the strongest observed
   signals in the sample.
5. Outputs are **decision support requiring human review**. No financial or
   customer-treatment decision may be automated from them, and no revenue or
   churn-reduction effect has been measured.
"""
        ),
    ]

    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return notebook


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build()

    print(f"Executing {len(notebook.cells)} cells...")
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001
        print(f"Notebook execution FAILED: {exc}", file=sys.stderr)
        return 1

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote executed notebook -> {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
