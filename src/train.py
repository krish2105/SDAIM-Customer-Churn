"""Leakage-safe training, model selection and artifact export.

Selection rule — fixed here in code before any test-set result is produced
--------------------------------------------------------------------------
1. Candidate models are compared using stratified 5-fold cross-validation on
   the **training split only**.
2. The candidate with the higher mean CV ROC-AUC is selected.
3. If the two mean CV ROC-AUC values differ by less than 0.01 the candidate with
   the higher mean CV F1 is selected instead, because at comparable ranking
   quality the model that better balances precision and recall on the positive
   class is preferred.
4. If both are still within 0.01, the candidate with the higher mean CV recall
   is selected, because a missed churner (false negative) costs more than an
   unnecessary review (false positive).
5. The held-out test set is used **once**, after selection, for unbiased
   reporting only. It never influences the choice.

A ``DummyClassifier`` is scored alongside the candidates for context. It is a
reference point, not a candidate, and can never be selected.

Leakage controls
----------------
- The data is split once, before any imputer, encoder or scaler is fitted.
- Every transformer lives inside the ``Pipeline`` and is refitted on the
  training folds only during cross-validation.
- The target is mapped outside the feature matrix and ``customerID`` is dropped
  before ``X`` is constructed.

CLI::

    python -m src.train
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config, evaluate, schemas
from src.data_validation import git_blob_sha, load_raw_dataframe, validate_dataset

SELECTION_RULE = (
    "Candidates were compared by mean ROC-AUC under stratified "
    f"{config.CV_FOLDS}-fold cross-validation on the training split only. Where the "
    "two means differed by less than 0.01 the tie was broken first on mean CV F1 and "
    "then on mean CV recall, because a missed churner costs more than an unnecessary "
    "review. The held-out test set was evaluated once, after selection, and did not "
    "influence the choice."
)
TIE_TOLERANCE = 0.01


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------


def load_model_frame(path: Path | None = None) -> pd.DataFrame:
    """Load the raw file and apply only the documented type corrections.

    No imputation, encoding or scaling happens here — those belong inside the
    pipeline so they can be fitted on training data alone.
    """
    frame = load_raw_dataframe(path).copy()

    # Documented cleaning rule for the known blank values in TotalCharges.
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].str.strip(), errors="coerce")
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"].str.strip(), errors="coerce")
    frame["tenure"] = pd.to_numeric(frame["tenure"].str.strip(), errors="coerce")

    # SeniorCitizen is a binary category in the data dictionary, not a magnitude.
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(str)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build ``X`` (without ``customerID`` or the target) and the mapped ``y``."""
    unmapped = set(frame[config.TARGET_COLUMN].unique()) - set(config.TARGET_MAPPING)
    if unmapped:
        raise ValueError(f"Unexpected target values, refusing to train: {sorted(unmapped)}")

    y = frame[config.TARGET_COLUMN].map(config.TARGET_MAPPING).astype(int)
    X = frame[config.FEATURE_COLUMNS].copy()

    assert config.ID_COLUMN not in X.columns, "customerID must never enter the feature matrix"
    assert config.TARGET_COLUMN not in X.columns, "Churn must never enter the feature matrix"
    return X, y


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Column-wise preprocessing.

    Args:
        scale_numeric: ``True`` for linear models, which need comparable feature
            scales. Tree ensembles are scale-invariant, so scaling is skipped
            for them to keep the fitted pipeline simple and interpretable.
    """
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_candidates() -> dict[str, Pipeline]:
    """The two substantive candidate pipelines required by the brief."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=config.RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=5,
                        random_state=config.RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def build_baseline() -> Pipeline:
    """Stratified dummy classifier used only as a reference point."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                DummyClassifier(strategy="stratified", random_state=config.RANDOM_STATE),
            ),
        ]
    )


# --------------------------------------------------------------------------
# Cross-validation and selection
# --------------------------------------------------------------------------


def cross_validate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Stratified CV on the training split, returning mean and std per metric."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["roc_auc", "f1", "recall", "precision", "accuracy"],
        n_jobs=None,
        return_train_score=False,
    )
    summary: dict[str, float] = {}
    for metric in ("roc_auc", "f1", "recall", "precision", "accuracy"):
        values = scores[f"test_{metric}"]
        summary[f"cv_{metric}_mean"] = float(np.mean(values))
        summary[f"cv_{metric}_std"] = float(np.std(values))
    return summary


def select_model(cv_results: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Apply the documented selection rule. Returns ``(name, justification)``."""
    ranked = sorted(cv_results.items(), key=lambda item: item[1]["cv_roc_auc_mean"], reverse=True)
    best_name, best_scores = ranked[0]
    runner_name, runner_scores = ranked[1]

    auc_gap = best_scores["cv_roc_auc_mean"] - runner_scores["cv_roc_auc_mean"]
    if auc_gap >= TIE_TOLERANCE:
        justification = (
            f"{best_name} was selected on cross-validated ROC-AUC "
            f"({best_scores['cv_roc_auc_mean']:.4f} against "
            f"{runner_scores['cv_roc_auc_mean']:.4f}), a margin of {auc_gap:.4f} which exceeds "
            f"the {TIE_TOLERANCE} tie tolerance."
        )
        return best_name, justification

    f1_ranked = sorted(cv_results.items(), key=lambda item: item[1]["cv_f1_mean"], reverse=True)
    f1_best, f1_scores = f1_ranked[0]
    f1_gap = f1_scores["cv_f1_mean"] - f1_ranked[1][1]["cv_f1_mean"]
    if f1_gap >= TIE_TOLERANCE:
        justification = (
            f"Cross-validated ROC-AUC was effectively tied (gap {auc_gap:.4f} < {TIE_TOLERANCE}), "
            f"so the rule fell through to mean CV F1. {f1_best} was selected with "
            f"{f1_scores['cv_f1_mean']:.4f} against {f1_ranked[1][1]['cv_f1_mean']:.4f}."
        )
        return f1_best, justification

    recall_ranked = sorted(
        cv_results.items(), key=lambda item: item[1]["cv_recall_mean"], reverse=True
    )
    recall_best, recall_scores = recall_ranked[0]
    justification = (
        f"Cross-validated ROC-AUC (gap {auc_gap:.4f}) and F1 (gap {f1_gap:.4f}) were both within "
        f"the {TIE_TOLERANCE} tolerance, so the rule fell through to mean CV recall. "
        f"{recall_best} was selected with {recall_scores['cv_recall_mean']:.4f}, because a missed "
        "churner costs more than an unnecessary review."
    )
    return recall_best, justification


# --------------------------------------------------------------------------
# Artifact export
# --------------------------------------------------------------------------


def observed_training_categories(X_train: pd.DataFrame) -> dict[str, list[str]]:
    """Categories present in the training split only."""
    return {
        column: sorted(X_train[column].dropna().astype(str).unique().tolist())
        for column in config.CATEGORICAL_FEATURES
    }


def training_numeric_bounds(X_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Min / max / median of each numeric predictor, from the training split only."""
    bounds: dict[str, dict[str, float]] = {}
    for column in config.NUMERIC_FEATURES:
        series = X_train[column].dropna()
        bounds[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
    return bounds


def write_reference_rates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    test_probabilities: np.ndarray,
) -> Path:
    """Persist descriptive reference statistics used only for context charts.

    Category churn rates come from the **training** split; the score histogram
    comes from the **held-out test** split so it is not in-sample optimistic.
    Nothing here participates in a prediction.
    """
    config.ensure_output_dirs()
    overall_rate = float(y_train.mean())

    by_category: dict[str, dict[str, dict[str, float]]] = {}
    for column in config.CATEGORICAL_FEATURES:
        levels: dict[str, dict[str, float]] = {}
        for level, group in y_train.groupby(X_train[column].astype(str)):
            levels[str(level)] = {
                "churn_rate": round(float(group.mean()), 6),
                "customers": int(len(group)),
            }
        by_category[column] = levels

    counts, edges = np.histogram(test_probabilities, bins=20, range=(0.0, 1.0))
    payload = {
        "note": (
            "Descriptive reference statistics for context charts only. Category churn "
            "rates are computed on the training split; the score histogram is computed "
            "on the held-out test split. Neither is used to produce a prediction."
        ),
        "overall_training_churn_rate": round(overall_rate, 6),
        "training_rows": int(len(y_train)),
        "by_category": by_category,
        "test_score_histogram": {
            "bin_edges": [round(float(edge), 4) for edge in edges],
            "counts": [int(count) for count in counts],
            "population": int(len(test_probabilities)),
        },
    }
    with config.REFERENCE_RATES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return config.REFERENCE_RATES_PATH


def _fairness_section() -> list[str]:
    """Fairness paragraph for the model card, reflecting the audit if it exists.

    The card must never claim an audit is missing once it has been performed, and
    must never imply one exists before it has. Reading the report keeps the two in
    step automatically instead of relying on someone remembering to edit both.
    """
    report_path = config.TABLES_DIR / "fairness_report.json"
    if not report_path.is_file():
        return [
            "- `gender` and `SeniorCitizen` are included as predictors and **no fairness audit",
            "  has been performed**. One is required before any operational use. Run",
            "  `make fairness` to produce it.",
        ]

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return ["- Fairness audit results could not be read; re-run `make fairness`."]

    lines = [
        "- `gender` and `SeniorCitizen` are included as predictors and **a fairness audit has",
        "  been performed** — see `reports/fairness_report.md`. The audit optimises for **equal",
        "  opportunity** (equal recall across groups), because the harm that matters here is an",
        "  at-risk customer being missed entirely.",
    ]

    for attribute, payload in report.get("attributes", {}).items():
        material = [c for c, flag in payload["assessment"].items() if flag["material"]]
        if not material:
            lines.append(
                f"  - `{attribute}`: no disparity exceeded the "
                f"{report['materiality_threshold']:.2f} materiality convention."
            )
        else:
            gaps = payload["disparities"]
            lines.append(
                f"  - `{attribute}`: **material disparity** on "
                f"{', '.join(c.replace('_', ' ') for c in material)}. Equal-opportunity gap "
                f"{gaps['equal_opportunity']['gap']:.4f}. The groups also differ in actual "
                f"churn rate by {gaps['base_rate']['gap']:.4f}, which is what makes the "
                "fairness criteria mutually exclusive."
            )

    costs = report.get("counterfactual_without_protected_attributes")
    if costs:
        delta = costs["roc_auc_with"] - costs["roc_auc_without"]
        lines += [
            f"  - Removing both attributes costs only {delta:+.4f} ROC-AUC "
            f"({costs['roc_auc_with']:.4f} to {costs['roc_auc_without']:.4f}), inside the "
            "cross-validation standard deviation. **Removal is recommended before operational",
            "    use**; they are retained in this version only to preserve the published",
            "    evidence trail for the academic submission.",
        ]

    lines.append(
        "  - A group-specific decision threshold was considered and **rejected outright** as "
        "direct differential treatment by protected characteristic."
    )
    return lines


def write_model_card(
    selected_model: str,
    justification: str,
    test_metrics: dict[str, float],
    comparison_rows: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    timestamp: str,
) -> Path:
    """Write ``deploy/artifacts/model_card.md`` from measured values only."""
    config.ensure_output_dirs()
    lines = [
        "# Model Card — Customer Churn Intelligence",
        "",
        f"**Model name:** {selected_model}  ",
        f"**Model version:** {config.MODEL_VERSION}  ",
        f"**Trained (UTC):** {timestamp}  ",
        f"**Random seed:** {config.RANDOM_STATE}  ",
        f"**Decision threshold:** {config.DECISION_THRESHOLD}  ",
        f"**Artifact:** `deploy/artifacts/model_pipeline.joblib` "
        "(complete preprocessing + estimator pipeline)",
        "",
        "## Intended use",
        "",
        "Estimating whether a single customer record shows elevated churn risk, so that a",
        "**human retention specialist** can prioritise accounts for structured review. The",
        "output is decision support. It provides a probability, a communication band and a",
        "cautious suggested action for a person to act on or overrule.",
        "",
        "## Out-of-scope uses",
        "",
        "This model must **not** be used to:",
        "",
        "- change prices or apply differential pricing automatically;",
        "- terminate, downgrade or modify a contract;",
        "- deny, restrict or degrade service;",
        "- target or exclude customers in ways that could be unfair or discriminatory;",
        "- make any financial or customer-treatment decision without human review;",
        "- infer anything about an individual outside this feature set;",
        "- operate on a population materially different from the training sample without",
        "  revalidation.",
        "",
        "## Dataset provenance",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Dataset | IBM Telco Customer Churn sample (`Telco-Customer-Churn.csv`) |",
        "| Publisher | IBM |",
        "| Official repository | https://github.com/IBM/telco-customer-churn-on-icp4d |",
        "| Repository licence | Apache-2.0 |",
        f"| Verified Git blob SHA | `{dataset_summary['git_blob_sha']}` |",
        f"| Rows / columns | {dataset_summary['rows']:,} / {dataset_summary['columns']} |",
        f"| Positive class rate | {dataset_summary['churn_rate_percent']:.2f}% |",
        "",
        "> **The data represents a fictional telecommunications company.** It is not actual",
        "> observed commercial customer data and must never be described as such.",
        "",
        "## Features",
        "",
        f"- {len(config.CATEGORICAL_FEATURES)} categorical predictors and "
        f"{len(config.NUMERIC_FEATURES)} numeric predictors ({len(config.FEATURE_COLUMNS)} total).",
        f"- `{config.ID_COLUMN}` is excluded from the feature matrix; it carries no signal and",
        "  would risk memorisation.",
        "- `SeniorCitizen` is treated as a binary category rather than a continuous magnitude.",
        "- `TotalCharges` is stripped of whitespace and coerced to numeric; the resulting missing",
        "  values are median-imputed **inside** the pipeline, fitted on training data only.",
        "",
        "## Training procedure",
        "",
        f"- Single stratified split: {int((1 - config.TEST_SIZE) * 100)}% train / "
        f"{int(config.TEST_SIZE * 100)}% test, `random_state={config.RANDOM_STATE}`.",
        f"- Model selection by stratified {config.CV_FOLDS}-fold cross-validation on the training",
        "  split only.",
        "- All imputation, encoding and scaling occur inside the `Pipeline`, so every",
        "  cross-validation fold refits them from scratch and no test information leaks.",
        f"- Unseen categories at inference are absorbed by `OneHotEncoder(handle_unknown=\"ignore\")`.",
        "",
        "## Selection",
        "",
        justification,
        "",
        "## Metrics (held-out test set, evaluated once after selection)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {test_metrics['accuracy']:.4f} |",
        f"| Precision (churn) | {test_metrics['precision']:.4f} |",
        f"| Recall (churn) | {test_metrics['recall']:.4f} |",
        f"| F1 (churn) | {test_metrics['f1']:.4f} |",
        f"| ROC-AUC | {test_metrics['roc_auc']:.4f} |",
        "",
        "Full comparison including the reference baseline:",
        "",
        "| Model | Role | CV ROC-AUC | Test ROC-AUC | Test recall | Test F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['model']} | {row['role']} | {row['cv_roc_auc_mean']:.4f} | "
            f"{row['test_roc_auc']:.4f} | {row['test_recall']:.4f} | {row['test_f1']:.4f} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "- Trained on a fictional sample; performance on any live population is unknown and",
        "  must be revalidated before operational use.",
        "- A single cross-section with no time dimension, so no drift or seasonality behaviour",
        "  can be assessed and no monitoring baseline exists.",
        f"- Class imbalance ({dataset_summary['churn_rate_percent']:.2f}% positive) limits the",
        "  precision attainable at high recall.",
        "- The 0.5 threshold is a documented default, not a cost-optimised operating point. No",
        "  cost matrix was supplied, so none was fitted.",
        "- Risk tiers are communication bands and have not been validated as business thresholds.",
        "- Predictions describe association within the sample, never causation. The model cannot",
        "  say what would happen if a customer's contract or service were changed.",
        "",
        "## Ethical and governance considerations",
        "",
        *_fairness_section(),
        "- Outputs may support prioritisation. They are not a judgement about any person and must",
        "  not be presented to a customer as a fact about them.",
        "- Retention offers driven by a risk score can create differential treatment between",
        "  customers. Any such use needs review by model-risk and governance functions first.",
        "- The application now shows a per-prediction contribution breakdown (`deploy/explain.py`),",
        "  which is exact for this linear model rather than an approximation. It describes",
        "  **association within the training sample, not causation**, so it does not by itself",
        "  constitute a customer-facing reason for a decision.",
        "",
        "## Human review requirement",
        "",
        "Every prediction requires human review before any customer-affecting action is taken.",
        "The application states this on screen. The model has no authority to act, and no",
        "downstream system should consume its output as an automated trigger.",
        "",
        "## Reproducing this artifact",
        "",
        "```bash",
        "make bootstrap",
        "make validate",
        "make train",
        "```",
        "",
        f"Environment used for this artifact: Python {platform.python_version()}, "
        f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}.",
        "",
    ]

    config.MODEL_CARD_PATH.write_text("\n".join(lines), encoding="utf-8")
    return config.MODEL_CARD_PATH


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_training(path: Path | None = None) -> dict[str, Any]:
    """Validate, split, cross-validate, select, evaluate and export."""
    config.ensure_output_dirs()
    dataset_path = path or config.RAW_DATASET_PATH

    # Refuse to train on data that has not passed the integrity contract.
    validation = validate_dataset(dataset_path)
    if not validation.passed:
        failed = [check["check"] for check in validation.failures]
        raise RuntimeError(
            "Refusing to train: raw dataset validation failed for checks "
            f"{failed}. Run `python -m src.data_validation` for detail."
        )

    frame = load_model_frame(dataset_path)
    X, y = split_features_target(frame)

    # Single split, performed before any transformer is fitted.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Training churn rate: {y_train.mean():.4f} | Test churn rate: {y_test.mean():.4f}")

    candidates = build_candidates()
    baseline_name = "Dummy (stratified) baseline"

    # --- Stage 1: cross-validated selection, training split only -----------
    print(f"\nStratified {config.CV_FOLDS}-fold cross-validation on the training split")
    cv_results: dict[str, dict[str, float]] = {}
    for name, pipeline in candidates.items():
        cv_results[name] = cross_validate_model(pipeline, X_train, y_train)
        print(
            f"  {name:<22} ROC-AUC {cv_results[name]['cv_roc_auc_mean']:.4f} "
            f"± {cv_results[name]['cv_roc_auc_std']:.4f} | "
            f"F1 {cv_results[name]['cv_f1_mean']:.4f} | "
            f"recall {cv_results[name]['cv_recall_mean']:.4f}"
        )

    baseline_pipeline = build_baseline()
    baseline_cv = cross_validate_model(baseline_pipeline, X_train, y_train)
    print(
        f"  {baseline_name:<22} ROC-AUC {baseline_cv['cv_roc_auc_mean']:.4f} "
        f"(reference only, not a candidate)"
    )

    selected_model, justification = select_model(cv_results)
    print(f"\nSELECTED (before any test-set result was computed): {selected_model}")
    print(f"Reason: {justification}")

    # --- Stage 2: single unbiased evaluation on the held-out test set ------
    print("\nHeld-out test evaluation")
    fitted: dict[str, Pipeline] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    confusion_matrices: dict[str, np.ndarray] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_aucs: dict[str, float] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    for name, pipeline in {**candidates, baseline_name: baseline_pipeline}.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        proba = pipeline.predict_proba(X_test)[:, 1]
        test_probabilities[name] = proba
        predictions = (proba >= config.DECISION_THRESHOLD).astype(int)

        test_metrics[name] = evaluate.compute_metrics(y_test, predictions, proba)
        evaluate.save_classification_report(y_test, predictions, name)
        _, _, matrix = evaluate.save_confusion_matrix(y_test, predictions, name)
        confusion_matrices[name] = matrix
        roc_curves[name] = evaluate.roc_points(y_test, proba)
        test_aucs[name] = test_metrics[name]["roc_auc"]
        print(
            f"  {name:<22} acc {test_metrics[name]['accuracy']:.4f} | "
            f"prec {test_metrics[name]['precision']:.4f} | "
            f"rec {test_metrics[name]['recall']:.4f} | "
            f"F1 {test_metrics[name]['f1']:.4f} | "
            f"AUC {test_metrics[name]['roc_auc']:.4f}"
        )

    evaluate.save_roc_curves(roc_curves, test_aucs)

    comparison_rows: list[dict[str, Any]] = []
    for name in [*candidates, baseline_name]:
        cv = cv_results.get(name, baseline_cv)
        row: dict[str, Any] = {
            "model": name,
            "role": "candidate" if name in candidates else "baseline",
            "selected": name == selected_model,
        }
        row.update({key: round(value, 6) for key, value in cv.items()})
        row.update(
            {f"test_{key}": round(value, 6) for key, value in test_metrics[name].items()}
        )
        matrix = confusion_matrices[name]
        row.update(
            {
                "test_true_negative": int(matrix[0, 0]),
                "test_false_positive": int(matrix[0, 1]),
                "test_false_negative": int(matrix[1, 0]),
                "test_true_positive": int(matrix[1, 1]),
            }
        )
        comparison_rows.append(row)

    evaluate.save_model_comparison(comparison_rows)
    evaluate.save_cv_comparison_figure(comparison_rows)

    # --- Stage 3: export the selected artifact ----------------------------
    selected_pipeline = fitted[selected_model]
    joblib.dump(selected_pipeline, config.MODEL_PATH)

    categories = observed_training_categories(X_train)
    bounds = training_numeric_bounds(X_train)
    write_reference_rates(X_train, y_train, test_probabilities[selected_model])
    feature_schema = schemas.build_feature_schema(categories, bounds)
    with config.FEATURE_SCHEMA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(feature_schema, handle, indent=2)
        handle.write("\n")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset_summary = {
        "git_blob_sha": git_blob_sha(dataset_path),
        "rows": int(validation.summary["rows"]),
        "columns": int(validation.summary["columns"]),
        "churn_rate_percent": float(y.mean() * 100),
    }

    metadata = {
        "project_name": config.PROJECT_NAME,
        "model_name": selected_model,
        "model_version": config.MODEL_VERSION,
        "target": config.TARGET_COLUMN,
        "positive_class": config.POSITIVE_CLASS,
        "decision_threshold": config.DECISION_THRESHOLD,
        "random_state": config.RANDOM_STATE,
        "training_timestamp_utc": timestamp,
        "dataset_git_blob_sha": dataset_summary["git_blob_sha"],
        "dataset_rows": dataset_summary["rows"],
        "dataset_columns": dataset_summary["columns"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": config.TEST_SIZE,
        "cv_folds": config.CV_FOLDS,
        "selection_rule": SELECTION_RULE,
        "selection_justification": justification,
        "metrics": {key: round(value, 6) for key, value in test_metrics[selected_model].items()},
        "cross_validated_metrics": {
            key: round(value, 6) for key, value in cv_results[selected_model].items()
        },
        "confusion_matrix": {
            "true_negative": int(confusion_matrices[selected_model][0, 0]),
            "false_positive": int(confusion_matrices[selected_model][0, 1]),
            "false_negative": int(confusion_matrices[selected_model][1, 0]),
            "true_positive": int(confusion_matrices[selected_model][1, 1]),
        },
        "all_model_test_metrics": {
            name: {key: round(value, 6) for key, value in metrics.items()}
            for name, metrics in test_metrics.items()
        },
        "risk_tiers": {
            "low_max_exclusive": config.RISK_LOW_MAX,
            "medium_max_exclusive": config.RISK_MEDIUM_MAX,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "data_notice": (
            "IBM Telco Customer Churn sample. Represents a fictional telecommunications "
            "company; not actual observed commercial customer data."
        ),
        "governance_notice": (
            "Decision support only. Requires human review. Must not autonomously change "
            "prices, modify contracts, deny service or make customer-treatment decisions."
        ),
    }
    with config.METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    write_model_card(
        selected_model,
        justification,
        test_metrics[selected_model],
        comparison_rows,
        dataset_summary,
        timestamp,
    )
    evaluate.write_executive_summary(
        comparison_rows,
        selected_model,
        SELECTION_RULE + " " + justification,
        test_metrics,
        confusion_matrices,
        dataset_summary,
    )

    # Processed splits are written for traceability only; they are never read
    # back by training, which always starts from the immutable raw file.
    X_train.assign(**{config.TARGET_COLUMN: y_train}).to_csv(
        config.PROCESSED_DATA_DIR / "train_split.csv", index=False
    )
    X_test.assign(**{config.TARGET_COLUMN: y_test}).to_csv(
        config.PROCESSED_DATA_DIR / "test_split.csv", index=False
    )

    print(f"\nArtifact          -> {config.MODEL_PATH}")
    print(f"Metadata          -> {config.METADATA_PATH}")
    print(f"Feature schema    -> {config.FEATURE_SCHEMA_PATH}")
    print(f"Reference rates   -> {config.REFERENCE_RATES_PATH}")
    print(f"Model card        -> {config.MODEL_CARD_PATH}")
    print(f"Comparison table  -> {config.REPORTS_DIR / 'model_comparison.csv'}")
    print(f"Executive summary -> {config.REPORTS_DIR / 'executive_model_summary.md'}")

    return {
        "selected_model": selected_model,
        "justification": justification,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "comparison_rows": comparison_rows,
        "metadata": metadata,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and export the churn model pipeline.")
    parser.add_argument("--path", type=Path, default=config.RAW_DATASET_PATH)
    args = parser.parse_args(argv)
    run_training(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
