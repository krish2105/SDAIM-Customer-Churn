"""Evaluation metrics, figures and the executive model summary.

Kept separate from ``src/train.py`` so evaluation can be re-run against a saved
artifact without retraining, and so the tests can exercise metric computation in
isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config  # noqa: E402

FIGURE_DPI = 200


def slugify(name: str) -> str:
    """Filesystem-safe identifier for a model name."""
    return "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Headline classification metrics for the positive (churn) class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_classification_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str,
) -> Path:
    """Write the per-class classification report as JSON."""
    config.ensure_output_dirs()
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Retained (0)", "Churn (1)"],
        output_dict=True,
        zero_division=0,
    )
    path = config.TABLES_DIR / f"classification_report_{slugify(model_name)}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return path


def save_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str,
) -> tuple[Path, Path, np.ndarray]:
    """Persist the confusion matrix as CSV and as a labelled figure."""
    config.ensure_output_dirs()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    frame = pd.DataFrame(
        matrix,
        index=["actual_retained", "actual_churn"],
        columns=["predicted_retained", "predicted_churn"],
    )
    csv_path = config.TABLES_DIR / f"confusion_matrix_{slugify(model_name)}.csv"
    frame.to_csv(csv_path)

    labels = np.array(
        [
            ["True negative", "False positive"],
            ["False negative", "True positive"],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted: retained", "Predicted: churn"])
    ax.set_yticks([0, 1], ["Actual: retained", "Actual: churn"])
    threshold = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}\n{labels[i, j]}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if matrix[i, j] > threshold else "black",
            )
    ax.set_title(f"Confusion matrix — {model_name}\n(held-out test set)")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Customers")
    figure_path = config.FIGURES_DIR / f"12_confusion_matrix_{slugify(model_name)}.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return csv_path, figure_path, matrix


def save_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    aucs: dict[str, float],
    filename: str = "13_roc_curves.png",
) -> Path:
    """Plot every model's ROC curve on shared axes."""
    config.ensure_output_dirs()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{name} (AUC = {aucs[name]:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves — held-out test set")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_points(y_true: np.ndarray | pd.Series, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return fpr, tpr


def save_model_comparison(rows: list[dict[str, Any]]) -> Path:
    """Write the cross-validated and held-out comparison table."""
    config.ensure_output_dirs()
    frame = pd.DataFrame(rows)
    path = config.REPORTS_DIR / "model_comparison.csv"
    frame.to_csv(path, index=False)
    return path


def save_cv_comparison_figure(rows: list[dict[str, Any]]) -> Path:
    """Bar chart of cross-validated ROC-AUC and F1 with standard-deviation bars."""
    config.ensure_output_dirs()
    substantive = [row for row in rows if row["role"] == "candidate"]
    names = [row["model"] for row in substantive]
    positions = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(
        positions - width / 2,
        [row["cv_roc_auc_mean"] for row in substantive],
        width,
        yerr=[row["cv_roc_auc_std"] for row in substantive],
        capsize=4,
        label="CV ROC-AUC",
        color="#4C72B0",
    )
    ax.bar(
        positions + width / 2,
        [row["cv_f1_mean"] for row in substantive],
        width,
        yerr=[row["cv_f1_std"] for row in substantive],
        capsize=4,
        label="CV F1",
        color="#DD8452",
    )
    ax.set_xticks(positions, names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Cross-validated model comparison ({config.CV_FOLDS}-fold, training split only)")
    ax.legend()
    path = config.FIGURES_DIR / "14_cv_model_comparison.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def write_executive_summary(
    comparison_rows: list[dict[str, Any]],
    selected_model: str,
    selection_rule: str,
    test_metrics: dict[str, dict[str, float]],
    confusion_matrices: dict[str, np.ndarray],
    dataset_summary: dict[str, Any],
) -> Path:
    """Write ``reports/executive_model_summary.md`` from measured values only."""
    config.ensure_output_dirs()
    selected = test_metrics[selected_model]
    matrix = confusion_matrices[selected_model]
    tn, fp, fn, tp = int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])
    test_total = tn + fp + fn + tp
    actual_churn = fn + tp
    flagged = fp + tp

    baseline = next((row for row in comparison_rows if row["role"] == "baseline"), None)

    lines = [
        "# Executive Model Summary",
        "",
        f"**Project:** {config.PROJECT_NAME}  ",
        f"**Model version:** {config.MODEL_VERSION}  ",
        f"**Selected model:** {selected_model}  ",
        f"**Decision threshold:** {config.DECISION_THRESHOLD}  ",
        f"**Random seed:** {config.RANDOM_STATE}",
        "",
        "> The underlying data is IBM's **fictional** telecommunications sample. Results",
        "> demonstrate technical feasibility and may support prioritisation of accounts for",
        "> human review. No revenue, churn-reduction or ROI effect has been measured, and",
        "> none is claimed.",
        "",
        "## 1. What this model does",
        "",
        "For one customer record it returns a churn class, a churn probability, a Low / Medium /",
        "High communication band, and a cautious suggested review action. It is **decision",
        "support for a human retention specialist**, not an autonomous decision system.",
        "",
        "## 2. Selection rule (fixed before the test set was examined)",
        "",
        f"{selection_rule}",
        "",
        "## 3. Model comparison",
        "",
        "Cross-validation is computed on the training split only; test columns come from the",
        "held-out 20% that no model saw during fitting or selection.",
        "",
        "| Model | Role | CV ROC-AUC | CV F1 | CV recall | Test ROC-AUC | Test F1 | Test recall | Test precision | Test accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['model']} | {row['role']} | "
            f"{row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f} | "
            f"{row['cv_f1_mean']:.4f} | {row['cv_recall_mean']:.4f} | "
            f"{row['test_roc_auc']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | "
            f"{row['test_precision']:.4f} | {row['test_accuracy']:.4f} |"
        )

    lines += [
        "",
        "## 4. Held-out performance of the selected model",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {selected['accuracy']:.4f} |",
        f"| Precision (churn) | {selected['precision']:.4f} |",
        f"| Recall (churn) | {selected['recall']:.4f} |",
        f"| F1 (churn) | {selected['f1']:.4f} |",
        f"| ROC-AUC | {selected['roc_auc']:.4f} |",
        "",
        "Confusion matrix on the held-out test set:",
        "",
        "| | Predicted: retained | Predicted: churn |",
        "|---|---:|---:|",
        f"| **Actual: retained** | {tn:,} (true negative) | {fp:,} (false positive) |",
        f"| **Actual: churn** | {fn:,} (false negative) | {tp:,} (true positive) |",
        "",
        "## 5. What the errors mean for the business",
        "",
        f"- **False negatives ({fn:,} of {actual_churn:,} actual churners, "
        f"{fn / actual_churn * 100:.1f}%)** — customers who churned but were not flagged. These are",
        "  the costly errors: the account receives no retention review at all, so the opportunity",
        "  to intervene is lost silently. This is why recall is weighted heavily in selection.",
        f"- **False positives ({fp:,} of {flagged:,} flagged accounts, "
        f"{fp / flagged * 100:.1f}%)** — customers flagged for review who would have stayed. The cost",
        "  is retention-specialist time and the risk of an unnecessary contact, which is recoverable.",
        f"- The model flags {flagged:,} of {test_total:,} test customers for review "
        f"({flagged / test_total * 100:.1f}%), giving retention teams a bounded workload.",
        "",
        "## 6. Why accuracy alone is insufficient",
        "",
        f"The sample contains {dataset_summary['churn_rate_percent']:.2f}% churners. A model that",
        f"always predicted \"retained\" would score {100 - dataset_summary['churn_rate_percent']:.2f}%",
        "accuracy while finding no churn risk whatsoever and providing zero business value.",
    ]
    if baseline is not None:
        lines.append(
            f"The `{baseline['model']}` baseline confirms this: it reaches "
            f"{baseline['test_accuracy']:.4f} test accuracy with {baseline['test_recall']:.4f} recall "
            f"and {baseline['test_roc_auc']:.4f} ROC-AUC."
        )
    lines += [
        "",
        "Recall, F1 and ROC-AUC are therefore the metrics that govern selection, because they",
        "measure whether at-risk customers are actually identified.",
        "",
        "## 7. Risk tiers",
        "",
        "| Tier | Probability | Suggested handling |",
        "|---|---|---|",
        f"| Low | p < {config.RISK_LOW_MAX:.2f} | No immediate action indicated; monitor in routine reporting. |",
        f"| Medium | {config.RISK_LOW_MAX:.2f} ≤ p < {config.RISK_MEDIUM_MAX:.2f} | May support inclusion in a periodic review queue. |",
        f"| High | p ≥ {config.RISK_MEDIUM_MAX:.2f} | May support prioritisation for structured human review. |",
        "",
        "These bands are **communication aids for triage only**. They have not been independently",
        "validated as business thresholds, and no cost-sensitive optimisation was performed.",
        "",
        "## 8. Governance",
        "",
        "The model must not autonomously change prices, terminate or modify contracts, deny",
        "service, target customers unfairly, or make any financial or customer-treatment decision.",
        "Every output requires human review before any customer is contacted.",
        "",
        "## 9. Limitations",
        "",
        "- The dataset is a fictional IBM sample; performance on any live population is unknown.",
        "- It is a single cross-section with no time dimension, so no drift behaviour can be assessed.",
        f"- The class balance ({dataset_summary['churn_rate_percent']:.2f}% positive) limits precision",
        "  attainable at high recall.",
        "- `gender` and `SeniorCitizen` are included as predictors; no formal fairness audit across",
        "  these attributes has been carried out, and one is recommended before any operational use.",
        "- The 0.5 decision threshold is a documented default, not a business-optimised operating point.",
        "",
    ]

    path = config.REPORTS_DIR / "executive_model_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
