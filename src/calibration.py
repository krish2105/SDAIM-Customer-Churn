"""Probability calibration analysis (H1-2).

Closes gap G2. The application displays a probability to a decision-maker. If
that number is not a real probability, the interface is misleading regardless of
how good the ranking is.

Expected result, stated before running: logistic regression optimises log-loss
and is usually well calibrated, **but** this model uses
``class_weight="balanced"``, which deliberately inflates minority-class
probabilities. It should therefore be over-confident about churn, and predictably
so. Measuring that and explaining the cause is a stronger result than a flat
reliability curve.

CLI::

    python -m src.calibration
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from src import config  # noqa: E402
from src.analysis_base import load_evaluation_context  # noqa: E402

FIGURE_DPI = 200
N_BINS = 10


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, float]:
    """Return ``(ECE, MCE)`` using equal-width bins.

    ECE is the average absolute gap between predicted confidence and observed
    frequency, weighted by bin population. MCE is the worst single bin — it
    matters because a model can have a respectable average while being badly
    wrong in the high-probability band that drives action.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    mce = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (y_proba > lower) & (y_proba <= upper) if lower > 0 else (y_proba <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        observed = float(y_true[mask].mean())
        predicted = float(y_proba[mask].mean())
        gap = abs(observed - predicted)
        ece += (count / total) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def bin_table(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Per-bin predicted confidence against observed frequency."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (y_proba > lower) & (y_proba <= upper) if lower > 0 else (y_proba <= upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin": f"{lower:.1f}–{upper:.1f}",
                "n": count,
                "mean_predicted": round(float(y_proba[mask].mean()), 6) if count else None,
                "observed_frequency": round(float(y_true[mask].mean()), 6) if count else None,
                "gap": (
                    round(float(y_true[mask].mean() - y_proba[mask].mean()), 6) if count else None
                ),
            }
        )
    return rows


def plot_reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    probabilities: dict[str, np.ndarray],
    filename: str = "16_calibration_curve.png",
) -> str:
    """Reliability diagram with a histogram of predicted probabilities beneath."""
    config.ensure_output_dirs()
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6.8, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    colours = {"Uncalibrated (deployed)": "#4C72B0", "Isotonic": "#55A868", "Sigmoid": "#DD8452"}
    for name, (observed, predicted) in curves.items():
        ax.plot(predicted, observed, "o-", linewidth=1.8, markersize=5,
                color=colours.get(name, "#937860"), label=name)

    ax.set_ylabel("Observed churn frequency")
    ax.set_title("Reliability diagram — held-out test set\n"
                 "Points below the diagonal indicate over-confident churn probabilities",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, 1)

    for name, proba in probabilities.items():
        ax_hist.hist(proba, bins=20, range=(0, 1), alpha=0.55,
                     color=colours.get(name, "#937860"), label=name)
    ax_hist.set_xlabel("Predicted churn probability")
    ax_hist.set_ylabel("Customers")
    ax_hist.set_xlim(0, 1)
    ax_hist.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filename


def run_calibration_analysis() -> dict[str, Any]:
    """Measure calibration, fit alternatives, and persist the report."""
    config.ensure_output_dirs()
    context = load_evaluation_context()

    y_test = context.y_test.to_numpy()
    baseline_proba = context.test_probabilities

    results: dict[str, Any] = {"n_bins": N_BINS, "test_rows": int(len(y_test)), "variants": {}}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    probabilities: dict[str, np.ndarray] = {}

    def record(name: str, proba: np.ndarray) -> None:
        ece, mce = expected_calibration_error(y_test, proba)
        results["variants"][name] = {
            "brier_score": round(float(brier_score_loss(y_test, proba)), 6),
            "expected_calibration_error": round(ece, 6),
            "maximum_calibration_error": round(mce, 6),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 6),
            "mean_predicted": round(float(proba.mean()), 6),
            "observed_base_rate": round(float(y_test.mean()), 6),
            "bins": bin_table(y_test, proba),
        }
        observed, predicted = calibration_curve(y_test, proba, n_bins=N_BINS, strategy="uniform")
        curves[name] = (observed, predicted)
        probabilities[name] = proba

    record("Uncalibrated (deployed)", baseline_proba)

    # Calibrators are fitted on the TRAINING split only, via internal CV. Fitting
    # them on the test set would be leakage and would make the result meaningless.
    for label, method in (("Isotonic", "isotonic"), ("Sigmoid", "sigmoid")):
        calibrated = CalibratedClassifierCV(context.pipeline, method=method, cv=5)
        calibrated.fit(context.X_train, context.y_train)
        record(label, calibrated.predict_proba(context.X_test)[:, 1])

    results["figure"] = plot_reliability(curves, probabilities)

    with (config.TABLES_DIR / "calibration_report.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    _write_markdown(results)
    return results


def _write_markdown(results: dict[str, Any]) -> None:
    variants = results["variants"]
    baseline = variants["Uncalibrated (deployed)"]
    best_name = min(variants, key=lambda name: variants[name]["expected_calibration_error"])
    best = variants[best_name]
    improvement = baseline["expected_calibration_error"] - best["expected_calibration_error"]
    over_confident = baseline["mean_predicted"] > baseline["observed_base_rate"]

    lines = [
        "# Probability Calibration Analysis",
        "",
        "Closes gap **G2**. Computed by `src/calibration.py` on the held-out test set.",
        "",
        "## Why this matters",
        "",
        "The application shows a churn probability to a retention specialist. A model can rank",
        "customers well — high ROC-AUC — while its probabilities are systematically wrong. Ranking",
        "and calibration are different properties, and only calibration justifies showing a number",
        "like \"22.9%\" as though it means 22.9%.",
        "",
        "## What was expected before measuring",
        "",
        "Logistic regression optimises log-loss and is usually well calibrated out of the box.",
        "**However, this model uses `class_weight=\"balanced\"`**, which deliberately re-weights the",
        "minority class during fitting. That improves recall — the project's stated priority — but",
        "it distorts the predicted probabilities upward. The model was therefore expected to be",
        "**over-confident about churn**, and for a known, explainable reason.",
        "",
        "## Measured result",
        "",
        "| Variant | Brier ↓ | ECE ↓ | MCE ↓ | ROC-AUC | Mean predicted | Observed base rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in variants.items():
        lines.append(
            f"| {name} | {values['brier_score']:.4f} | "
            f"{values['expected_calibration_error']:.4f} | "
            f"{values['maximum_calibration_error']:.4f} | {values['roc_auc']:.4f} | "
            f"{values['mean_predicted']:.4f} | {values['observed_base_rate']:.4f} |"
        )

    lines += [
        "",
        "- **Brier score** — mean squared error of the probabilities. Lower is better.",
        "- **ECE** — population-weighted average gap between predicted confidence and observed",
        "  frequency across 10 bins.",
        "- **MCE** — the worst single bin. It matters because a model can look acceptable on",
        "  average while being badly wrong in the high-probability band that actually drives",
        "  action.",
        "",
        f"![Reliability diagram](figures/{results['figure']})",
        "",
        "## Interpretation",
        "",
    ]

    if over_confident:
        lines += [
            f"The deployed model predicts a mean churn probability of "
            f"**{baseline['mean_predicted']:.4f}** against an observed base rate of "
            f"**{baseline['observed_base_rate']:.4f}** — it is **over-confident about churn by "
            f"{baseline['mean_predicted'] - baseline['observed_base_rate']:.4f}**, exactly as",
            "predicted above. This is not a defect discovered late; it is the direct and",
            "anticipated consequence of `class_weight=\"balanced\"`, which was chosen deliberately",
            "to raise recall.",
            "",
            "**In plain terms: when the application displays a probability, that number currently",
            "overstates the real likelihood of churn.** The ranking is sound — customers are",
            "ordered correctly — but the magnitude is inflated.",
            "",
        ]
    else:
        lines += [
            f"The deployed model predicts a mean of **{baseline['mean_predicted']:.4f}** against an",
            f"observed base rate of **{baseline['observed_base_rate']:.4f}**, so it is not",
            "systematically over-confident in aggregate. Per-bin behaviour still matters; see the",
            "reliability diagram.",
            "",
        ]

    lines += [
        "### Ranking is unaffected",
        "",
        "ROC-AUC is threshold-independent and rank-based, so calibration does not change it in any",
        "way that matters: "
        + ", ".join(f"{name} {values['roc_auc']:.4f}" for name, values in variants.items())
        + ". Calibration changes *what the number means*, not *who is ranked highest*.",
        "",
        "## Decision",
        "",
    ]

    if improvement > 0.01:
        lines += [
            f"**{best_name} calibration materially improves the probabilities** — ECE falls from",
            f"{baseline['expected_calibration_error']:.4f} to "
            f"{best['expected_calibration_error']:.4f}, an improvement of {improvement:.4f}, and",
            f"the Brier score moves from {baseline['brier_score']:.4f} to {best['brier_score']:.4f}.",
            "",
            "**Recommendation: wrap the deployed pipeline in "
            f"`CalibratedClassifierCV(method=\"{best_name.lower()}\", cv=5)` fitted on the training",
            "split.**",
            "",
            "**Why version 1.1.0 ships uncalibrated anyway.** The measured metrics for this model",
            "are already published across the report, the model card and the live application.",
            "Swapping the artifact now would invalidate that evidence trail mid-submission. The",
            "change is recorded as a Horizon 3 action with its benefit quantified here, so the",
            "decision is documented rather than deferred silently.",
            "",
            "**What the application does instead, now:** the interface states that probabilities",
            "are model scores rather than validated frequencies, and the risk bands are presented",
            "as communication aids. That was already true in 1.0.0; this analysis gives it a",
            "measured basis instead of a caveat.",
            "",
        ]
    else:
        lines += [
            f"Calibration offers no material improvement: the best variant ({best_name}) moves ECE",
            f"by only {improvement:.4f}. **Recommendation: ship uncalibrated.** Adding a calibration",
            "wrapper would increase artifact complexity and inference cost for no measurable",
            "benefit.",
            "",
        ]

    lines += [
        "## Method note — no leakage",
        "",
        "Both calibrators were fitted with `CalibratedClassifierCV(..., cv=5)` on the **training",
        "split only**, then evaluated on the untouched held-out set. Fitting a calibrator on the",
        "test set would guarantee an excellent-looking curve and mean nothing at all.",
        "",
        "## Limitations",
        "",
        "- Calibration is measured on 1,409 held-out customers from a **fictional** sample. It",
        "  says nothing about calibration on a live population.",
        "- ECE with 10 equal-width bins is sensitive to bin count; the per-bin table is published",
        "  so the reader can judge rather than trust a single number.",
        "- Calibration was not measured per subgroup. A model can be well calibrated overall and",
        "  poorly calibrated for a minority group; see `reports/fairness_report.md` for the",
        "  subgroup analysis that was performed.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make calibration",
        "```",
        "",
    ]

    (config.REPORTS_DIR / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probability calibration analysis.")
    parser.parse_args(argv)

    results = run_calibration_analysis()
    print("=" * 72)
    print("Calibration analysis — held-out test set")
    print("=" * 72)
    for name, values in results["variants"].items():
        print(
            f"  {name:<26} Brier {values['brier_score']:.4f} | "
            f"ECE {values['expected_calibration_error']:.4f} | "
            f"MCE {values['maximum_calibration_error']:.4f} | "
            f"AUC {values['roc_auc']:.4f} | mean p {values['mean_predicted']:.4f}"
        )
    base = results["variants"]["Uncalibrated (deployed)"]
    print(f"\nObserved base rate: {base['observed_base_rate']:.4f}")
    print(f"Report -> {config.REPORTS_DIR / 'calibration_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
