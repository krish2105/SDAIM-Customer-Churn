"""Cost-sensitive decision-threshold analysis (H1-3).

Closes gap G3. Version 1.0.0 used a 0.50 threshold with no justification beyond
"it is the default", which is not an answer.

Real intervention costs were never supplied, and inventing them would be exactly
the kind of fabricated business claim this project avoids everywhere else. The
professionally correct response is therefore not to pick a number but to publish
the **sensitivity curve**: given a ratio between the cost of missing a churner
and the cost of an unnecessary review, the optimal threshold follows
deterministically. The business supplies the ratio; the model supplies the curve.

CLI::

    python -m src.threshold
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402

from src import config  # noqa: E402
from src.analysis_base import EvaluationContext, load_evaluation_context  # noqa: E402

FIGURE_DPI = 200

#: Cost ratios swept: how many unnecessary reviews one missed churner is worth.
COST_RATIOS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)

#: Threshold grid. 0.01 steps are fine enough to locate the optimum without
#: implying more precision than 1,409 test rows can support.
THRESHOLD_GRID: np.ndarray = np.round(np.arange(0.05, 0.96, 0.01), 2)


def sweep_thresholds(context: EvaluationContext) -> list[dict[str, Any]]:
    """Confusion counts and metrics at every threshold on the grid."""
    proba = context.test_probabilities
    y_true = context.y_test.to_numpy()

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        predictions = (proba >= threshold).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]),
                          int(matrix[1, 0]), int(matrix[1, 1]))
        rows.append(
            {
                "threshold": float(threshold),
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "flagged": fp + tp,
                "flagged_share": round((fp + tp) / len(y_true), 6),
                "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
                "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
                "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
            }
        )
    return rows


def optimal_thresholds(rows: list[dict[str, Any]], ratios=COST_RATIOS) -> list[dict[str, Any]]:
    """Cost-minimising threshold for each cost ratio.

    Cost is expressed in units of one unnecessary review, so ``C_review = 1`` and
    ``C_miss = ratio``. Only the ratio matters, not the absolute currency values —
    which is precisely why no currency needs to be invented.
    """
    results: list[dict[str, Any]] = []
    for ratio in ratios:
        costs = [row["false_negative"] * ratio + row["false_positive"] for row in rows]
        best_index = int(np.argmin(costs))
        best = rows[best_index]
        results.append(
            {
                "cost_ratio": ratio,
                "optimal_threshold": best["threshold"],
                "total_cost_review_units": round(float(costs[best_index]), 2),
                "recall": best["recall"],
                "precision": best["precision"],
                "f1": best["f1"],
                "flagged": best["flagged"],
                "flagged_share": best["flagged_share"],
                "false_negative": best["false_negative"],
                "false_positive": best["false_positive"],
            }
        )
    return results


def plot_analysis(rows: list[dict[str, Any]], optima: list[dict[str, Any]]) -> str:
    """Two panels: metric trade-off, and optimal threshold against cost ratio."""
    config.ensure_output_dirs()
    thresholds = [row["threshold"] for row in rows]

    fig, (ax_metrics, ax_cost) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- Panel 1: how the trade-off moves with the threshold ---------------
    ax_metrics.set_axisbelow(True)
    ax_metrics.plot(thresholds, [r["recall"] for r in rows], linewidth=2,
                    color="#C44E52", label="Recall (churners found)")
    ax_metrics.plot(thresholds, [r["precision"] for r in rows], linewidth=2,
                    color="#4C72B0", label="Precision (flags that are right)")
    ax_metrics.plot(thresholds, [r["f1"] for r in rows], linewidth=1.6,
                    color="#55A868", linestyle="--", label="F1")
    ax_metrics.plot(thresholds, [r["flagged_share"] for r in rows], linewidth=1.4,
                    color="#937860", linestyle=":", label="Share of book flagged")
    ax_metrics.axvline(config.DECISION_THRESHOLD, color="#8C8C8C", linewidth=1.2,
                       label=f"Deployed default ({config.DECISION_THRESHOLD:.2f})")
    ax_metrics.set_xlabel("Decision threshold")
    ax_metrics.set_ylabel("Rate")
    ax_metrics.set_title("Threshold trade-off — held-out test set", fontsize=10, loc="left")
    ax_metrics.legend(fontsize=8)
    ax_metrics.set_xlim(min(thresholds), max(thresholds))
    ax_metrics.set_ylim(0, 1.02)

    # --- Panel 2: optimum as a function of the cost ratio -----------------
    ax_cost.set_axisbelow(True)
    ratios = [o["cost_ratio"] for o in optima]
    best = [o["optimal_threshold"] for o in optima]
    ax_cost.plot(ratios, best, "o-", linewidth=2, markersize=6, color="#4C72B0")
    for opt in optima:
        ax_cost.annotate(f"{opt['optimal_threshold']:.2f}",
                         (opt["cost_ratio"], opt["optimal_threshold"]),
                         textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax_cost.axhline(config.DECISION_THRESHOLD, color="#8C8C8C", linestyle="--", linewidth=1.2,
                    label=f"Deployed default ({config.DECISION_THRESHOLD:.2f})")
    ax_cost.set_xlabel("Cost ratio — unnecessary reviews per missed churner")
    ax_cost.set_ylabel("Cost-minimising threshold")
    ax_cost.set_title("Optimal threshold is a business input, not a modelling output",
                      fontsize=10, loc="left")
    ax_cost.set_ylim(0, 1)
    ax_cost.legend(fontsize=8)

    fig.tight_layout()
    filename = "17_threshold_analysis.png"
    fig.savefig(config.FIGURES_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filename


def run_threshold_analysis() -> dict[str, Any]:
    """Sweep, optimise, plot and persist."""
    config.ensure_output_dirs()
    context = load_evaluation_context()

    rows = sweep_thresholds(context)
    optima = optimal_thresholds(rows)
    figure = plot_analysis(rows, optima)

    f1_best = max(rows, key=lambda r: r["f1"])
    deployed = next(r for r in rows if abs(r["threshold"] - config.DECISION_THRESHOLD) < 1e-9)

    results = {
        "test_rows": int(len(context.y_test)),
        "deployed_threshold": config.DECISION_THRESHOLD,
        "deployed_operating_point": deployed,
        "f1_optimal": f1_best,
        "cost_ratio_optima": optima,
        "sweep": rows,
        "figure": figure,
    }

    with (config.TABLES_DIR / "threshold_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    _write_markdown(results)
    return results


def _write_markdown(results: dict[str, Any]) -> None:
    deployed = results["deployed_operating_point"]
    f1_best = results["f1_optimal"]
    optima = results["cost_ratio_optima"]
    total = results["test_rows"]

    def find(ratio: float) -> dict[str, Any]:
        return next(o for o in optima if o["cost_ratio"] == ratio)

    lines = [
        "# Decision Threshold Analysis",
        "",
        "Closes gap **G3**. Computed by `src/threshold.py` on the held-out test set",
        f"({total:,} customers).",
        "",
        "## The question this answers",
        "",
        "Version 1.0.0 used a decision threshold of 0.50 and documented it honestly as \"a",
        "documented default, not an optimised operating point\". *Why 0.50?* had no better answer",
        "than *because it is the default*.",
        "",
        "## Why no single number is given",
        "",
        "The optimal threshold depends entirely on one business quantity: **how many unnecessary",
        "reviews is one missed churner worth?** That is a commercial judgement about the cost of",
        "a lost customer against the cost of a specialist's time. It was never supplied, and",
        "inventing a currency figure would be fabricating a business claim.",
        "",
        "So instead of guessing, this analysis publishes the whole sensitivity curve. Only the",
        "*ratio* matters, not the absolute costs, which is precisely why no currency needs to be",
        "invented:",
        "",
        "```",
        "Cost(t) = FN(t) × C_miss + FP(t) × C_review        [set C_review = 1]",
        "        = FN(t) × ratio  + FP(t)",
        "```",
        "",
        "**The business supplies the ratio. The model supplies the curve.**",
        "",
        "## Optimal threshold by cost ratio",
        "",
        "| Cost ratio | Optimal threshold | Recall | Precision | Flagged | Missed churners |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for opt in optima:
        lines.append(
            f"| {opt['cost_ratio']:.0f}:1 | **{opt['optimal_threshold']:.2f}** | "
            f"{opt['recall']:.4f} | {opt['precision']:.4f} | "
            f"{opt['flagged']:,} ({opt['flagged_share'] * 100:.1f}%) | {opt['false_negative']:,} |"
        )

    low, mid, high = find(1.0), find(5.0), find(20.0)
    lines += [
        "",
        f"![Threshold analysis](figures/{results['figure']})",
        "",
        "## Reading the table",
        "",
        f"- At **1:1** — a missed churner costs the same as one unnecessary review — the optimum",
        f"  is {low['optimal_threshold']:.2f}. The model flags {low['flagged']:,} customers",
        f"  ({low['flagged_share'] * 100:.1f}% of the book) and misses {low['false_negative']:,}.",
        f"- At **5:1** the optimum moves to {mid['optimal_threshold']:.2f}: recall rises to",
        f"  {mid['recall']:.4f} and the review workload grows to {mid['flagged_share'] * 100:.1f}%.",
        f"- At **20:1** — a lost customer is very expensive relative to a review — the optimum is",
        f"  {high['optimal_threshold']:.2f}, catching {high['recall'] * 100:.1f}% of churners while",
        f"  flagging {high['flagged_share'] * 100:.1f}% of the book.",
        "",
        "The direction is intuitive and worth stating plainly: **the more a lost customer costs",
        "relative to a review, the lower the threshold should go** — you accept more false alarms",
        "to avoid missing anyone.",
        "",
        "## Where the deployed default sits",
        "",
        "| Operating point | Threshold | Recall | Precision | F1 | Flagged | Missed |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **Deployed default** | {deployed['threshold']:.2f} | {deployed['recall']:.4f} | "
        f"{deployed['precision']:.4f} | {deployed['f1']:.4f} | {deployed['flagged']:,} | "
        f"{deployed['false_negative']:,} |",
        f"| F1-maximising | {f1_best['threshold']:.2f} | {f1_best['recall']:.4f} | "
        f"{f1_best['precision']:.4f} | {f1_best['f1']:.4f} | {f1_best['flagged']:,} | "
        f"{f1_best['false_negative']:,} |",
        "",
        f"The deployed 0.50 threshold is cost-optimal at a ratio of approximately "
        f"**{_implied_ratio(optima, deployed['threshold'])}**. In other words, using 0.50 is an",
        "implicit assertion that one missed churner is worth about that many unnecessary reviews.",
        "Stating the assumption is the point: the number was always there, it was simply never",
        "made explicit.",
        "",
        "## Recommendation",
        "",
        "1. **Ask the business for the ratio.** One question — *how many unnecessary retention",
        "   reviews would you trade for catching one more churner?* — resolves the threshold",
        "   entirely. Most retention functions land between 3:1 and 10:1.",
        "2. **Until that answer exists, keep 0.50** and label it as an assumption rather than an",
        "   optimum. That is what the application now does.",
        "3. **Expose the threshold in the interface** so the sensitivity is visible rather than",
        "   buried in a report. The application now provides an adjustable threshold control,",
        "   defaulting to 0.50, with the risk bands and outputs recomputing live.",
        "",
        "## Limitations",
        "",
        "- The curve is computed on 1,409 held-out customers from a **fictional** sample. The",
        "  shape is informative; the exact optima are not precise to two decimal places.",
        "- Costs are assumed constant per customer. In reality the value of a retained customer",
        "  varies with tenure, contract and monthly charges — a per-customer cost model would be",
        "  a genuine improvement and is out of scope here.",
        "- The analysis assumes the probabilities rank customers correctly. It does **not** assume",
        "  they are calibrated; see `reports/calibration_report.md`, which shows they are not.",
        "- No account is taken of intervention effectiveness. A flagged customer who is contacted",
        "  may still churn; nothing here measures whether retention actions work.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make threshold",
        "```",
        "",
    ]

    (config.REPORTS_DIR / "threshold_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _implied_ratio(optima: list[dict[str, Any]], deployed_threshold: float) -> str:
    """Cost ratio whose optimum sits closest to the deployed threshold."""
    closest = min(optima, key=lambda o: abs(o["optimal_threshold"] - deployed_threshold))
    return f"{closest['cost_ratio']:.0f}:1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost-sensitive threshold analysis.")
    parser.parse_args(argv)

    results = run_threshold_analysis()
    print("=" * 72)
    print("Threshold analysis — held-out test set")
    print("=" * 72)
    print(f"{'ratio':>8}  {'threshold':>10}  {'recall':>8}  {'precision':>10}  {'flagged':>8}")
    for opt in results["cost_ratio_optima"]:
        print(
            f"{opt['cost_ratio']:>7.0f}:1  {opt['optimal_threshold']:>10.2f}  "
            f"{opt['recall']:>8.4f}  {opt['precision']:>10.4f}  {opt['flagged']:>8,}"
        )
    deployed = results["deployed_operating_point"]
    print(
        f"\nDeployed 0.50 -> recall {deployed['recall']:.4f}, "
        f"precision {deployed['precision']:.4f}, flagged {deployed['flagged']:,}"
    )
    print(f"Report -> {config.REPORTS_DIR / 'threshold_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
