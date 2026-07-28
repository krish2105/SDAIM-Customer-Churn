"""Revenue-weighted churn metrics: gross revenue churn, logo churn, and

expected revenue at risk.

A retention team does not experience every lost customer equally — a $95/month
fibre subscriber leaving costs more than a $20/month subscriber leaving, even
though both count as one churned logo. This module reports the standard
executive-level split between the two, computed on the held-out test set, and
states plainly which of the conventional revenue-churn metrics this dataset
can and cannot support.

Three numbers, three different epistemic status
-------------------------------------------------
- **Logo churn rate** and **Gross revenue churn rate** — **measured**,
  retrospective facts about the held-out test set. They use the actual
  ``Churn`` label, not a model output.
- **Net revenue churn rate** — **not computable**. It requires an
  expansion/contraction/reactivation revenue time series (upgrades,
  downgrades, win-backs across periods). This is a single fictional
  cross-section with no such series, so this project reports the gap rather
  than approximating it — the same honesty constraint already applied to
  drift detection in ``src/drift.py``.
- **Expected revenue at risk** — **prospective and model-based**. It combines
  the deployed model's probability with the survival-based expected remaining
  tenure from ``src/survival.py``. It is an exposure estimate for
  prioritisation, not a forecast of realised loss.

CLI::

    python -m src.revenue
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.analysis_base import EvaluationContext, load_evaluation_context  # noqa: E402
from src.survival import expected_remaining_tenure  # noqa: E402

FIGURE_DPI = 200


def _load_survival_reference() -> dict[str, Any]:
    if not config.SURVIVAL_REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            f"Survival reference not found at {config.SURVIVAL_REFERENCE_PATH}. "
            "Run `python -m src.survival` first."
        )
    return json.loads(config.SURVIVAL_REFERENCE_PATH.read_text(encoding="utf-8"))


def measured_revenue_churn(context: EvaluationContext) -> dict[str, float]:
    """Logo and gross revenue churn, computed from the actual test-set outcome."""
    frame = context.X_test
    actual = context.y_test.to_numpy()
    monthly = frame["MonthlyCharges"].to_numpy(dtype=float)

    total_customers = len(frame)
    total_revenue = float(monthly.sum())
    churned_customers = int(actual.sum())
    churned_revenue = float(monthly[actual == 1].sum())

    return {
        "test_rows": total_customers,
        "logo_churn_rate": float(churned_customers / total_customers),
        "total_monthly_revenue": round(total_revenue, 2),
        "churned_monthly_revenue": round(churned_revenue, 2),
        "gross_revenue_churn_rate": float(churned_revenue / total_revenue),
    }


def prospective_exposure(
    context: EvaluationContext, threshold: float, survival_reference: dict[str, Any]
) -> dict[str, Any]:
    """Model-based revenue exposure: who is flagged, and how much is at stake."""
    frame = context.X_test
    probabilities = context.test_probabilities
    monthly = frame["MonthlyCharges"].to_numpy(dtype=float)
    tenure = frame["tenure"].to_numpy(dtype=float)
    contract = frame["Contract"].astype(str).to_numpy()

    flagged = probabilities >= threshold
    total_revenue = float(monthly.sum())
    flagged_revenue = float(monthly[flagged].sum())
    flagged_customer_share = float(flagged.mean())
    flagged_revenue_share = flagged_revenue / total_revenue if total_revenue else 0.0

    remaining = np.array(
        [
            expected_remaining_tenure(t, survival_reference["segments"][c]["survival"])
            for t, c in zip(tenure, contract)
        ]
    )
    exposure = probabilities * monthly * remaining

    return {
        "threshold": threshold,
        "flagged_customers": int(flagged.sum()),
        "flagged_customer_share": flagged_customer_share,
        "flagged_monthly_revenue": round(flagged_revenue, 2),
        "flagged_revenue_share": flagged_revenue_share,
        "revenue_concentration_ratio": (
            round(flagged_revenue_share / flagged_customer_share, 4)
            if flagged_customer_share else None
        ),
        "total_expected_revenue_at_risk": round(float(exposure.sum()), 2),
        "flagged_expected_revenue_at_risk": round(float(exposure[flagged].sum()), 2),
    }


def plot_revenue_churn(
    measured: dict[str, float], filename: str = "21_revenue_at_risk.png"
) -> str:
    """Logo churn rate against gross revenue churn rate, side by side."""
    config.ensure_output_dirs()
    labels = ["Logo churn rate\n(share of customers)", "Gross revenue churn rate\n(share of monthly revenue)"]
    values = [measured["logo_churn_rate"] * 100, measured["gross_revenue_churn_rate"] * 100]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.set_axisbelow(True)
    bars = ax.bar(labels, values, color=["#4C72B0", "#C44E52"], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(
        "Logo churn vs. gross revenue churn — held-out test set\n"
        "Measured from the actual outcome, not a model output",
        fontsize=10, loc="left",
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filename


def run_revenue_analysis(threshold: float | None = None) -> dict[str, Any]:
    """Compute, persist and report every revenue-churn metric."""
    config.ensure_output_dirs()
    context = load_evaluation_context()
    decision_threshold = threshold if threshold is not None else float(
        context.metadata.get("decision_threshold", config.DECISION_THRESHOLD)
    )
    survival_reference = _load_survival_reference()

    measured = measured_revenue_churn(context)
    prospective = prospective_exposure(context, decision_threshold, survival_reference)
    figure = plot_revenue_churn(measured)

    results = {"measured": measured, "prospective": prospective, "figure": figure}
    with (config.TABLES_DIR / "revenue_churn_report.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    _write_markdown(results)
    return results


def _write_markdown(results: dict[str, Any]) -> None:
    measured = results["measured"]
    prospective = results["prospective"]

    concentration = prospective["revenue_concentration_ratio"]
    concentration_note = (
        f"The flagged group holds **{concentration:.2f}x** the revenue share its customer "
        "share would imply" if concentration is not None else "No customers were flagged."
    )

    lines = [
        "# Revenue-Weighted Churn Metrics",
        "",
        "Computed by `src/revenue.py` on the held-out test set. A retention team does not",
        "experience every departure equally — this splits the standard **logo churn** figure",
        "from its **revenue-weighted** counterpart and reports a **model-based revenue exposure**",
        "for the accounts the deployed threshold currently flags.",
        "",
        "## Measured (retrospective, from the actual test-set outcome)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test-set customers | {measured['test_rows']:,} |",
        f"| Logo churn rate | {measured['logo_churn_rate']:.4f} |",
        f"| Total monthly revenue in the test set | ${measured['total_monthly_revenue']:,.2f} |",
        f"| Monthly revenue attached to churned customers | ${measured['churned_monthly_revenue']:,.2f} |",
        f"| **Gross revenue churn rate** | **{measured['gross_revenue_churn_rate']:.4f}** |",
        "",
    ]

    gap = measured["gross_revenue_churn_rate"] - measured["logo_churn_rate"]
    if abs(gap) > 0.01:
        direction = "above" if gap > 0 else "below"
        lines += [
            f"Gross revenue churn is **{abs(gap):.4f} {direction}** logo churn — churned customers",
            "are not a representative cross-section of the account book by spend. See the segment",
            "reference rates for which contract types drive this.",
            "",
        ]
    else:
        lines += [
            "Gross revenue churn and logo churn are close in this sample — churned customers'",
            "average spend is not materially different from the book as a whole.",
            "",
        ]

    lines += [
        f"![Logo churn vs revenue churn](figures/{results['figure']})",
        "",
        "## Net revenue churn — not computable",
        "",
        "**Net revenue churn cannot be measured from this dataset**, and no substitute figure is",
        "reported in its place. It requires an expansion/contraction/reactivation revenue series —",
        "upgrades, downgrades and win-backs tracked across periods — and this project has exactly",
        "one fictional cross-section with no time dimension. This is the same constraint already",
        "documented for drift detection in `reports/drift_report.md`: the honest response to a",
        "metric the data cannot support is to say so, not to approximate it.",
        "",
        "## Prospective (model-based revenue exposure at the deployed threshold)",
        "",
        f"At the decision threshold of **{prospective['threshold']:.2f}**:",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Customers flagged | {prospective['flagged_customers']:,} "
        f"({prospective['flagged_customer_share']:.1%} of the test set) |",
        f"| Monthly revenue flagged | ${prospective['flagged_monthly_revenue']:,.2f} "
        f"({prospective['flagged_revenue_share']:.1%} of test-set revenue) |",
        f"| **Total expected revenue at risk** (whole test set) | "
        f"**${prospective['total_expected_revenue_at_risk']:,.2f}** |",
        f"| Expected revenue at risk (flagged accounts only) | "
        f"${prospective['flagged_expected_revenue_at_risk']:,.2f} |",
        "",
        concentration_note + ", meaning the model's review queue is "
        + (
            "weighted toward higher-spend accounts than a random sample of the same size would be."
            if concentration and concentration > 1.05
            else (
                "weighted toward lower-spend accounts than a random sample of the same size would be."
                if concentration and concentration < 0.95
                else "roughly proportionate to revenue — no material concentration either way."
            )
        ) if concentration is not None else "",
        "",
        "**Expected revenue at risk** multiplies, per customer: the model's churn probability x",
        "`MonthlyCharges` x an expected-remaining-tenure figure from the Kaplan-Meier survival",
        "curves in `reports/survival_report.md`. It is a transparent combination of three",
        "already-disclosed numbers, offered to help prioritise a review queue by commercial",
        "stake rather than probability alone. **It is not a forecast of realised loss and no",
        "return on any retention action has been measured.**",
        "",
        "## Limitations",
        "",
        "- Measured figures describe the **fictional** IBM sample's held-out test set, not a live",
        "  population.",
        "- Expected revenue at risk depends on the survival curves' own limitations (single",
        "  cross-section, `Contract`-only stratification) — see `reports/survival_report.md`.",
        "- No customer-level financial outcome has been validated against this exposure figure;",
        "  it is arithmetic on disclosed numbers, not a measured return.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make survival",
        "make revenue",
        "```",
        "",
    ]

    (config.REPORTS_DIR / "revenue_churn_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revenue-weighted churn metrics.")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)

    results = run_revenue_analysis(args.threshold)
    print("=" * 72)
    print("Revenue-weighted churn metrics — held-out test set")
    print("=" * 72)
    measured = results["measured"]
    prospective = results["prospective"]
    print(f"  Logo churn rate            {measured['logo_churn_rate']:.4f}")
    print(f"  Gross revenue churn rate   {measured['gross_revenue_churn_rate']:.4f}")
    print(f"  Net revenue churn          not computable (no expansion/contraction series)")
    print(f"\n  Flagged at threshold {prospective['threshold']:.2f}: "
          f"{prospective['flagged_customers']:,} customers, "
          f"${prospective['flagged_monthly_revenue']:,.2f}/mo")
    print(f"  Total expected revenue at risk: ${prospective['total_expected_revenue_at_risk']:,.2f}")
    print(f"\nReport -> {config.REPORTS_DIR / 'revenue_churn_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
