"""Survival analysis: observed time-to-churn by contract segment.

Extends the project beyond a point-in-time probability into *when* a churn
event tends to happen. The dataset already has exactly the two columns a
survival analysis needs, with no extra assumption required:

- **Duration** — ``tenure``, months with the company.
- **Event** — ``Churn``. ``Yes`` is an observed event at that tenure. ``No``
  is a **right-censored** observation: the customer was still active after at
  least that many months, and what happens next is unknown.

This is the standard construction for a churn survival analysis (Kaplan-Meier
on tenure, stratified by a commercial attribute) and requires nothing beyond
what the raw file already records.

Descriptive, not predictive
----------------------------
This module characterises **observed** churn timing by `Contract`. It plays
no part in the classifier's pipeline, so it carries no leakage risk from using
the full validated dataset rather than the train/test split reserved for
evaluating the classifier's *predictions* (that split matters for fairness,
calibration and drift because those measure the model; this measures the raw
data).

What this is used for
----------------------
The per-segment survival curve is condensed into a **discrete expected
remaining tenure** given a customer's current tenure, and persisted as
``deploy/artifacts/survival_reference.json`` — a lookup table, not a model.
It is consumed at inference time by ``deploy/valuation.py`` to convert a churn
probability into an estimated dollar exposure (probability x monthly charge x
expected remaining months), without adding a runtime dependency on
``lifelines``.

CLI::

    python -m src.survival
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402
from lifelines.utils import restricted_mean_survival_time  # noqa: E402

from src import config  # noqa: E402
from src.data_validation import validate_dataset  # noqa: E402
from src.train import load_model_frame  # noqa: E402

FIGURE_DPI = 200

#: A customer surviving to the horizon, or a segment whose curve barely moves,
#: is still assumed to have *some* remaining commercial life rather than none.
#: One month is a floor, not a measurement — stated wherever it is applied.
FLOOR_REMAINING_MONTHS = 1.0
EPSILON = 1e-6


def _event_indicator(frame: pd.DataFrame) -> pd.Series:
    """``1`` for an observed churn event, ``0`` for right-censored."""
    return (frame[config.TARGET_COLUMN] == "Yes").astype(int)


def fit_segment_curves(
    frame: pd.DataFrame, tau: int
) -> dict[str, KaplanMeierFitter]:
    """One Kaplan-Meier fit per `Contract` level."""
    event = _event_indicator(frame)
    fits: dict[str, KaplanMeierFitter] = {}
    for level in config.EXPECTED_CATEGORIES["Contract"]:
        mask = frame["Contract"] == level
        kmf = KaplanMeierFitter()
        kmf.fit(frame.loc[mask, "tenure"], event_observed=event.loc[mask], label=level)
        fits[level] = kmf
    return fits


def step_function(kmf: KaplanMeierFitter, tau: int) -> list[float]:
    """Survival probability at every integer month ``0..tau``, held forward.

    lifelines holds the last computed value forward past the segment's own
    last observed duration. That is an **extrapolation**, stated as a
    limitation in the report: no segment recorded churn beyond the horizon in
    this dataset, but that is a fact about the sample, not a proof of it.
    """
    times = np.arange(0, tau + 1)
    values = kmf.survival_function_at_times(times).to_numpy()
    return [float(v) for v in values]


def expected_remaining_tenure(
    current_tenure: float, survival: list[float], floor: float = FLOOR_REMAINING_MONTHS
) -> float:
    """Discrete restricted-mean residual life given survival to ``current_tenure``.

    ``E[T - t0 | T > t0] ~= sum_{u=t0+1}^{tau} S(u) / S(t0)`` — the standard
    discretisation of the continuous residual-life integral, using the
    persisted step function rather than the raw KM fit so this same arithmetic
    runs at inference time with no ``lifelines`` dependency.
    """
    tau = len(survival) - 1
    t0 = int(round(max(0.0, min(current_tenure, tau))))
    if t0 >= tau:
        return floor
    s_t0 = max(survival[t0], EPSILON)
    remaining = sum(survival[t0 + 1 : tau + 1]) / s_t0
    return max(remaining, floor)


def build_reference(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit every segment and condense it into the persisted lookup table."""
    tau = int(frame["tenure"].max())
    fits = fit_segment_curves(frame, tau)

    segments: dict[str, Any] = {}
    for level, kmf in fits.items():
        survival = step_function(kmf, tau)
        median = kmf.median_survival_time_
        segments[level] = {
            "n": int((frame["Contract"] == level).sum()),
            "events": int(_event_indicator(frame.loc[frame["Contract"] == level]).sum()),
            "survival": survival,
            "median_survival_months": None if np.isinf(median) else float(median),
            "restricted_mean_survival_months": float(
                restricted_mean_survival_time(kmf, t=tau)
            ),
        }

    return {
        "note": (
            "Descriptive Kaplan-Meier survival curves by Contract, fitted on the full "
            "validated dataset (duration=tenure, event=Churn=='Yes', 'No' right-censored "
            "at that tenure). Used only to estimate expected remaining tenure for revenue "
            "valuation; plays no part in the churn classifier."
        ),
        "tau_months": tau,
        "floor_remaining_months": FLOOR_REMAINING_MONTHS,
        "segments": segments,
    }


def logrank_test(frame: pd.DataFrame) -> dict[str, float]:
    """Whether the three contract curves differ by more than sampling noise."""
    result = multivariate_logrank_test(
        frame["tenure"], frame["Contract"], _event_indicator(frame)
    )
    return {"test_statistic": float(result.test_statistic), "p_value": float(result.p_value)}


def plot_survival_curves(
    fits: dict[str, KaplanMeierFitter], filename: str = "20_survival_curves_by_contract.png"
) -> str:
    """Kaplan-Meier curves for all three contract segments on one axis."""
    config.ensure_output_dirs()
    colours = {"Month-to-month": "#C44E52", "One year": "#DD8452", "Two year": "#55A868"}

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_axisbelow(True)
    for level, kmf in fits.items():
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colours.get(level, "#4C72B0"))
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival probability (still a customer)")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Observed retention curves by contract type\n"
        "Kaplan-Meier estimate; shaded band is the 95% confidence interval",
        fontsize=10, loc="left",
    )
    ax.legend(title="Contract", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filename


def run_survival_analysis() -> dict[str, Any]:
    """Fit, persist the runtime lookup, and write the report."""
    config.ensure_output_dirs()
    validation = validate_dataset(config.RAW_DATASET_PATH)
    if not validation.passed:
        raise RuntimeError("Refusing to run survival analysis: raw dataset validation failed.")

    frame = load_model_frame()
    tau = int(frame["tenure"].max())
    fits = fit_segment_curves(frame, tau)
    reference = build_reference(frame)
    rank_test = logrank_test(frame)
    figure = plot_survival_curves(fits)

    with config.SURVIVAL_REFERENCE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=2)
        handle.write("\n")

    summary_rows = [
        {
            "contract": level,
            "n": payload["n"],
            "events_observed": payload["events"],
            "median_survival_months": payload["median_survival_months"],
            "restricted_mean_survival_months": round(
                payload["restricted_mean_survival_months"], 2
            ),
        }
        for level, payload in reference["segments"].items()
    ]
    pd.DataFrame(summary_rows).to_csv(config.TABLES_DIR / "survival_summary.csv", index=False)

    results = {
        "tau_months": tau,
        "segments": summary_rows,
        "logrank_test": rank_test,
        "figure": figure,
    }
    with (config.TABLES_DIR / "survival_report.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    _write_markdown(results)
    return results


def _write_markdown(results: dict[str, Any]) -> None:
    rank_test = results["logrank_test"]
    lines = [
        "# Survival Analysis — Time to Churn by Contract",
        "",
        "Computed by `src/survival.py` on the full validated dataset. Extends the "
        "point-in-time churn probability into an estimate of **when** a customer of a given "
        "contract type tends to leave, and how much commercial life remains for one who has "
        "already reached a given tenure.",
        "",
        "## Method",
        "",
        "- **Duration** — `tenure` (months with the company).",
        "- **Event** — `Churn == 'Yes'` is an observed event at that tenure. `Churn == 'No'` is "
        "  **right-censored**: the customer was known to be active for at least that many "
        "  months, with no further information.",
        "- One Kaplan-Meier curve fitted per `Contract` level.",
        "- Fitted on the **full validated dataset**, not the train/test split used elsewhere. "
        "  This module describes observed data, not the classifier's predictions, so the split "
        "  that protects against leakage into a model has no equivalent requirement here.",
        "",
        "## Are the three curves actually different?",
        "",
        f"Multivariate log-rank test across the three `Contract` groups: statistic "
        f"**{rank_test['test_statistic']:.2f}**, p-value **{rank_test['p_value']:.2e}**. The "
        "curves differ far beyond what sampling noise would produce — consistent with the "
        "EDA finding that `Contract` is the strongest single correlate of churn.",
        "",
        f"![Survival curves by contract](figures/{results['figure']})",
        "",
        "## Segment summary",
        "",
        "| Contract | n | Events observed | Median survival (months) | "
        f"Restricted mean survival to {results['tau_months']} months |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["segments"]:
        median = "not reached" if row["median_survival_months"] is None else f"{row['median_survival_months']:.0f}"
        lines.append(
            f"| {row['contract']} | {row['n']:,} | {row['events_observed']:,} | {median} | "
            f"{row['restricted_mean_survival_months']:.1f} |"
        )

    lines += [
        "",
        "**Median survival** is the tenure at which half the segment has churned. For One year",
        "and Two year contracts the curve never falls below 50% inside the observed horizon, so",
        "the median is reported as \"not reached\" rather than a fabricated number — the",
        "**restricted mean survival time** (area under the curve up to the horizon) is used",
        "instead, which is well-defined for every segment.",
        "",
        "## From a curve to a number: expected remaining tenure",
        "",
        "The step function behind each curve is condensed into a discrete restricted-mean",
        "residual-life estimate:",
        "",
        "```",
        "E[remaining months | survived to t0] ~= sum_{u=t0+1..tau} S(u) / S(t0)",
        "```",
        "",
        f"floored at **{FLOOR_REMAINING_MONTHS:.0f} month** so a customer already at the horizon",
        "is not assigned zero remaining value. This is the figure `deploy/valuation.py` multiplies",
        "by the churn probability and `MonthlyCharges` to produce the revenue-at-risk estimate",
        "shown in the application — see `reports/revenue_churn_report.md`.",
        "",
        "## Limitations",
        "",
        "- This is a **single cross-section**: every customer was observed once, at their",
        "  current tenure. There is no way to distinguish a genuine plateau in the hazard from an",
        "  artefact of how long the company itself has existed.",
        "- Values beyond a segment's own last observed duration are **held forward** from the",
        "  last computed point (a lifelines convention for step-function survival curves), which",
        "  is an extrapolation, not an additional observation.",
        "- Contract type is the only stratification variable. A customer's true residual life",
        "  almost certainly also depends on tenure-independent attributes (`InternetService`,",
        "  `TechSupport`) not modelled here — this analysis trades that resolution for a stable,",
        "  legible lookup table with enough events per stratum to fit reliably.",
        "- This describes the fictional IBM sample. No claim is made about a live population.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make survival",
        "```",
        "",
    ]

    (config.REPORTS_DIR / "survival_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Survival analysis by contract segment.")
    parser.parse_args(argv)

    results = run_survival_analysis()
    print("=" * 72)
    print("Survival analysis — time to churn by contract")
    print("=" * 72)
    for row in results["segments"]:
        median = "not reached" if row["median_survival_months"] is None else f"{row['median_survival_months']:.0f}"
        print(
            f"  {row['contract']:<16} n={row['n']:<6} events={row['events_observed']:<5} "
            f"median={median:<12} RMST={row['restricted_mean_survival_months']:.1f}"
        )
    print(f"\nLog-rank test: statistic={results['logrank_test']['test_statistic']:.2f} "
          f"p={results['logrank_test']['p_value']:.2e}")
    print(f"Reference -> {config.SURVIVAL_REFERENCE_PATH}")
    print(f"Report -> {config.REPORTS_DIR / 'survival_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
