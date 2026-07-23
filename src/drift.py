"""Data-drift detection apparatus (H2-2).

Closes gap G6 — with an important honesty constraint stated up front.

**No real drift can be observed in this project.** The dataset is a single
cross-section with no time dimension, so there is no "later" distribution to
compare against. Any claim to be *monitoring* drift would be false.

What can be built, and is built here, is the **apparatus**: a persisted baseline
profile, two detectors, and a demonstration that they fire on a deliberately
shifted sample and stay quiet on an unshifted one. A detector that has never been
shown to fire is not evidence of anything — the same standard applied to the
secret scanner in version 1.0.0.

Detectors
---------
- **Numeric:** Population Stability Index against baseline deciles.
- **Categorical:** PSI over category frequencies, plus a chi-squared test.

PSI bands (< 0.10 stable, 0.10–0.25 warning, > 0.25 alert) are a long-standing
credit-risk convention. They are **not** derived from this data and are labelled
as convention wherever they are reported.

Dependency note: `evidently` was evaluated and rejected. It brings a large
transitive dependency tree for two detectors implemented here in a few dozen
lines, and the team could not defend its internals under questioning.

CLI::

    python -m src.drift --build-baseline
    python -m src.drift --demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src import config  # noqa: E402
from src.analysis_base import load_evaluation_context  # noqa: E402

FIGURE_DPI = 200

BASELINE_PATH = config.ARTIFACTS_DIR / "drift_baseline.json"

#: Industry-convention PSI bands. Stated as convention, not derived here.
PSI_WARNING = 0.10
PSI_ALERT = 0.25

#: Significance level for the categorical chi-squared test.
CHI2_ALPHA = 0.05

#: Laplace smoothing keeps PSI finite when a bin empties in one distribution.
EPSILON = 1e-6

N_QUANTILE_BINS = 10


@dataclass
class FeatureDrift:
    """Drift measurement for a single feature."""

    feature: str
    kind: str
    psi: float
    status: str
    chi2_statistic: float | None = None
    chi2_p_value: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return self.status != "stable"


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _status(psi: float) -> str:
    if psi > PSI_ALERT:
        return "alert"
    if psi > PSI_WARNING:
        return "warning"
    return "stable"


def build_baseline(save: bool = True) -> dict[str, Any]:
    """Profile the training split and persist it as the reference distribution.

    The baseline comes from the **training** split because that is the population
    the model learned. Comparing live data to the test split would conflate drift
    with sampling noise in the holdout.
    """
    config.ensure_output_dirs()
    context = load_evaluation_context()
    frame = context.X_train

    numeric: dict[str, Any] = {}
    for column in config.NUMERIC_FEATURES:
        series = frame[column].dropna().to_numpy(dtype=float)
        # Quantile edges adapt to the actual distribution; equal-width bins would
        # leave most mass in one bin for skewed features such as TotalCharges.
        edges = np.unique(np.quantile(series, np.linspace(0, 1, N_QUANTILE_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(series, bins=edges)
        numeric[column] = {
            "bin_edges": [float(e) for e in edges],
            "proportions": (counts / counts.sum()).tolist(),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(np.median(series)),
        }

    categorical: dict[str, Any] = {}
    for column in config.CATEGORICAL_FEATURES:
        counts = frame[column].astype(str).value_counts()
        categorical[column] = {
            "categories": counts.index.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
            "counts": counts.astype(int).tolist(),
        }

    scores = context.train_probabilities
    score_counts, score_edges = np.histogram(scores, bins=20, range=(0.0, 1.0))

    baseline = {
        "note": (
            "Reference distribution profiled from the training split. No live data has been "
            "observed; this project cannot detect real drift because the dataset is a single "
            "cross-section with no time dimension."
        ),
        "source": "training split",
        "rows": int(len(frame)),
        "psi_bands": {"warning": PSI_WARNING, "alert": PSI_ALERT,
                      "note": "Credit-risk convention, not derived from this data."},
        "numeric": numeric,
        "categorical": categorical,
        "prediction_scores": {
            "bin_edges": [float(e) for e in score_edges],
            "proportions": (score_counts / score_counts.sum()).tolist(),
            "mean": float(scores.mean()),
        },
    }

    if save:
        with BASELINE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(baseline, handle, indent=2)
            handle.write("\n")
    return baseline


def load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(
            f"Drift baseline not found at {BASELINE_PATH}. "
            "Run `python -m src.drift --build-baseline` first."
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def detect_numeric_drift(column: str, values: pd.Series, baseline: dict[str, Any]) -> FeatureDrift:
    """PSI of *values* against the baseline bins for *column*."""
    reference = baseline["numeric"][column]
    edges = np.array(reference["bin_edges"], dtype=float)
    observed = values.dropna().to_numpy(dtype=float)

    counts, _ = np.histogram(observed, bins=edges)
    proportions = counts / max(counts.sum(), 1)
    psi = _psi(np.array(reference["proportions"]), proportions)

    return FeatureDrift(
        feature=column,
        kind="numeric",
        psi=round(psi, 6),
        status=_status(psi),
        detail={
            "baseline_mean": round(reference["mean"], 4),
            "observed_mean": round(float(observed.mean()), 4) if observed.size else None,
            "baseline_median": round(reference["median"], 4),
            "observed_median": round(float(np.median(observed)), 4) if observed.size else None,
        },
    )


def detect_categorical_drift(
    column: str, values: pd.Series, baseline: dict[str, Any]
) -> FeatureDrift:
    """PSI plus a chi-squared goodness-of-fit test against baseline frequencies."""
    reference = baseline["categorical"][column]
    categories = reference["categories"]
    expected_proportions = np.array(reference["proportions"], dtype=float)

    observed_counts = values.astype(str).value_counts()
    counts = np.array([float(observed_counts.get(c, 0)) for c in categories])
    total = counts.sum()
    proportions = counts / max(total, 1)

    psi = _psi(expected_proportions, proportions)

    chi2_stat: float | None = None
    p_value: float | None = None
    if total > 0:
        expected_counts = expected_proportions * total
        # Chi-squared is unreliable when expected cell counts are very small.
        if float(expected_counts.min()) >= 5.0:
            chi2_stat, p_value = stats.chisquare(f_obs=counts, f_exp=expected_counts)
            chi2_stat, p_value = float(chi2_stat), float(p_value)

    # PSI alone governs the status. The chi-squared p-value is reported for
    # information but deliberately does NOT escalate: at n≈1,400 it flags
    # differences far too small to matter. This was not a theoretical concern —
    # an earlier version let the p-value escalate and produced a false positive
    # on the unshifted control set (OnlineBackup, PSI 0.0044, p = 0.047). A
    # detector that cries wolf on its own holdout would not be trusted.
    status = _status(psi)

    unseen = sorted(set(observed_counts.index) - set(categories))
    return FeatureDrift(
        feature=column,
        kind="categorical",
        psi=round(psi, 6),
        status=status,
        chi2_statistic=round(chi2_stat, 6) if chi2_stat is not None else None,
        chi2_p_value=round(p_value, 6) if p_value is not None else None,
        detail={
            "unseen_categories": unseen,
            "baseline_top": categories[0] if categories else None,
            "observed_top": str(observed_counts.index[0]) if len(observed_counts) else None,
        },
    )


def detect_drift(frame: pd.DataFrame, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run every detector over *frame* and summarise."""
    reference = baseline or load_baseline()
    results: list[FeatureDrift] = []

    for column in config.NUMERIC_FEATURES:
        if column in frame.columns:
            results.append(detect_numeric_drift(column, frame[column], reference))
    for column in config.CATEGORICAL_FEATURES:
        if column in frame.columns:
            results.append(detect_categorical_drift(column, frame[column], reference))

    drifted = [r for r in results if r.drifted]
    alerts = [r for r in results if r.status == "alert"]

    return {
        "rows_checked": int(len(frame)),
        "features_checked": len(results),
        "drifted_count": len(drifted),
        "alert_count": len(alerts),
        "overall_status": "alert" if alerts else ("warning" if drifted else "stable"),
        "features": [
            {
                "feature": r.feature,
                "kind": r.kind,
                "psi": r.psi,
                "status": r.status,
                "chi2_p_value": r.chi2_p_value,
                "detail": r.detail,
            }
            for r in sorted(results, key=lambda r: r.psi, reverse=True)
        ],
    }


def make_shifted_sample(frame: pd.DataFrame, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Construct a deliberately shifted population to prove the detectors fire.

    The shift is a plausible business scenario rather than random noise: an
    acquisition campaign that brings in mostly new month-to-month fibre
    customers. That moves `Contract`, `InternetService` and `tenure` together,
    which is what real drift looks like.
    """
    rng = np.random.default_rng(random_state)
    month_to_month = frame[frame["Contract"] == "Month-to-month"]
    short_tenure = month_to_month[month_to_month["tenure"] <= 12]

    if len(short_tenure) < 50:  # pragma: no cover - guard for unexpected data
        return frame.sample(frac=0.5, random_state=random_state)

    take = min(len(short_tenure), 600)
    indices = rng.choice(short_tenure.index, size=take, replace=True)
    shifted = frame.loc[indices].copy()
    shifted["InternetService"] = "Fiber optic"
    return shifted


def plot_drift(report: dict[str, Any], filename: str = "18_drift_detection.png") -> str:
    """Horizontal PSI chart with the convention bands marked."""
    config.ensure_output_dirs()
    features = report["features"][:12]
    names = [f["feature"] for f in features]
    values = [f["psi"] for f in features]
    palette = {"stable": "#55A868", "warning": "#DD8452", "alert": "#C44E52"}
    colours = [palette[f["status"]] for f in features]

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(features) + 1.9))
    ax.set_axisbelow(True)
    ax.barh(range(len(features)), values, color=colours, height=0.62)
    ax.axvline(PSI_WARNING, color="#DD8452", linestyle="--", linewidth=1.2,
               label=f"Warning ({PSI_WARNING})")
    ax.axvline(PSI_ALERT, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Alert ({PSI_ALERT})")
    for y, value in enumerate(values):
        ax.text(value, y, f"  {value:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(features)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Population Stability Index")
    ax.set_title(
        f"Drift detection — overall status: {report['overall_status'].upper()}\n"
        "Bands are a credit-risk convention, not derived from this data",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filename


def run_demonstration() -> dict[str, Any]:
    """Prove the detectors fire on a shift and stay quiet without one."""
    config.ensure_output_dirs()
    baseline = build_baseline()
    context = load_evaluation_context()

    control = detect_drift(context.X_test, baseline)
    shifted_frame = make_shifted_sample(context.X_test)
    shifted = detect_drift(shifted_frame, baseline)
    figure = plot_drift(shifted)

    results = {
        "baseline_rows": baseline["rows"],
        "control": control,
        "shifted": shifted,
        "figure": figure,
        "detector_validated": bool(
            control["overall_status"] == "stable" and shifted["overall_status"] == "alert"
        ),
    }

    with (config.TABLES_DIR / "drift_report.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    _write_markdown(results)
    return results


def _write_markdown(results: dict[str, Any]) -> None:
    control, shifted = results["control"], results["shifted"]
    lines = [
        "# Drift Detection",
        "",
        "Closes gap **G6**. Produced by `src/drift.py`.",
        "",
        "## What this is, and what it is not",
        "",
        "**This project cannot detect real drift.** The IBM sample is a single cross-section",
        "with no time dimension, so there is no later distribution to compare against. Claiming",
        "to *monitor* drift here would be false.",
        "",
        "What has been built is the **apparatus**: a persisted baseline profile of the training",
        "population, two detectors, and a demonstration that they fire on a deliberately shifted",
        "sample and stay quiet on an unshifted one. A detector never shown to fire is not",
        "evidence of anything — the same standard this project applied to its secret scanner.",
        "",
        "## Method",
        "",
        "| Component | Approach |",
        "|---|---|",
        "| Baseline | Training split only, profiled into quantile bins and category frequencies |",
        "| Numeric detector | Population Stability Index over baseline deciles |",
        "| Categorical detector | PSI over category frequencies, plus a chi-squared goodness-of-fit test |",
        f"| Bands | PSI < {PSI_WARNING} stable · {PSI_WARNING}–{PSI_ALERT} warning · > {PSI_ALERT} alert |",
        "",
        f"The PSI bands are a **credit-risk industry convention**, not a threshold derived from",
        "this data. They are reported as such wherever they appear.",
        "",
        "Quantile bins are used rather than equal-width bins because `TotalCharges` and `tenure`",
        "are strongly skewed; equal-width binning would place most of the mass in one bin and",
        "make PSI insensitive to exactly the shifts worth catching.",
        "",
        "### Why not Evidently",
        "",
        "`evidently` was evaluated and rejected. It brings a substantial transitive dependency",
        "tree for two detectors implemented here in a few dozen lines, and the team could not",
        "defend its internals under questioning. The dependency cost exceeded the benefit.",
        "",
        "## Demonstration",
        "",
        "### Control — held-out test set against the training baseline",
        "",
        f"- Features checked: **{control['features_checked']}**",
        f"- Flagged: **{control['drifted_count']}**",
        f"- Overall status: **{control['overall_status'].upper()}**",
        "",
        "The held-out set is drawn from the same population as the training split, so a correct",
        "detector should report stability here. It does. This is the false-positive check.",
        "",
        "### Shifted — simulated acquisition campaign",
        "",
        "A deliberately shifted population was constructed: mostly new month-to-month fibre",
        "customers, as an acquisition campaign would produce. The shift moves `Contract`,",
        "`InternetService` and `tenure` together, which is what real drift looks like — unlike",
        "random noise, which moves nothing systematically.",
        "",
        f"- Features checked: **{shifted['features_checked']}**",
        f"- Flagged: **{shifted['drifted_count']}**",
        f"- At alert level: **{shifted['alert_count']}**",
        f"- Overall status: **{shifted['overall_status'].upper()}**",
        "",
        "| Feature | Kind | PSI | Status |",
        "|---|---|---:|---|",
    ]
    for feature in shifted["features"][:10]:
        lines.append(
            f"| `{feature['feature']}` | {feature['kind']} | {feature['psi']:.4f} | "
            f"**{feature['status']}** |"
        )

    lines += [
        "",
        f"![Drift detection](figures/{results['figure']})",
        "",
        "## Verdict",
        "",
    ]
    if results["detector_validated"]:
        lines += [
            "**The detector is validated.** It reports `stable` on an unshifted sample and",
            "`alert` on a shifted one. Both halves matter: a detector that always fires is as",
            "useless as one that never does.",
            "",
        ]
    else:
        lines += [
            f"**The detector did not behave as required.** Control status "
            f"`{control['overall_status']}`, shifted status `{shifted['overall_status']}`. Both",
            "conditions must hold — stable on control, alert on shifted — before this apparatus",
            "can be relied on. Recorded as failing rather than presented as working.",
            "",
        ]

    lines += [
        "## How this would be used in production",
        "",
        "1. Score a batch of live customers.",
        "2. Run `detect_drift()` against the persisted baseline.",
        "3. On `warning`, log and review at the next model-risk meeting.",
        "4. On `alert`, block automated use and require human revalidation before the model",
        "   continues to inform prioritisation.",
        "5. Retrain only when drift is confirmed *and* labelled outcomes are available —",
        "   retraining on drifted but unlabelled data would encode the drift rather than correct",
        "   it.",
        "",
        "## Limitations",
        "",
        "- **No real drift has been observed.** Everything here is apparatus and simulation.",
        "- PSI detects distribution change, not performance decay. A feature can shift without",
        "  harming accuracy, and accuracy can decay with no feature shift at all (concept drift).",
        "- Detecting **concept** drift needs outcome labels, which arrive only after customers",
        "  have actually churned. That lag is a property of the problem, not of this design.",
        "- The chi-squared test is sensitive at large sample sizes: with enough rows, trivial",
        "  differences become significant. That is why PSI, not the p-value, governs the status.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make drift",
        "```",
        "",
    ]

    (config.REPORTS_DIR / "drift_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data-drift detection apparatus.")
    parser.add_argument("--build-baseline", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Run the full demonstration.")
    args = parser.parse_args(argv)

    if args.build_baseline and not args.demo:
        baseline = build_baseline()
        print(f"Baseline built from {baseline['rows']:,} training rows -> {BASELINE_PATH}")
        return 0

    results = run_demonstration()
    print("=" * 72)
    print("Drift detection demonstration")
    print("=" * 72)
    print(f"Control  (unshifted holdout): {results['control']['overall_status'].upper()}  "
          f"({results['control']['drifted_count']} of "
          f"{results['control']['features_checked']} features flagged)")
    print(f"Shifted  (simulated campaign): {results['shifted']['overall_status'].upper()}  "
          f"({results['shifted']['drifted_count']} of "
          f"{results['shifted']['features_checked']} features flagged, "
          f"{results['shifted']['alert_count']} at alert)")
    print("\nTop shifted features:")
    for feature in results["shifted"]["features"][:5]:
        print(f"  {feature['feature']:<20} PSI {feature['psi']:.4f}  [{feature['status']}]")
    print(f"\nDetector validated: {results['detector_validated']}")
    print(f"Report -> {config.REPORTS_DIR / 'drift_report.md'}")
    return 0 if results["detector_validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
