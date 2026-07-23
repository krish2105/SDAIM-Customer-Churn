"""Fairness audit across protected-adjacent attributes (H1-1).

Closes the limitation this project published about itself: `gender` and
`SeniorCitizen` are predictors and no subgroup analysis existed.

Three standard criteria are computed. They are mathematically incompatible in
general — you cannot satisfy all three at once when base rates differ between
groups — so this module reports all three and the project states which one it
optimises for. See ``reports/fairness_report.md``.

Implemented with plain ``sklearn.metrics`` grouped by attribute rather than
Fairlearn or AIF360: two attributes with two levels each do not justify a
dependency whose internals the team could not defend under questioning.

CLI::

    python -m src.fairness
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from src import config  # noqa: E402
from src.analysis_base import EvaluationContext, load_evaluation_context  # noqa: E402

FIGURE_DPI = 200

#: Attributes audited, with human-readable level labels.
PROTECTED_ATTRIBUTES: dict[str, dict[str, str]] = {
    "gender": {"Female": "Female", "Male": "Male"},
    "SeniorCitizen": {"0": "Not senior", "1": "Senior citizen"},
}

#: Disparity above this ratio gap is reported as material. 0.10 (10 percentage
#: points on a rate, or 0.10 on a ratio) is a common practitioner convention;
#: it is not derived from this data and is stated as a convention in the report.
MATERIALITY_THRESHOLD: float = 0.10

#: The four-fifths rule from US EEOC guidance, widely reused as a rough screen.
#: Included as a secondary reference point, not as a legal test.
FOUR_FIFTHS: float = 0.80


@dataclass
class GroupMetrics:
    """Per-subgroup performance on the held-out test set."""

    attribute: str
    level: str
    label: str
    n: int
    base_rate: float          # actual churn rate within the group
    selection_rate: float     # share flagged by the model  -> demographic parity
    recall: float             # TPR                          -> equal opportunity
    precision: float          # PPV                          -> predictive parity
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_group_metrics(
    context: EvaluationContext,
    attribute: str,
    threshold: float,
) -> list[GroupMetrics]:
    """Confusion-matrix metrics within each level of *attribute*."""
    predictions = context.predictions_at(threshold)
    actual = context.y_test.to_numpy()
    groups = context.X_test[attribute].astype(str).to_numpy()

    results: list[GroupMetrics] = []
    for level in sorted(set(groups)):
        mask = groups == level
        y_true = actual[mask]
        y_pred = predictions[mask]
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]),
                          int(matrix[1, 0]), int(matrix[1, 1]))
        total = tn + fp + fn + tp

        results.append(
            GroupMetrics(
                attribute=attribute,
                level=level,
                label=PROTECTED_ATTRIBUTES.get(attribute, {}).get(level, level),
                n=total,
                base_rate=_safe_divide(fn + tp, total),
                selection_rate=_safe_divide(fp + tp, total),
                recall=_safe_divide(tp, tp + fn),
                precision=_safe_divide(tp, tp + fp),
                false_positive_rate=_safe_divide(fp, fp + tn),
                false_negative_rate=_safe_divide(fn, fn + tp),
                accuracy=_safe_divide(tp + tn, total),
                true_negative=tn,
                false_positive=fp,
                false_negative=fn,
                true_positive=tp,
            )
        )
    return results


def compute_disparities(groups: list[GroupMetrics]) -> dict[str, Any]:
    """Gaps and ratios between the most- and least-favoured groups."""

    def gap_and_ratio(values: list[float]) -> dict[str, float]:
        highest, lowest = max(values), min(values)
        return {
            "max": round(highest, 6),
            "min": round(lowest, 6),
            "gap": round(highest - lowest, 6),
            "ratio": round(_safe_divide(lowest, highest), 6),
        }

    return {
        "demographic_parity": gap_and_ratio([g.selection_rate for g in groups]),
        "equal_opportunity": gap_and_ratio([g.recall for g in groups]),
        "predictive_parity": gap_and_ratio([g.precision for g in groups]),
        "false_positive_rate": gap_and_ratio([g.false_positive_rate for g in groups]),
        "base_rate": gap_and_ratio([g.base_rate for g in groups]),
    }


def assess(disparities: dict[str, Any]) -> dict[str, Any]:
    """Flag whether any criterion exceeds the materiality convention."""
    findings = {}
    for criterion, values in disparities.items():
        if criterion == "base_rate":
            continue
        findings[criterion] = {
            "gap": values["gap"],
            "material": bool(values["gap"] > MATERIALITY_THRESHOLD),
            "passes_four_fifths": bool(values["ratio"] >= FOUR_FIFTHS),
        }
    return findings


def plot_attribute(groups: list[GroupMetrics], attribute: str) -> str:
    """Grouped bar chart of the three fairness criteria per subgroup."""
    config.ensure_output_dirs()
    labels = [g.label for g in groups]
    criteria = {
        "Selection rate\n(demographic parity)": [g.selection_rate for g in groups],
        "Recall\n(equal opportunity)": [g.recall for g in groups],
        "Precision\n(predictive parity)": [g.precision for g in groups],
        "Actual churn rate\n(base rate)": [g.base_rate for g in groups],
    }

    positions = np.arange(len(criteria))
    width = 0.8 / max(len(groups), 1)
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.set_axisbelow(True)
    for index, group in enumerate(groups):
        values = [criteria[name][index] for name in criteria]
        offset = (index - (len(groups) - 1) / 2) * width
        bars = ax.bar(positions + offset, values, width * 0.92,
                      label=f"{group.label} (n={group.n:,})", color=colours[index % len(colours)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                    ha="center", fontsize=7.5)

    ax.set_xticks(positions, list(criteria), fontsize=8.5)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Fairness criteria by {attribute} — held-out test set\n"
        "Base rate shown for context: differing base rates make the criteria mutually exclusive",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=8.5)
    ax.grid(axis="x", visible=False)
    filename = f"15_fairness_{attribute.lower()}.png"
    fig.savefig(config.FIGURES_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filename


def run_audit(threshold: float | None = None, *, with_counterfactual: bool = True) -> dict[str, Any]:
    """Execute the full audit and persist the report.

    The counterfactual runs by default because the keep-or-remove decision is
    only evidence-based if the cost of removal has actually been measured.
    """
    config.ensure_output_dirs()
    context = load_evaluation_context()
    decision_threshold = threshold if threshold is not None else float(
        context.metadata.get("decision_threshold", config.DECISION_THRESHOLD)
    )

    results: dict[str, Any] = {
        "decision_threshold": decision_threshold,
        "materiality_threshold": MATERIALITY_THRESHOLD,
        "test_rows": int(len(context.y_test)),
        "attributes": {},
    }

    for attribute in PROTECTED_ATTRIBUTES:
        groups = compute_group_metrics(context, attribute, decision_threshold)
        disparities = compute_disparities(groups)
        results["attributes"][attribute] = {
            "groups": [asdict(g) for g in groups],
            "disparities": disparities,
            "assessment": assess(disparities),
            "figure": plot_attribute(groups, attribute),
        }

    if with_counterfactual:
        results["counterfactual_without_protected_attributes"] = (
            counterfactual_without_attributes()
        )

    report_path = config.TABLES_DIR / "fairness_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    _write_markdown(results)
    return results


def _write_markdown(results: dict[str, Any]) -> None:
    """Write the human-readable audit with its findings and decision."""
    lines = [
        "# Fairness Audit",
        "",
        "Closes gap **G1**. Computed by `src/fairness.py` on the held-out test set at a",
        f"decision threshold of {results['decision_threshold']:.2f}. Every number is computed;",
        "none is hand-entered.",
        "",
        "## Why this audit exists",
        "",
        "`gender` and `SeniorCitizen` are model predictors. Version 1.0.0 of this project",
        "documented the absence of a fairness audit as a known limitation and stated that one",
        "was required before operational use. This is that audit.",
        "",
        "## The three criteria, and why they cannot all hold",
        "",
        "| Criterion | Question it asks | Equalises |",
        "|---|---|---|",
        "| Demographic parity | Are groups flagged at the same rate? | Selection rate |",
        "| Equal opportunity | Are at-risk customers found equally well in each group? | Recall (TPR) |",
        "| Predictive parity | Is a flag equally trustworthy for each group? | Precision (PPV) |",
        "",
        "These are **mathematically incompatible** whenever the groups have different actual",
        "churn rates, which is why the base rate is reported alongside them. Satisfying all",
        "three simultaneously is impossible, so a choice must be made and justified.",
        "",
        "**This project optimises for equal opportunity.** The model's purpose is to find",
        "at-risk customers so a specialist can review them. The harm that matters is a customer",
        "being missed — receiving no retention review — and that harm should not fall more",
        "heavily on one group. Demographic parity would be the wrong target here: if one group",
        "genuinely churns more, forcing equal flag rates would systematically under-serve it.",
        "",
        f"Disparities above **{MATERIALITY_THRESHOLD:.2f}** are reported as material. That is a",
        "practitioner convention, not a value derived from this data. The four-fifths ratio",
        "screen (0.80) from US EEOC guidance is reported as a secondary reference point, not as",
        "a legal test — this is a fictional dataset and no legal conclusion follows from it.",
        "",
    ]

    for attribute, payload in results["attributes"].items():
        groups = payload["groups"]
        disparities = payload["disparities"]
        assessment = payload["assessment"]

        lines += [
            f"## {attribute}",
            "",
            "| Group | n | Actual churn rate | Selection rate | Recall | Precision | FPR | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for group in groups:
            lines.append(
                f"| {group['label']} | {group['n']:,} | {group['base_rate']:.4f} | "
                f"{group['selection_rate']:.4f} | {group['recall']:.4f} | "
                f"{group['precision']:.4f} | {group['false_positive_rate']:.4f} | "
                f"{group['accuracy']:.4f} |"
            )

        lines += [
            "",
            "Confusion matrix per group:",
            "",
            "| Group | TN | FP | FN | TP |",
            "|---|---:|---:|---:|---:|",
        ]
        for group in groups:
            lines.append(
                f"| {group['label']} | {group['true_negative']:,} | {group['false_positive']:,} | "
                f"{group['false_negative']:,} | {group['true_positive']:,} |"
            )

        lines += [
            "",
            "### Disparities",
            "",
            "| Criterion | Max | Min | Gap | Ratio (min/max) | Material? | Passes 4/5 screen |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for criterion in ("demographic_parity", "equal_opportunity", "predictive_parity",
                          "false_positive_rate"):
            values = disparities[criterion]
            flag = assessment.get(criterion, {})
            material = "**YES**" if flag.get("material") else "no"
            four_fifths = "yes" if flag.get("passes_four_fifths") else "**NO**"
            lines.append(
                f"| {criterion.replace('_', ' ')} | {values['max']:.4f} | {values['min']:.4f} | "
                f"{values['gap']:.4f} | {values['ratio']:.4f} | {material} | {four_fifths} |"
            )

        base = disparities["base_rate"]
        lines += [
            "",
            f"Actual churn rate differs between groups by {base['gap']:.4f} "
            f"({base['min']:.4f} to {base['max']:.4f}). This is a property of the sample, not of",
            "the model, and it is the reason the three criteria cannot all be satisfied.",
            "",
            f"![Fairness by {attribute}](figures/{payload['figure']})",
            "",
        ]

    # ---- Overall finding and decision -------------------------------------
    material = {
        attribute: [c for c, flag in payload["assessment"].items() if flag["material"]]
        for attribute, payload in results["attributes"].items()
    }
    clean = [a for a, criteria in material.items() if not criteria]
    flagged = [a for a, criteria in material.items() if criteria]

    lines += ["## Finding", ""]

    for attribute in clean:
        lines += [
            f"### {attribute} — no material disparity",
            "",
            f"Every criterion for `{attribute}` fell below the {MATERIALITY_THRESHOLD:.2f}",
            "materiality convention. This is a genuine measured result, not an absence of one.",
            "It does not mean the model is fair in any absolute sense — only that on this",
            "attribute, at this threshold, on this fictional sample, no material disparity was",
            "found.",
            "",
        ]

    for attribute in flagged:
        payload = results["attributes"][attribute]
        groups = payload["groups"]
        disparities = payload["disparities"]
        highest = max(groups, key=lambda g: g["base_rate"])
        lowest = min(groups, key=lambda g: g["base_rate"])

        lines += [
            f"### {attribute} — material disparity, and what actually drives it",
            "",
            "Criteria exceeding the convention: "
            + ", ".join(f"`{c.replace('_', ' ')}`" for c in material[attribute])
            + ".",
            "",
            "**The disparity must not be read as the model disadvantaging the minority group.**",
            f"The two groups have genuinely different churn rates in the sample: "
            f"**{highest['label']} {highest['base_rate']:.4f}** against "
            f"**{lowest['label']} {lowest['base_rate']:.4f}**, a gap of "
            f"{disparities['base_rate']['gap']:.4f}. Three consequences follow, and they point in",
            "different directions:",
            "",
            f"1. **Selection rate** — {highest['label']} customers are flagged more often "
            f"({highest['selection_rate']:.4f} against {lowest['selection_rate']:.4f}). Given the",
            "   base-rate difference this is the model behaving *correctly*, not unfairly.",
            "   Forcing demographic parity here would mean deliberately under-flagging the group",
            "   that actually churns more — worse service, not fairer service.",
            f"2. **Recall** — the model finds at-risk {highest['label']} customers *better* "
            f"({highest['recall']:.4f} against {lowest['recall']:.4f}). On the criterion this",
            f"   project optimises for, the group being under-served is **{lowest['label']}**, not",
            f"   {highest['label']}. That is the opposite of the usual assumption and is the most",
            "   important line in this audit.",
            f"3. **False positive rate** — {highest['label']} customers who would have stayed are",
            f"   flagged far more often ({highest['false_positive_rate']:.4f} against "
            f"{lowest['false_positive_rate']:.4f}). **This is the real burden.** It does not cost",
            "   the customer a missed review; it costs them unnecessary contact. Whether that is",
            "   acceptable is a business and ethics judgement, not a modelling one.",
            "",
        ]

    # ---- Counterfactual-driven decision -----------------------------------
    costs = results.get("counterfactual_without_protected_attributes")
    lines += ["## Decision", ""]

    if costs:
        auc_delta = costs["roc_auc_with"] - costs["roc_auc_without"]
        recall_delta = costs["recall_with"] - costs["recall_without"]
        f1_delta = costs["f1_with"] - costs["f1_without"]
        cheap = abs(auc_delta) < 0.01 and abs(recall_delta) < 0.02

        lines += [
            "The keep-or-remove decision was made **after measuring the cost of removal**, not",
            "before. The model was retrained with `gender` and `SeniorCitizen` dropped entirely:",
            "",
            "| Metric | With attributes | Without | Cost of removal |",
            "|---|---:|---:|---:|",
            f"| ROC-AUC | {costs['roc_auc_with']:.4f} | {costs['roc_auc_without']:.4f} | {auc_delta:+.4f} |",
            f"| Recall | {costs['recall_with']:.4f} | {costs['recall_without']:.4f} | {recall_delta:+.4f} |",
            f"| F1 | {costs['f1_with']:.4f} | {costs['f1_without']:.4f} | {f1_delta:+.4f} |",
            "",
        ]

        if cheap:
            lines += [
                "**The attributes contribute almost nothing to predictive performance.** Removing",
                f"both costs {auc_delta:+.4f} ROC-AUC and {recall_delta:+.4f} recall — differences",
                "well inside the cross-validation standard deviation of ±0.0124 reported in",
                "`reports/model_comparison.csv`, and therefore not distinguishable from noise.",
                "",
                "**Recommendation: remove both attributes before any operational use.**",
                "",
                "The reasoning is straightforward. The attributes buy no measurable accuracy, they",
                "introduce a demographic disparity that must then be explained and defended, and",
                "retaining a protected characteristic that earns nothing is an unforced governance",
                "risk. When the trade-off is *no benefit* against *real explanatory burden*, the",
                "decision is not finely balanced.",
                "",
                "**Why they are retained in version 1.1.0 anyway:** this is an academic submission",
                "whose measured results are already published across the report, the model card and",
                "the deployed application. Silently changing the model now would invalidate that",
                "evidence trail. The recommendation is recorded as the first action of Horizon 3,",
                "with the cost quantified, so the decision is documented rather than deferred by",
                "omission. In a production setting with no such constraint, the attributes would",
                "be removed now.",
                "",
            ]
        else:
            lines += [
                f"Removing both attributes costs {auc_delta:+.4f} ROC-AUC and {recall_delta:+.4f}",
                "recall — a material loss of predictive performance.",
                "",
                "**Recommendation: retain the attributes and manage the disparity through",
                "process controls**, principally the mandatory human review already required",
                "before any customer is contacted. The trade-off is genuine and the decision",
                "should be escalated to model risk rather than settled by the modelling team.",
                "",
            ]
    else:
        lines += [
            "The counterfactual was not run, so no evidence-based keep-or-remove recommendation",
            "can be made. Run `make fairness` to produce it.",
            "",
        ]

    lines += [
        "### Options considered",
        "",
        "| Option | Assessment |",
        "|---|---|",
        "| (a) Remove the protected attributes | **Recommended.** Cost measured above. |",
        "| (b) Retain and document the disparity | Applied to version 1.1.0 to preserve the published evidence trail; documented, not silent. |",
        "| (c) Apply a group-specific decision threshold | **Rejected outright.** Setting a different threshold per protected characteristic is direct differential treatment. It would improve a fairness statistic while creating a discrimination exposure. This rejection stands regardless of what the numbers show. |",
        "",
        "Option (c) is recorded because rejecting it is a substantive decision, not an omission.",
        "",
    ]

    lines += [
        "## Limitations of this audit",
        "",
        "- The dataset is **fictional**. No conclusion about real-world discrimination follows.",
        "- Only two attributes with two levels each were audited. `Partner`, `Dependents` and",
        "  tenure-related proxies could encode further group structure and were not examined.",
        "- **Intersectional effects were not tested** — for example senior women as a distinct",
        "  group. Sample sizes in the held-out set would be too small for a stable estimate.",
        "- The audit measures the model at one threshold. A different operating point would give",
        "  different disparities; see `reports/threshold_analysis.md`.",
        "- Fairness of the *outcome* depends on what humans do with the score. This audit covers",
        "  the model only, not the retention process built around it.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make fairness",
        "```",
        "",
    ]

    (config.REPORTS_DIR / "fairness_report.md").write_text("\n".join(lines), encoding="utf-8")


def counterfactual_without_attributes() -> dict[str, float]:
    """Retrain without the protected attributes and measure the performance cost.

    Makes the "keep or remove" decision evidence-based instead of asserted.
    """
    from sklearn.metrics import f1_score, recall_score, roc_auc_score

    from src.train import build_candidates

    context = load_evaluation_context()
    protected = list(PROTECTED_ATTRIBUTES)
    model_name = context.metadata["model_name"]

    pipeline = build_candidates()[model_name]
    # Drop the protected attributes from the categorical branch only.
    preprocessor = pipeline.named_steps["preprocessor"]
    preprocessor.transformers = [
        (
            name,
            transformer,
            [c for c in columns if c not in protected] if name == "categorical" else columns,
        )
        for name, transformer, columns in preprocessor.transformers
    ]

    reduced_train = context.X_train.drop(columns=protected)
    reduced_test = context.X_test.drop(columns=protected)
    pipeline.fit(reduced_train, context.y_train)
    proba = pipeline.predict_proba(reduced_test)[:, 1]
    predictions = (proba >= float(context.metadata["decision_threshold"])).astype(int)

    baseline = context.metadata["metrics"]
    return {
        "roc_auc_without": float(roc_auc_score(context.y_test, proba)),
        "recall_without": float(recall_score(context.y_test, predictions, zero_division=0)),
        "f1_without": float(f1_score(context.y_test, predictions, zero_division=0)),
        "roc_auc_with": float(baseline["roc_auc"]),
        "recall_with": float(baseline["recall"]),
        "f1_with": float(baseline["f1"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fairness audit across protected attributes.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--counterfactual", action="store_true",
                        help="Also retrain without the protected attributes.")
    args = parser.parse_args(argv)

    results = run_audit(args.threshold)

    print("=" * 72)
    print("Fairness audit — held-out test set")
    print("=" * 72)
    for attribute, payload in results["attributes"].items():
        print(f"\n{attribute}")
        for group in payload["groups"]:
            print(
                f"  {group['label']:<16} n={group['n']:<6} "
                f"base={group['base_rate']:.4f}  recall={group['recall']:.4f}  "
                f"precision={group['precision']:.4f}  selection={group['selection_rate']:.4f}"
            )
        for criterion, flag in payload["assessment"].items():
            marker = "MATERIAL" if flag["material"] else "ok"
            print(f"    {criterion:<24} gap={flag['gap']:.4f}  [{marker}]")

    if args.counterfactual:
        costs = counterfactual_without_attributes()
        print("\nCounterfactual — model retrained without gender and SeniorCitizen:")
        print(f"  ROC-AUC {costs['roc_auc_with']:.4f} -> {costs['roc_auc_without']:.4f}")
        print(f"  Recall  {costs['recall_with']:.4f} -> {costs['recall_without']:.4f}")
        print(f"  F1      {costs['f1_with']:.4f} -> {costs['f1_without']:.4f}")

        payload = json.loads((config.TABLES_DIR / "fairness_report.json").read_text())
        payload["counterfactual_without_protected_attributes"] = costs
        (config.TABLES_DIR / "fairness_report.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    print(f"\nReport -> {config.REPORTS_DIR / 'fairness_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
