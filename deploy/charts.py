"""Themed matplotlib charts for the churn intelligence application.

Every chart states its own units and reference points in the axis or title, so
none of them relies on colour alone to carry meaning. All figures inherit the
active theme's tokens from :mod:`theme`.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import theme as theme_module  # noqa: E402


def _new_figure(active_theme: str, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    with plt.rc_context(theme_module.matplotlib_rc(active_theme)):
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def risk_position_chart(
    probability: float,
    active_theme: str,
    low_max: float,
    medium_max: float,
) -> plt.Figure:
    """Horizontal band chart placing the score within the three risk tiers."""
    tokens = theme_module.tokens_for(active_theme)
    with plt.rc_context(theme_module.matplotlib_rc(active_theme)):
        fig, ax = plt.subplots(figsize=(6.6, 1.9))
        ax.grid(False)

        bands = [
            (0.0, low_max, tokens["risk_low"], "Low"),
            (low_max, medium_max, tokens["risk_medium"], "Medium"),
            (medium_max, 1.0, tokens["risk_high"], "High"),
        ]
        for start, end, colour, label in bands:
            ax.barh(0, end - start, left=start, height=0.42, color=colour, alpha=0.30)
            ax.text(
                (start + end) / 2,
                -0.46,
                f"{label}\n{start:.2f}–{end:.2f}",
                ha="center",
                va="top",
                fontsize=8,
                color=tokens["text_muted"],
            )

        # Confined to the band itself so the marker cannot overrun the tier labels.
        ax.axvline(probability, color=tokens["text"], linewidth=2.4, ymin=0.49, ymax=0.78)
        ax.annotate(
            f"{probability:.1%}",
            xy=(probability, 0.24),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=tokens["text"],
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.95, 0.55)
        ax.set_yticks([])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_xlabel("Estimated churn probability")
        ax.set_title("Where this score falls within the risk bands", fontsize=10, loc="left")
        for spine in ("left", "bottom"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
    return fig


def segment_reference_chart(
    selections: dict[str, str],
    reference: dict[str, Any],
    active_theme: str,
) -> plt.Figure | None:
    """Training-split churn rate for each of this customer's selected segments.

    These are descriptive sample rates, not model outputs and not predictions
    about this individual customer.
    """
    tokens = theme_module.tokens_for(active_theme)
    by_category = reference.get("by_category", {})
    overall = float(reference.get("overall_training_churn_rate", 0.0)) * 100

    labels: list[str] = []
    values: list[float] = []
    counts: list[int] = []
    for column, level in selections.items():
        entry = by_category.get(column, {}).get(str(level))
        if entry is None:
            continue
        labels.append(f"{column}\n{level}")
        values.append(float(entry["churn_rate"]) * 100)
        counts.append(int(entry["customers"]))

    if not labels:
        return None

    with plt.rc_context(theme_module.matplotlib_rc(active_theme)):
        fig, ax = plt.subplots(figsize=(6.6, 3.5))
        ax.set_axisbelow(True)
        positions = range(len(labels))
        ax.barh(list(positions), values, color=tokens["chart_series"], height=0.6)
        ax.axvline(
            overall,
            color=tokens["chart_reference"],
            linestyle="--",
            linewidth=1.3,
            label=f"Overall training churn rate {overall:.1f}%",
        )
        for y, (value, count) in enumerate(zip(values, counts)):
            ax.text(
                value + 0.8,
                y,
                f"{value:.1f}%  (n={count:,})",
                va="center",
                fontsize=8.5,
                color=tokens["text_muted"],
            )
        ax.set_yticks(list(positions), labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(max(values) * 1.42, overall * 1.5))
        ax.set_xlabel("Churn rate in the training sample (%)")
        ax.set_title(
            "Sample churn rate for this customer's segments\n(descriptive context, not a prediction)",
            fontsize=10,
            loc="left",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
    return fig


def score_distribution_chart(
    probability: float,
    reference: dict[str, Any],
    active_theme: str,
) -> plt.Figure | None:
    """This score against the distribution of held-out test-set scores."""
    tokens = theme_module.tokens_for(active_theme)
    histogram = reference.get("test_score_histogram")
    if not histogram:
        return None

    edges = histogram["bin_edges"]
    counts = histogram["counts"]
    population = histogram.get("population", sum(counts))
    widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]

    # Share of the scored test population at or below this score.
    below = sum(count for centre, count in zip(centres, counts) if centre <= probability)
    percentile = below / population * 100 if population else 0.0

    with plt.rc_context(theme_module.matplotlib_rc(active_theme)):
        fig, ax = plt.subplots(figsize=(6.6, 3.3))
        ax.set_axisbelow(True)
        ax.bar(
            centres,
            counts,
            width=[w * 0.92 for w in widths],
            color=tokens["chart_series"],
            alpha=0.75,
        )
        ax.axvline(probability, color=tokens["accent"], linewidth=2.2)
        ax.annotate(
            f"This customer: {probability:.1%}\n≈{percentile:.0f}% of scored customers are lower",
            xy=(probability, max(counts) * 0.92),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left" if probability < 0.6 else "right",
            fontsize=8.5,
            color=tokens["text"],
        )
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_xlabel("Estimated churn probability")
        ax.set_ylabel("Customers")
        ax.set_title(
            f"Score distribution across the held-out test set (n={population:,})",
            fontsize=10,
            loc="left",
        )
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
    return fig
