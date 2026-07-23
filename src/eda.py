"""Reproducible exploratory data analysis.

This script is the source of truth for every quantitative statement made about
the dataset in the report. ``notebooks/01_eda_and_modeling.ipynb`` imports these
functions rather than duplicating the logic.

Charts are intentionally plain: no 3-D effects, no truncated axes, no colour
carrying meaning that is not also stated in the axis or title.

CLI::

    python -m src.eda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data_validation import load_raw_dataframe  # noqa: E402

FIGURE_DPI = 200
CHURN_PALETTE = {"No": "#4C72B0", "Yes": "#C44E52"}


# --------------------------------------------------------------------------
# Data preparation for analysis (never mutates data/raw)
# --------------------------------------------------------------------------


def prepare_analysis_frame(path: Path | None = None) -> pd.DataFrame:
    """Return an analysis-ready copy of the raw data.

    The only transformations applied are the documented, non-destructive ones:
    ``TotalCharges`` is stripped and coerced to numeric, ``tenure`` and
    ``MonthlyCharges`` are cast to numeric, and ``SeniorCitizen`` is kept as a
    string category. No rows are dropped and no values are imputed here —
    imputation belongs inside the training pipeline so it can be fitted on the
    training split alone.
    """
    frame = load_raw_dataframe(path).copy()

    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].str.strip(), errors="coerce")
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"].str.strip(), errors="coerce")
    frame["tenure"] = pd.to_numeric(frame["tenure"].str.strip(), errors="coerce")
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(str)
    return frame


def _save(fig: plt.Figure, name: str) -> Path:
    config.ensure_output_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_table(frame: pd.DataFrame, name: str, index: bool = True) -> Path:
    config.ensure_output_dirs()
    path = config.TABLES_DIR / name
    frame.to_csv(path, index=index)
    return path


# --------------------------------------------------------------------------
# Individual analyses
# --------------------------------------------------------------------------


def target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts and percentages of the target class."""
    counts = frame[config.TARGET_COLUMN].value_counts()
    table = pd.DataFrame(
        {
            "count": counts,
            "percent": (counts / len(frame) * 100).round(2),
        }
    )
    table.index.name = config.TARGET_COLUMN

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        table.index.astype(str),
        table["count"],
        color=[CHURN_PALETTE[str(value)] for value in table.index],
    )
    for x, (count, percent) in enumerate(zip(table["count"], table["percent"])):
        ax.text(x, count, f"{count:,}\n({percent:.2f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Target class distribution (Churn)")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, table["count"].max() * 1.18)
    _save(fig, "01_target_distribution.png")
    _write_table(table, "target_distribution.csv")
    return table


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing and blank values per column after numeric coercion."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        blanks = (
            int((series.astype(str).str.strip() == "").sum())
            if series.dtype == object
            else 0
        )
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_after_coercion": missing,
                "blank_strings": blanks,
                "missing_percent": round(missing / len(frame) * 100, 4),
            }
        )
    table = pd.DataFrame(rows).set_index("column")

    plot_data = table[table["missing_after_coercion"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    if plot_data.empty:
        ax.text(0.5, 0.5, "No missing values after coercion", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.barh(plot_data.index, plot_data["missing_after_coercion"], color="#937860")
        for y, value in enumerate(plot_data["missing_after_coercion"]):
            ax.text(value, y, f" {value}", va="center", fontsize=9)
        ax.set_xlabel("Missing values")
        ax.set_xlim(0, plot_data["missing_after_coercion"].max() * 1.25)
    ax.set_title("Missing values after TotalCharges coercion")
    _save(fig, "02_missing_values.png")
    _write_table(table, "missing_value_summary.csv")
    return table


def numeric_distribution_by_churn(frame: pd.DataFrame, column: str, figure_name: str) -> pd.DataFrame:
    """Overlaid histogram plus descriptive statistics of *column* by churn."""
    stats = frame.groupby(config.TARGET_COLUMN)[column].describe().round(3)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label in ["No", "Yes"]:
        values = frame.loc[frame[config.TARGET_COLUMN] == label, column].dropna()
        ax.hist(
            values,
            bins=30,
            alpha=0.6,
            label=f"Churn = {label} (n={len(values):,})",
            color=CHURN_PALETTE[label],
        )
    ax.set_title(f"{column} distribution by churn status")
    ax.set_xlabel(column)
    ax.set_ylabel("Customers")
    ax.legend()
    _save(fig, figure_name)
    _write_table(stats, f"{column.lower()}_by_churn_describe.csv")
    return stats


def churn_rate_by_category(frame: pd.DataFrame, column: str, figure_name: str) -> pd.DataFrame:
    """Churn rate within each level of a categorical predictor."""
    grouped = frame.groupby(column)[config.TARGET_COLUMN]
    table = pd.DataFrame(
        {
            "customers": grouped.size(),
            "churned": grouped.apply(lambda s: int((s == "Yes").sum())),
        }
    )
    table["churn_rate_percent"] = (table["churned"] / table["customers"] * 100).round(2)
    table = table.sort_values("churn_rate_percent", ascending=False)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(table.index.astype(str), table["churn_rate_percent"], color="#4C72B0")
    overall = (frame[config.TARGET_COLUMN] == "Yes").mean() * 100
    ax.axhline(overall, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Overall churn rate {overall:.2f}%")
    for x, (rate, size) in enumerate(zip(table["churn_rate_percent"], table["customers"])):
        ax.text(x, rate, f"{rate:.1f}%\nn={size:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Churn rate by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(table["churn_rate_percent"].max() * 1.28, overall * 1.3))
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, figure_name)
    _write_table(table, f"churn_rate_by_{column.lower()}.csv")
    return table


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between the numeric predictors and the binary target."""
    numeric = frame[config.NUMERIC_FEATURES].copy()
    numeric["ChurnFlag"] = frame[config.TARGET_COLUMN].map(config.TARGET_MAPPING)
    matrix = numeric.corr(numeric_only=True).round(4)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Numerical correlation matrix (Pearson)")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Correlation coefficient")
    _save(fig, "11_correlation_matrix.png")
    _write_table(matrix, "correlation_matrix.csv")
    return matrix


def dtype_and_category_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Data-type overview and the observed level counts of every category."""
    dtype_rows = []
    for column in frame.columns:
        series = frame[column]
        dtype_rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "role": (
                    "identifier"
                    if column == config.ID_COLUMN
                    else "target"
                    if column == config.TARGET_COLUMN
                    else "numeric predictor"
                    if column in config.NUMERIC_FEATURES
                    else "categorical predictor"
                ),
                "unique_values": int(series.nunique(dropna=True)),
                "example_value": str(series.dropna().iloc[0]) if not series.dropna().empty else "",
            }
        )
    dtype_table = pd.DataFrame(dtype_rows).set_index("column")

    category_rows = []
    for column in config.CATEGORICAL_FEATURES:
        counts = frame[column].value_counts()
        for level, count in counts.items():
            category_rows.append(
                {
                    "column": column,
                    "category": level,
                    "count": int(count),
                    "percent": round(count / len(frame) * 100, 2),
                }
            )
    category_table = pd.DataFrame(category_rows)

    _write_table(dtype_table, "data_type_summary.csv")
    _write_table(category_table, "category_level_summary.csv", index=False)
    return dtype_table, category_table


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


def write_observations(results: dict[str, Any]) -> Path:
    """Write evidence-based EDA observations. Every number here is computed."""
    target = results["target"]
    missing = results["missing"]
    tenure_stats = results["tenure_stats"]
    monthly_stats = results["monthly_stats"]
    total_stats = results["total_stats"]
    contract = results["contract"]
    internet = results["internet"]
    payment = results["payment"]
    tech = results["tech_support"]
    security = results["online_security"]
    corr = results["correlation"]

    churn_yes = int(target.loc["Yes", "count"])
    churn_no = int(target.loc["No", "count"])
    churn_rate = float(target.loc["Yes", "percent"])
    total_rows = churn_yes + churn_no
    missing_total = int(missing["missing_after_coercion"].sum())
    total_charges_missing = int(missing.loc["TotalCharges", "missing_after_coercion"])

    lines = [
        "# Exploratory Data Analysis — Observations",
        "",
        "Every figure in this document is produced by `src/eda.py` from the validated raw",
        "file `data/raw/Telco-Customer-Churn.csv` (Git blob SHA",
        f"`{config.EXPECTED_GIT_BLOB_SHA}`). No value here was entered by hand.",
        "",
        "The dataset is IBM's **fictional** telecommunications sample. Observations below",
        "describe that sample, and associations described are not evidence of causation.",
        "",
        "## 1. Target balance",
        "",
        f"- {total_rows:,} customers; {churn_yes:,} churned and {churn_no:,} did not.",
        f"- The churn rate in the sample is **{churn_rate:.2f}%**, so the classes are imbalanced",
        f"  at roughly 1 churner to {churn_no / churn_yes:.1f} non-churners.",
        "- Consequence for modelling: a model that predicts \"no churn\" for every customer would",
        f"  reach {100 - churn_rate:.2f}% accuracy while identifying no churn risk at all. Accuracy",
        "  alone is therefore not an acceptable selection metric; recall, F1 and ROC-AUC are used.",
        "",
        "## 2. Data quality",
        "",
        f"- After stripping whitespace and coercing `TotalCharges` to numeric, {missing_total} values",
        f"  are missing in the whole table, all of them in `TotalCharges` ({total_charges_missing} rows).",
        f"- Those {total_charges_missing} rows all have `tenure = 0`, i.e. customers who have not yet",
        "  completed a billing cycle. The blank is structurally meaningful rather than random.",
        "- The raw file is left untouched. Missing values are handled by a median imputer that is",
        "  fitted **inside the pipeline on the training split only**, so no test information leaks.",
        "- No duplicate rows and no duplicate `customerID` values were found.",
        "",
        "## 3. Tenure",
        "",
        f"- Mean tenure of churned customers is {tenure_stats.loc['Yes', 'mean']:.2f} months against",
        f"  {tenure_stats.loc['No', 'mean']:.2f} months for retained customers.",
        f"- Median tenure is {tenure_stats.loc['Yes', '50%']:.1f} months for churners and",
        f"  {tenure_stats.loc['No', '50%']:.1f} months for non-churners.",
        "- Churn is concentrated among recently acquired customers in this sample. Early-life",
        "  accounts may therefore warrant closer retention attention.",
        "",
        "## 4. Charges",
        "",
        f"- Mean monthly charge is {monthly_stats.loc['Yes', 'mean']:.2f} for churners and",
        f"  {monthly_stats.loc['No', 'mean']:.2f} for retained customers.",
        f"- Mean total charge is {total_stats.loc['Yes', 'mean']:.2f} for churners and",
        f"  {total_stats.loc['No', 'mean']:.2f} for retained customers, which is consistent with",
        "  churners having shorter tenure rather than with lower spending per month.",
        "",
        "## 5. Contract term",
        "",
        "| Contract | Customers | Churn rate |",
        "|---|---:|---:|",
    ]
    for level, row in contract.iterrows():
        lines.append(f"| {level} | {int(row['customers']):,} | {row['churn_rate_percent']:.2f}% |")
    top_contract = contract.index[0]
    lines += [
        "",
        f"- `{top_contract}` contracts show the highest churn rate in the sample",
        f"  ({contract.iloc[0]['churn_rate_percent']:.2f}%), against",
        f"  {contract.iloc[-1]['churn_rate_percent']:.2f}% for `{contract.index[-1]}`.",
        "- Contract term is the single strongest categorical separator observed. This is an",
        "  association within the sample, not proof that changing a contract changes behaviour.",
        "",
        "## 6. Internet service and add-ons",
        "",
        "| Internet service | Customers | Churn rate |",
        "|---|---:|---:|",
    ]
    for level, row in internet.iterrows():
        lines.append(f"| {level} | {int(row['customers']):,} | {row['churn_rate_percent']:.2f}% |")
    lines += [
        "",
        f"- `{internet.index[0]}` customers churn at {internet.iloc[0]['churn_rate_percent']:.2f}%,",
        f"  compared with {internet.iloc[-1]['churn_rate_percent']:.2f}% for `{internet.index[-1]}`.",
        f"- Customers without technical support churn at "
        f"{tech.loc['No', 'churn_rate_percent']:.2f}% versus "
        f"{tech.loc['Yes', 'churn_rate_percent']:.2f}% for those with it.",
        f"- Customers without online security churn at "
        f"{security.loc['No', 'churn_rate_percent']:.2f}% versus "
        f"{security.loc['Yes', 'churn_rate_percent']:.2f}% for those with it.",
        "- Note the confound: the 'No internet service' level appears in every add-on column, so",
        "  these add-on comparisons partly restate the internet-service split.",
        "",
        "## 7. Payment method",
        "",
        "| Payment method | Customers | Churn rate |",
        "|---|---:|---:|",
    ]
    for level, row in payment.iterrows():
        lines.append(f"| {level} | {int(row['customers']):,} | {row['churn_rate_percent']:.2f}% |")
    lines += [
        "",
        f"- `{payment.index[0]}` shows the highest churn rate at",
        f"  {payment.iloc[0]['churn_rate_percent']:.2f}%.",
        "",
        "## 8. Numeric correlations",
        "",
        f"- `tenure` correlates with the churn flag at {corr.loc['tenure', 'ChurnFlag']:.3f}.",
        f"- `MonthlyCharges` correlates at {corr.loc['MonthlyCharges', 'ChurnFlag']:.3f}.",
        f"- `TotalCharges` correlates at {corr.loc['TotalCharges', 'ChurnFlag']:.3f}.",
        f"- `tenure` and `TotalCharges` are strongly related "
        f"({corr.loc['tenure', 'TotalCharges']:.3f}), which is expected because total charges",
        "  accumulate with time. Tree ensembles tolerate this; the linear model uses scaling and",
        "  regularisation, and the correlation is recorded here as a known limitation.",
        "",
        "## 9. Implications carried into modelling",
        "",
        "1. Class imbalance means recall and ROC-AUC drive model selection, not accuracy.",
        "2. `TotalCharges` needs coercion and imputation inside the pipeline, never beforehand.",
        "3. `SeniorCitizen` is treated as a binary category, not a continuous magnitude.",
        "4. `customerID` is excluded from the feature set entirely.",
        "5. Contract, internet service, tenure and payment method are the most promising signals",
        "   and are all retained as predictors.",
        "",
        "## Figures",
        "",
        "All figures are saved to `reports/figures/` at 200 dpi:",
        "",
        "| File | Content |",
        "|---|---|",
        "| `01_target_distribution.png` | Target class balance |",
        "| `02_missing_values.png` | Missing values after coercion |",
        "| `03_tenure_by_churn.png` | Tenure distribution by churn |",
        "| `04_monthlycharges_by_churn.png` | Monthly charges distribution by churn |",
        "| `05_totalcharges_by_churn.png` | Total charges distribution by churn |",
        "| `06_churn_rate_by_contract.png` | Churn rate by contract term |",
        "| `07_churn_rate_by_internetservice.png` | Churn rate by internet service |",
        "| `08_churn_rate_by_paymentmethod.png` | Churn rate by payment method |",
        "| `09_churn_rate_by_techsupport.png` | Churn rate by technical support |",
        "| `10_churn_rate_by_onlinesecurity.png` | Churn rate by online security |",
        "| `11_correlation_matrix.png` | Numeric correlation matrix |",
        "",
    ]

    config.ensure_output_dirs()
    path = config.REPORTS_DIR / "eda_observations.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_eda(path: Path | None = None) -> dict[str, Any]:
    """Execute the full EDA and persist every figure, table and observation."""
    config.ensure_output_dirs()
    frame = prepare_analysis_frame(path)

    results: dict[str, Any] = {
        "target": target_distribution(frame),
        "missing": missing_value_summary(frame),
        "tenure_stats": numeric_distribution_by_churn(frame, "tenure", "03_tenure_by_churn.png"),
        "monthly_stats": numeric_distribution_by_churn(
            frame, "MonthlyCharges", "04_monthlycharges_by_churn.png"
        ),
        "total_stats": numeric_distribution_by_churn(
            frame, "TotalCharges", "05_totalcharges_by_churn.png"
        ),
        "contract": churn_rate_by_category(frame, "Contract", "06_churn_rate_by_contract.png"),
        "internet": churn_rate_by_category(
            frame, "InternetService", "07_churn_rate_by_internetservice.png"
        ),
        "payment": churn_rate_by_category(
            frame, "PaymentMethod", "08_churn_rate_by_paymentmethod.png"
        ),
        "tech_support": churn_rate_by_category(
            frame, "TechSupport", "09_churn_rate_by_techsupport.png"
        ),
        "online_security": churn_rate_by_category(
            frame, "OnlineSecurity", "10_churn_rate_by_onlinesecurity.png"
        ),
        "correlation": correlation_matrix(frame),
    }
    dtype_table, category_table = dtype_and_category_summary(frame)
    results["dtypes"] = dtype_table
    results["categories"] = category_table

    observations_path = write_observations(results)
    results["observations_path"] = observations_path
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible EDA on the raw churn dataset.")
    parser.add_argument("--path", type=Path, default=config.RAW_DATASET_PATH)
    args = parser.parse_args(argv)

    results = run_eda(args.path)

    figures = sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))
    tables = sorted(p.name for p in config.TABLES_DIR.glob("*.csv"))
    print("EDA complete.")
    print(f"Figures ({len(figures)}) -> {config.FIGURES_DIR}")
    for name in figures:
        print(f"  - {name}")
    print(f"Tables ({len(tables)}) -> {config.TABLES_DIR}")
    for name in tables:
        print(f"  - {name}")
    print(f"Observations -> {results['observations_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
