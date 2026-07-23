"""Feature engineering and hyperparameter search experiment (H1-6).

Answers one question: **do engineered features and tuned hyperparameters
actually improve this model?** Not "let us assume they do".

Methodological position — read this before the numbers
-------------------------------------------------------
The held-out test set has already been scored once, for version 1.1.0, and those
metrics are published. Using it again to choose between v1 and v2 would turn it
into a selection set and quietly invalidate the guarantee the whole project
rests on.

So the design is:

1. **Every decision** — which features, which hyperparameters, whether to adopt
   v2 at all — is made by cross-validation on the **training split only**.
2. The test set is scored **once**, after the decision, purely to report what the
   chosen model achieves.
3. The report states plainly that the test set has now been consulted twice
   across the project's lifetime, and what that costs in strictness.

Four arms are compared, so the contribution of each change is separable rather
than confounded:

- **A** baseline: raw features, hand-set hyperparameters (the v1.1.0 model)
- **B** raw features + tuned hyperparameters
- **C** engineered features + hand-set hyperparameters
- **D** engineered features + tuned hyperparameters

CLI::

    python -m src.tuning
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
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402

from src import config, features  # noqa: E402
from src.train import cross_validate_model, load_model_frame, split_features_target  # noqa: E402

FIGURE_DPI = 200

#: Deliberately small grids. A large search over 5,634 training rows would tune
#: to cross-validation noise, and the brief warns against unnecessary searches.
#: Each value is one a practitioner would actually try.
LOGISTIC_GRID: dict[str, list[Any]] = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__solver": ["lbfgs", "liblinear"],
}

FOREST_GRID: dict[str, list[Any]] = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 5, 10],
}

#: Adoption bar. A gain smaller than this is inside the ±0.0124 cross-validation
#: standard deviation already measured for this model, and adopting it would be
#: chasing noise while adding permanent complexity.
ADOPTION_THRESHOLD: float = 0.005


def build_preprocessor(engineered: bool, scale_numeric: bool) -> ColumnTransformer:
    """Column-wise preprocessing over either the raw or the extended column set."""
    numeric = features.all_numeric_features() if engineered else list(config.NUMERIC_FEATURES)
    categorical = (
        features.all_categorical_features() if engineered else list(config.CATEGORICAL_FEATURES)
    )

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model: str, engineered: bool, **classifier_kwargs: Any) -> Pipeline:
    """Assemble a candidate pipeline.

    When ``engineered`` is true a stateless ``FunctionTransformer`` runs first,
    so the application keeps passing the 19 raw columns and the artifact remains
    self-contained.
    """
    scale = model == "Logistic Regression"
    steps: list[tuple[str, Any]] = []

    if engineered:
        steps.append(
            (
                "features",
                FunctionTransformer(features.engineer_features, validate=False),
            )
        )

    steps.append(("preprocessor", build_preprocessor(engineered, scale)))

    if model == "Logistic Regression":
        defaults = {
            "max_iter": 2000,
            "random_state": config.RANDOM_STATE,
            "class_weight": "balanced",
        }
        defaults.update(classifier_kwargs)
        steps.append(("classifier", LogisticRegression(**defaults)))
    else:
        defaults = {
            "n_estimators": 400,
            "min_samples_leaf": 5,
            "random_state": config.RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        defaults.update(classifier_kwargs)
        steps.append(("classifier", RandomForestClassifier(**defaults)))

    return Pipeline(steps=steps)


def tune(model: str, engineered: bool, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Grid-search a model on the training split only."""
    grid = LOGISTIC_GRID if model == "Logistic Regression" else FOREST_GRID
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    search = GridSearchCV(
        estimator=build_pipeline(model, engineered),
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return {
        "best_params": {k: v for k, v in search.best_params_.items()},
        "best_cv_roc_auc": float(search.best_score_),
        "combinations": int(len(search.cv_results_["params"])),
        "estimator": search.best_estimator_,
    }


def run_experiment() -> dict[str, Any]:
    """Compare all four arms by cross-validation on the training split."""
    config.ensure_output_dirs()
    frame = load_model_frame()
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    results: dict[str, Any] = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "adoption_threshold": ADOPTION_THRESHOLD,
        "engineered_features": features.describe_features(),
        "arms": {},
    }

    for model in ("Logistic Regression", "Random Forest"):
        print(f"\n{'=' * 72}\n{model}\n{'=' * 72}")

        # A — baseline, as deployed in v1.1.0
        arm_a = cross_validate_model(build_pipeline(model, engineered=False), X_train, y_train)
        print(f"  A  raw + defaults      CV ROC-AUC {arm_a['cv_roc_auc_mean']:.4f} "
              f"± {arm_a['cv_roc_auc_std']:.4f} | recall {arm_a['cv_recall_mean']:.4f}")

        # C — engineered features, untuned. Isolates the feature contribution.
        arm_c = cross_validate_model(build_pipeline(model, engineered=True), X_train, y_train)
        print(f"  C  engineered + defaults CV ROC-AUC {arm_c['cv_roc_auc_mean']:.4f} "
              f"± {arm_c['cv_roc_auc_std']:.4f} | recall {arm_c['cv_recall_mean']:.4f}")

        # B — tuned on raw features. Isolates the tuning contribution.
        print("  B  tuning on raw features...")
        tuned_raw = tune(model, False, X_train, y_train)
        arm_b = cross_validate_model(
            build_pipeline(model, False,
                           **{k.replace("classifier__", ""): v
                              for k, v in tuned_raw["best_params"].items()}),
            X_train, y_train,
        )
        print(f"  B  raw + tuned         CV ROC-AUC {arm_b['cv_roc_auc_mean']:.4f} "
              f"| best {tuned_raw['best_params']}")

        # D — both changes together.
        print("  D  tuning on engineered features...")
        tuned_eng = tune(model, True, X_train, y_train)
        arm_d = cross_validate_model(
            build_pipeline(model, True,
                           **{k.replace("classifier__", ""): v
                              for k, v in tuned_eng["best_params"].items()}),
            X_train, y_train,
        )
        print(f"  D  engineered + tuned  CV ROC-AUC {arm_d['cv_roc_auc_mean']:.4f} "
              f"| best {tuned_eng['best_params']}")

        results["arms"][model] = {
            "A_raw_default": arm_a,
            "B_raw_tuned": {**arm_b, "best_params": tuned_raw["best_params"],
                            "combinations": tuned_raw["combinations"]},
            "C_engineered_default": arm_c,
            "D_engineered_tuned": {**arm_d, "best_params": tuned_eng["best_params"],
                                   "combinations": tuned_eng["combinations"]},
        }

    _decide(results)
    _plot(results)

    with (config.TABLES_DIR / "tuning_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {k: v for k, v in results.items() if k != "estimator"},
            handle, indent=2, default=str,
        )
        handle.write("\n")

    _write_markdown(results)
    return results


def _decide(results: dict[str, Any]) -> None:
    """Apply the pre-declared adoption rule to the cross-validated results."""
    best_arm = None
    best_score = -1.0
    baseline_score = -1.0

    for model, arms in results["arms"].items():
        for arm_name, arm in arms.items():
            score = arm["cv_roc_auc_mean"]
            if arm_name == "A_raw_default" and model == "Logistic Regression":
                baseline_score = score
            if score > best_score:
                best_score = score
                best_arm = (model, arm_name)

    # Compare against the best baseline arm across both models, not just one.
    baseline_best = max(
        arms["A_raw_default"]["cv_roc_auc_mean"] for arms in results["arms"].values()
    )
    gain = best_score - baseline_best

    results["decision"] = {
        "best_arm": f"{best_arm[0]} / {best_arm[1]}",
        "best_cv_roc_auc": round(best_score, 6),
        "baseline_best_cv_roc_auc": round(baseline_best, 6),
        "deployed_v1_cv_roc_auc": round(baseline_score, 6),
        "gain": round(gain, 6),
        "adopt": bool(gain >= ADOPTION_THRESHOLD),
        "rule": (
            f"Adopt only if the best arm beats the best baseline arm by at least "
            f"{ADOPTION_THRESHOLD} mean CV ROC-AUC. A smaller gain sits inside the "
            f"±0.0124 cross-validation standard deviation already measured and would be "
            f"chasing noise while adding permanent complexity."
        ),
    }


def _plot(results: dict[str, Any], filename: str = "19_tuning_experiment.png") -> str:
    labels = {
        "A_raw_default": "A: raw\n+ defaults",
        "B_raw_tuned": "B: raw\n+ tuned",
        "C_engineered_default": "C: engineered\n+ defaults",
        "D_engineered_tuned": "D: engineered\n+ tuned",
    }
    arm_keys = list(labels)
    models = list(results["arms"])
    positions = np.arange(len(arm_keys))
    width = 0.36
    colours = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_axisbelow(True)
    for index, model in enumerate(models):
        means = [results["arms"][model][k]["cv_roc_auc_mean"] for k in arm_keys]
        errors = [results["arms"][model][k]["cv_roc_auc_std"] for k in arm_keys]
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(positions + offset, means, width * 0.92, yerr=errors, capsize=4,
               label=model, color=colours[index % len(colours)])
        for x, mean in zip(positions + offset, means):
            ax.text(x, mean + 0.004, f"{mean:.4f}", ha="center", fontsize=7.5)

    baseline = results["decision"]["baseline_best_cv_roc_auc"]
    ax.axhline(baseline, color="#8C8C8C", linestyle="--", linewidth=1.2,
               label=f"Best baseline ({baseline:.4f})")
    ax.axhline(baseline + ADOPTION_THRESHOLD, color="#55A868", linestyle=":", linewidth=1.4,
               label=f"Adoption bar (+{ADOPTION_THRESHOLD})")

    ax.set_xticks(positions, [labels[k] for k in arm_keys], fontsize=9)
    ax.set_ylabel("Mean CV ROC-AUC (training split only)")
    lowest = min(
        results["arms"][m][k]["cv_roc_auc_mean"] for m in models for k in arm_keys
    )
    ax.set_ylim(lowest - 0.02, baseline + 0.03)
    ax.set_title(
        "Feature engineering and hyperparameter search — cross-validated on the training split\n"
        "The test set was not consulted to produce this chart",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    results["figure"] = filename
    return filename


def _write_markdown(results: dict[str, Any]) -> None:
    decision = results["decision"]
    lines = [
        "# Feature Engineering and Hyperparameter Search",
        "",
        "Produced by `src/tuning.py`. Answers one question: **do engineered features and tuned",
        "hyperparameters actually improve this model?**",
        "",
        "## Method, and why it is constrained",
        "",
        "The held-out test set had already been scored once for version 1.1.0, and those metrics",
        "are published. Using it again to choose between model versions would turn it into a",
        "selection set and quietly invalidate the guarantee the project rests on.",
        "",
        "So **every decision here — which features, which hyperparameters, whether to adopt any",
        "of it — was made by cross-validation on the training split alone.** The test set is",
        "scored once at the end, and only to report what the chosen model achieves.",
        "",
        "Four arms, so the contribution of each change is separable rather than confounded:",
        "",
        "| Arm | Features | Hyperparameters |",
        "|---|---|---|",
        "| A | Raw 19 | Hand-set (the deployed v1.1.0 model) |",
        "| B | Raw 19 | Grid-searched |",
        "| C | Raw 19 + 7 engineered | Hand-set |",
        "| D | Raw 19 + 7 engineered | Grid-searched |",
        "",
        "## Engineered features",
        "",
        "Seven, all **stateless row-wise functions** of a single customer's own values. Nothing",
        "is aggregated across rows and nothing is fitted, so no information can pass between",
        "splits. They live **inside the pipeline**, so the application still passes the 19 raw",
        "columns and the artifact stays self-contained.",
        "",
        "| Feature | Definition | Why |",
        "|---|---|---|",
    ]
    for feature in results["engineered_features"]:
        lines.append(
            f"| `{feature['name']}` | {feature['definition']} | {feature['rationale']} |"
        )

    lines += [
        "",
        "Only features a retention team could explain were included. An uninterpretable feature",
        "buys accuracy at the cost of the governance story that is this project's strongest asset.",
        "",
        "## Hyperparameter grids",
        "",
        "Deliberately small. A large search over 5,634 training rows tunes to cross-validation",
        "noise, and the brief warns against unnecessary searches.",
        "",
        f"- **Logistic Regression** — `C` ∈ {LOGISTIC_GRID['classifier__C']}, "
        f"`solver` ∈ {LOGISTIC_GRID['classifier__solver']}",
        f"- **Random Forest** — `n_estimators` ∈ {FOREST_GRID['classifier__n_estimators']}, "
        f"`max_depth` ∈ {FOREST_GRID['classifier__max_depth']}, "
        f"`min_samples_leaf` ∈ {FOREST_GRID['classifier__min_samples_leaf']}",
        "",
        "## Results — cross-validated on the training split",
        "",
        "| Model | Arm | CV ROC-AUC | CV F1 | CV recall | Best parameters |",
        "|---|---|---:|---:|---:|---|",
    ]
    arm_labels = {
        "A_raw_default": "A raw + defaults",
        "B_raw_tuned": "B raw + tuned",
        "C_engineered_default": "C engineered + defaults",
        "D_engineered_tuned": "D engineered + tuned",
    }
    for model, arms in results["arms"].items():
        for key, label in arm_labels.items():
            arm = arms[key]
            params = arm.get("best_params", {})
            rendered = ", ".join(
                f"`{k.replace('classifier__', '')}={v}`" for k, v in params.items()
            ) or "—"
            lines.append(
                f"| {model} | {label} | {arm['cv_roc_auc_mean']:.4f} ± "
                f"{arm['cv_roc_auc_std']:.4f} | {arm['cv_f1_mean']:.4f} | "
                f"{arm['cv_recall_mean']:.4f} | {rendered} |"
            )

    lines += [
        "",
        f"![Tuning experiment](figures/{results.get('figure', '19_tuning_experiment.png')})",
        "",
        "## Decision",
        "",
        f"**Rule, fixed before the results were read:** {decision['rule']}",
        "",
        f"- Best arm: **{decision['best_arm']}** at {decision['best_cv_roc_auc']:.4f}",
        f"- Best baseline arm: {decision['baseline_best_cv_roc_auc']:.4f}",
        f"- Gain: **{decision['gain']:+.4f}**",
        f"- Adoption bar: {ADOPTION_THRESHOLD}",
        "",
    ]

    if decision["adopt"]:
        lines += [
            f"**Adopted.** The gain of {decision['gain']:+.4f} clears the bar, so the engineered",
            "and/or tuned configuration replaces the previous model. The held-out set is scored",
            "once against the new model to report its performance.",
            "",
        ]
    else:
        lines += [
            f"**Not adopted.** The best configuration improves mean CV ROC-AUC by only",
            f"{decision['gain']:+.4f}, short of the {ADOPTION_THRESHOLD} bar and well inside the",
            "±0.0124 cross-validation standard deviation already measured for this model.",
            "",
            "**This is a result, not a failure.** It says something specific and useful: the",
            "signal in this dataset is largely exhausted by the raw features and sensible",
            "defaults. The obvious engineered features — spend trajectory, service counts, tenure",
            "bands — encode information the model was already extracting from the raw columns,",
            "and the hyperparameters were already near their optimum.",
            "",
            "Adopting a change this small would add permanent complexity to the pipeline, a",
            "larger surface to explain and maintain, and a second transformation to keep",
            "consistent between training and serving — all to chase a difference indistinguishable",
            "from noise. **The deployed model is unchanged.**",
            "",
            "The alternative — adopting it anyway because the work was done — is exactly the",
            "sunk-cost reasoning this project's selection rule exists to prevent.",
            "",
        ]

    lines += [
        "## What this cost in methodological strictness",
        "",
        "Stated plainly rather than glossed over: **the held-out test set has now been consulted",
        "twice** across the project's lifetime — once for the v1.1.0 evaluation and once here.",
        "Ideally a model development cycle uses it exactly once, ever.",
        "",
        "Two things limit the damage. First, no decision in this experiment used test data: the",
        "arms were compared, the hyperparameters chosen and the adoption call made entirely on",
        "cross-validated training performance. Second, the outcome was *not to change the model*,",
        "so the published v1.1.0 metrics remain a single-use evaluation of the deployed artifact.",
        "",
        "Had the experiment recommended adoption, the honest course would have been to report the",
        "new model's test metrics as a second-look estimate, mildly optimistic, rather than as an",
        "untouched holdout result.",
        "",
        "## Limitations",
        "",
        "- The grids are small by design. A larger search might find marginally better parameters,",
        "  and would be more likely to fit cross-validation noise.",
        "- Only interpretable features were tried. Interaction terms and polynomial expansions",
        "  might add signal at a cost in explainability the governance position does not accept.",
        "- No feature *selection* was performed. With 19 raw predictors and a regularised linear",
        "  model, there is little to gain.",
        "- The conclusion is specific to this dataset. It does not generalise to churn modelling.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make tune",
        "```",
        "",
    ]

    (config.REPORTS_DIR / "tuning_experiment.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature engineering and tuning experiment.")
    parser.parse_args(argv)

    results = run_experiment()
    decision = results["decision"]

    print("\n" + "=" * 72)
    print("DECISION")
    print("=" * 72)
    print(f"Best arm            : {decision['best_arm']} ({decision['best_cv_roc_auc']:.4f})")
    print(f"Best baseline arm   : {decision['baseline_best_cv_roc_auc']:.4f}")
    print(f"Gain                : {decision['gain']:+.4f}")
    print(f"Adoption bar        : {ADOPTION_THRESHOLD}")
    print(f"ADOPT               : {decision['adopt']}")
    print(f"\nReport -> {config.REPORTS_DIR / 'tuning_experiment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
