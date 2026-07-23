"""Tests for the Horizon 1 and 2 analysis modules.

Covers fairness, calibration, threshold and drift. Each test guards a specific
way the analysis could be silently wrong rather than merely absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis_base import load_evaluation_context


@pytest.fixture(scope="module")
def context():
    if not config.MODEL_PATH.is_file():
        pytest.skip("Model artifact missing — run `make train` first.")
    return load_evaluation_context()


# --------------------------------------------------------------------------
# Shared evaluation context
# --------------------------------------------------------------------------


def test_context_split_matches_training_configuration(context) -> None:
    """The analyses are only valid if they use the split the model was fitted on."""
    total = len(context.X_train) + len(context.X_test)
    assert total == config.EXPECTED_ROWS
    assert len(context.X_test) == pytest.approx(total * config.TEST_SIZE, abs=1)


def test_context_split_is_stratified(context) -> None:
    assert float(context.y_train.mean()) == pytest.approx(float(context.y_test.mean()), abs=0.01)


def test_context_probabilities_are_valid(context) -> None:
    proba = context.test_probabilities
    assert proba.shape == (len(context.y_test),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_context_excludes_identifier_and_target(context) -> None:
    for frame in (context.X_train, context.X_test):
        assert config.ID_COLUMN not in frame.columns
        assert config.TARGET_COLUMN not in frame.columns


# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------


def test_fairness_group_metrics_partition_the_test_set(context) -> None:
    """Subgroup counts must sum to the whole. A gap would mean silent dropping."""
    from src.fairness import compute_group_metrics

    for attribute in ("gender", "SeniorCitizen"):
        groups = compute_group_metrics(context, attribute, config.DECISION_THRESHOLD)
        assert sum(g.n for g in groups) == len(context.y_test)
        for group in groups:
            assert group.n == (
                group.true_negative + group.false_positive
                + group.false_negative + group.true_positive
            )


def test_fairness_rates_are_within_unit_interval(context) -> None:
    from src.fairness import compute_group_metrics

    for attribute in ("gender", "SeniorCitizen"):
        for group in compute_group_metrics(context, attribute, config.DECISION_THRESHOLD):
            for rate in (group.base_rate, group.selection_rate, group.recall,
                         group.precision, group.false_positive_rate, group.accuracy):
                assert 0.0 <= rate <= 1.0


def test_fairness_disparity_gap_is_consistent_with_max_and_min() -> None:
    from src.fairness import GroupMetrics, compute_disparities

    def make(label: str, recall: float) -> GroupMetrics:
        return GroupMetrics(
            attribute="x", level=label, label=label, n=100, base_rate=0.3,
            selection_rate=0.4, recall=recall, precision=0.5,
            false_positive_rate=0.2, false_negative_rate=0.1, accuracy=0.7,
            true_negative=50, false_positive=10, false_negative=10, true_positive=30,
        )

    disparities = compute_disparities([make("a", 0.9), make("b", 0.6)])
    assert disparities["equal_opportunity"]["gap"] == pytest.approx(0.3)
    assert disparities["equal_opportunity"]["ratio"] == pytest.approx(0.6 / 0.9)


def test_fairness_report_exists_and_reports_a_decision() -> None:
    path = config.REPORTS_DIR / "fairness_report.md"
    if not path.is_file():
        pytest.skip("Fairness report not generated — run `make fairness`.")
    content = path.read_text(encoding="utf-8")
    assert "## Decision" in content
    # A group-specific threshold must always be rejected, whatever the numbers say.
    assert "Rejected outright" in content
    assert "gender" in content and "SeniorCitizen" in content


def test_model_card_no_longer_claims_the_audit_is_missing() -> None:
    """The card said no audit had been done. Once done, it must not still say so."""
    if not config.MODEL_CARD_PATH.is_file():
        pytest.skip("Model card missing — run `make train`.")
    content = config.MODEL_CARD_PATH.read_text(encoding="utf-8")
    if not (config.REPORTS_DIR / "fairness_report.md").is_file():
        pytest.skip("Fairness audit not yet run.")
    assert "**No formal fairness audit**" not in content
    assert "no formal fairness audit has been carried out" not in content.lower()


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_expected_calibration_error_is_zero_for_perfect_predictions() -> None:
    from src.calibration import expected_calibration_error

    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0])
    ece, mce = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert mce == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_detects_overconfidence() -> None:
    from src.calibration import expected_calibration_error

    # Predicts 0.9 for everyone; only half actually churn.
    y_true = np.array([1, 0] * 50)
    y_proba = np.full(100, 0.9)
    ece, _ = expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece == pytest.approx(0.4, abs=0.01)


def test_calibration_bins_cover_every_row() -> None:
    from src.calibration import bin_table

    rng = np.random.default_rng(config.RANDOM_STATE)
    y_proba = rng.random(500)
    y_true = (rng.random(500) < y_proba).astype(int)
    rows = bin_table(y_true, y_proba, n_bins=10)
    assert sum(row["n"] for row in rows) == 500


def test_calibration_report_records_a_decision() -> None:
    path = config.REPORTS_DIR / "calibration_report.md"
    if not path.is_file():
        pytest.skip("Calibration report not generated — run `make calibration`.")
    content = path.read_text(encoding="utf-8")
    assert "## Decision" in content
    assert "Brier" in content


def test_calibration_does_not_change_ranking() -> None:
    """Calibration is monotonic, so ROC-AUC must be effectively unchanged."""
    path = config.TABLES_DIR / "calibration_report.json"
    if not path.is_file():
        pytest.skip("Calibration results not generated.")
    variants = json.loads(path.read_text(encoding="utf-8"))["variants"]
    aucs = [v["roc_auc"] for v in variants.values()]
    assert max(aucs) - min(aucs) < 0.01


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------


def test_threshold_sweep_recall_is_monotonically_non_increasing(context) -> None:
    """Raising the bar can never find more churners."""
    from src.threshold import sweep_thresholds

    rows = sweep_thresholds(context)
    recalls = [row["recall"] for row in rows]
    assert all(a >= b - 1e-9 for a, b in zip(recalls, recalls[1:]))


def test_threshold_sweep_flagged_count_decreases(context) -> None:
    from src.threshold import sweep_thresholds

    rows = sweep_thresholds(context)
    flagged = [row["flagged"] for row in rows]
    assert all(a >= b for a, b in zip(flagged, flagged[1:]))


def test_higher_miss_cost_lowers_the_optimal_threshold(context) -> None:
    """The core economic claim of the analysis must actually hold."""
    from src.threshold import optimal_thresholds, sweep_thresholds

    rows = sweep_thresholds(context)
    optima = optimal_thresholds(rows, ratios=(1.0, 20.0))
    cheap_miss, expensive_miss = optima[0], optima[1]
    assert expensive_miss["optimal_threshold"] < cheap_miss["optimal_threshold"]
    assert expensive_miss["recall"] > cheap_miss["recall"]


def test_threshold_confusion_counts_sum_to_the_test_set(context) -> None:
    from src.threshold import sweep_thresholds

    total = len(context.y_test)
    for row in sweep_thresholds(context):
        assert (row["true_negative"] + row["false_positive"]
                + row["false_negative"] + row["true_positive"]) == total


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions() -> None:
    from src.drift import _psi

    proportions = np.array([0.2, 0.3, 0.5])
    assert _psi(proportions, proportions) == pytest.approx(0.0, abs=1e-9)


def test_psi_increases_with_divergence() -> None:
    from src.drift import _psi

    baseline = np.array([0.2, 0.3, 0.5])
    mild = _psi(baseline, np.array([0.25, 0.3, 0.45]))
    severe = _psi(baseline, np.array([0.7, 0.2, 0.1]))
    assert 0 < mild < severe


def test_drift_detector_is_quiet_on_unshifted_data(context) -> None:
    """The false-positive half of the test. A detector that always fires is useless."""
    from src.drift import build_baseline, detect_drift

    baseline = build_baseline(save=False)
    report = detect_drift(context.X_test, baseline)
    assert report["overall_status"] == "stable", (
        f"Detector flagged unshifted holdout data: "
        f"{[f['feature'] for f in report['features'] if f['status'] != 'stable']}"
    )


def test_drift_detector_fires_on_shifted_data(context) -> None:
    """The other half. A detector never shown to fire is not evidence of anything."""
    from src.drift import build_baseline, detect_drift, make_shifted_sample

    baseline = build_baseline(save=False)
    shifted = make_shifted_sample(context.X_test)
    report = detect_drift(shifted, baseline)
    assert report["overall_status"] == "alert"
    assert report["alert_count"] > 0
    flagged = {f["feature"] for f in report["features"] if f["status"] != "stable"}
    assert {"Contract", "tenure"} <= flagged


def test_drift_baseline_proportions_sum_to_one() -> None:
    from src.drift import build_baseline

    baseline = build_baseline(save=False)
    for column, profile in baseline["numeric"].items():
        assert sum(profile["proportions"]) == pytest.approx(1.0, abs=1e-6), column
    for column, profile in baseline["categorical"].items():
        assert sum(profile["proportions"]) == pytest.approx(1.0, abs=1e-6), column


def test_drift_baseline_is_built_from_the_training_split_only() -> None:
    """Profiling the whole dataset would leak the holdout into the reference."""
    from src.drift import build_baseline

    baseline = build_baseline(save=False)
    assert baseline["source"] == "training split"
    assert baseline["rows"] == int(config.EXPECTED_ROWS * (1 - config.TEST_SIZE))


# --------------------------------------------------------------------------
# Feature engineering (H1-6)
# --------------------------------------------------------------------------


def test_engineered_features_are_added_without_mutating_input() -> None:
    """scikit-learn may pass a slice during CV; mutating it would corrupt folds."""
    from src.features import engineer_features
    from src.train import load_model_frame

    frame = load_model_frame().head(100)
    before = list(frame.columns)
    out = engineer_features(frame)
    assert list(frame.columns) == before, "engineer_features mutated its input"
    assert len(out.columns) > len(before)


def test_engineered_features_are_row_wise_only() -> None:
    """The leakage argument depends on this: no feature may depend on other rows.

    Scoring one row alone must give the same values as scoring it inside a batch.
    If it does not, the feature aggregates across rows and the split guarantee
    would be broken.
    """
    from src.features import ENGINEERED_NUMERIC, engineer_features
    from src.train import load_model_frame

    frame = load_model_frame().head(200)
    batch = engineer_features(frame)
    for position in (0, 57, 199):
        alone = engineer_features(frame.iloc[[position]])
        for column in ENGINEERED_NUMERIC:
            assert float(alone[column].iloc[0]) == pytest.approx(
                float(batch[column].iloc[position]), abs=1e-12
            ), f"{column} depends on other rows"


def test_engineered_features_are_never_infinite() -> None:
    """Infinities would survive imputation and poison the scaler. NaN would not."""
    from src.features import ENGINEERED_NUMERIC, engineer_features
    from src.train import load_model_frame

    out = engineer_features(load_model_frame())
    values = out[ENGINEERED_NUMERIC].to_numpy(dtype=float)
    assert not np.isinf(values).any()


def test_missing_average_spend_occurs_only_where_total_charges_is_missing() -> None:
    """AvgMonthlySpend is NaN for the 11 blank-TotalCharges rows, deliberately.

    Filling it inside the feature function would bake in a domain assumption.
    Leaving it lets the pipeline's median imputer handle it exactly as it
    already handles TotalCharges, keeping every fitted statistic inside the
    pipeline where the leakage guarantee lives.
    """
    from src.features import engineer_features
    from src.train import load_model_frame

    frame = load_model_frame()
    out = engineer_features(frame)
    missing_spend = out["AvgMonthlySpend"].isna()
    missing_total = frame["TotalCharges"].isna()
    assert missing_spend.equals(missing_total)
    assert int(missing_spend.sum()) == 11


def test_zero_tenure_customers_do_not_divide_by_zero() -> None:
    """The 11 documented zero-tenure rows are exactly where this would break."""
    from src.features import engineer_features
    from src.train import load_model_frame

    frame = load_model_frame()
    zero_tenure = frame[frame["tenure"] == 0]
    assert len(zero_tenure) == 11, "Expected the documented zero-tenure customers"
    out = engineer_features(zero_tenure)
    # No infinities, and the neutral fallback applied to the trend.
    assert not np.isinf(out["AvgMonthlySpend"].to_numpy(dtype=float)).any()
    assert (out["ChargesTrend"] == 1.0).all()


def test_engineered_pipeline_scores_zero_tenure_customers_end_to_end() -> None:
    """The NaN must survive the imputer and still produce a valid probability."""
    from src.tuning import build_pipeline
    from src.train import load_model_frame, split_features_target

    frame = load_model_frame()
    X, y = split_features_target(frame)
    pipeline = build_pipeline("Logistic Regression", engineered=True)
    pipeline.fit(X, y)

    zero_tenure = X[frame["tenure"] == 0]
    proba = pipeline.predict_proba(zero_tenure)[:, 1]
    assert len(proba) == 11
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_service_counts_are_within_bounds() -> None:
    from src.features import PROTECTIVE_ADDONS, SERVICE_COLUMNS, engineer_features
    from src.train import load_model_frame

    out = engineer_features(load_model_frame())
    assert out["NumServices"].between(0, len(SERVICE_COLUMNS)).all()
    assert out["NumProtectiveAddons"].between(0, len(PROTECTIVE_ADDONS)).all()
    # The protective count is a subset of the service count by construction.
    assert (out["NumProtectiveAddons"] <= out["NumServices"]).all()


def test_tenure_buckets_cover_every_customer() -> None:
    from src.features import TENURE_LABELS, engineer_features
    from src.train import load_model_frame

    out = engineer_features(load_model_frame())
    assert set(out["TenureBucket"].unique()) <= set(TENURE_LABELS)
    assert "nan" not in set(out["TenureBucket"].unique())


def test_tuning_experiment_records_a_decision_against_a_fixed_bar() -> None:
    """The adoption rule must be applied, not narrated after the fact."""
    path = config.TABLES_DIR / "tuning_experiment.json"
    if not path.is_file():
        pytest.skip("Tuning experiment not run — run `make tune`.")
    results = json.loads(path.read_text(encoding="utf-8"))
    decision = results["decision"]

    assert "adopt" in decision and isinstance(decision["adopt"], bool)
    assert decision["adopt"] == (decision["gain"] >= results["adoption_threshold"]), (
        "The recorded decision does not follow the stated adoption rule"
    )
    # All four arms must be present for both models, or the comparison is partial.
    for arms in results["arms"].values():
        assert set(arms) == {
            "A_raw_default", "B_raw_tuned", "C_engineered_default", "D_engineered_tuned"
        }
