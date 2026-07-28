"""Tests for the deployed application's new capabilities.

Explainability, batch scoring and the guardrailed retention brief. These import
from ``deploy/`` the same way the running application does, so a break in the
import path is caught here rather than in the Space.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from src import config

# The application does not use the src package and is not installed, so the
# deploy directory is added exactly as the container does when running app.py.
DEPLOY_DIR = config.DEPLOY_DIR
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))


@pytest.fixture(scope="module")
def schema() -> dict:
    if not config.FEATURE_SCHEMA_PATH.is_file():
        pytest.skip("Feature schema missing — run `make train`.")
    return json.loads(config.FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline():
    if not config.MODEL_PATH.is_file():
        pytest.skip("Model artifact missing — run `make train`.")
    import joblib

    return joblib.load(config.MODEL_PATH)


@pytest.fixture(scope="module")
def default_row(schema: dict) -> pd.DataFrame:
    row = pd.DataFrame([{f["name"]: f["default"] for f in schema["features"]}])
    row = row[schema["feature_order"]]
    for column in schema["numeric_features"]:
        row[column] = pd.to_numeric(row[column])
    return row


@pytest.fixture(scope="module")
def raw_customers() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATASET_PATH, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def survival_reference() -> dict:
    if not config.SURVIVAL_REFERENCE_PATH.is_file():
        pytest.skip("Survival reference missing — run `make survival`.")
    return json.loads(config.SURVIVAL_REFERENCE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------


def test_explanation_reconstructs_the_model_exactly(pipeline, default_row, schema) -> None:
    """The decomposition IS the model, so it must reproduce it to float precision.

    If this fails the contribution chart is lying and must not be displayed.
    """
    import explain

    explanation = explain.explain_prediction(pipeline, default_row, schema)
    assert explanation.supported
    assert explanation.reconstructs()

    from_pipeline = float(pipeline.predict_proba(default_row)[0, 1])
    assert explanation.probability == pytest.approx(from_pipeline, abs=1e-12)


def test_explanation_reconstructs_across_many_customers(pipeline, schema, raw_customers) -> None:
    import explain

    sample = raw_customers.head(25)
    for _, record in sample.iterrows():
        row = pd.DataFrame([record])[schema["feature_order"]]
        for column in schema["numeric_features"]:
            row[column] = pd.to_numeric(row[column].astype(str).str.strip(), errors="coerce")
        explanation = explain.explain_prediction(pipeline, row, schema)
        assert explanation.reconstructs(), "Contribution decomposition drifted from the model"


def test_explanation_produces_both_directions(pipeline, default_row, schema) -> None:
    import explain

    explanation = explain.explain_prediction(pipeline, default_row, schema)
    increases, decreases = explanation.top(5)
    assert increases and decreases
    assert all(c.contribution > 0 for c in increases)
    assert all(c.contribution < 0 for c in decreases)


def test_explanation_display_names_are_human_readable(pipeline, default_row, schema) -> None:
    """Encoded names such as Contract_Two year must not reach the interface."""
    import explain

    explanation = explain.explain_prediction(pipeline, default_row, schema)
    labels = {f["label"] for f in schema["features"]}
    for contribution in explanation.contributions:
        assert contribution.display_name in labels, contribution.display_name


def test_explanation_carries_a_causal_disclaimer() -> None:
    import explain

    text = explain.CAUSAL_DISCLAIMER.lower()
    assert "not reasons" in text or "not causes" in text
    assert "association" in text


# --------------------------------------------------------------------------
# Batch scoring
# --------------------------------------------------------------------------


def test_batch_validation_accepts_the_project_dataset(schema, raw_customers) -> None:
    import batch

    result = batch.validate_batch(raw_customers.head(500), schema)
    assert result.ok, result.errors
    assert result.identifier_column == config.ID_COLUMN


def test_batch_validation_rejects_missing_columns(schema, raw_customers) -> None:
    import batch

    broken = raw_customers.head(20).drop(columns=["Contract"])
    result = batch.validate_batch(broken, schema)
    assert not result.ok
    assert any("Contract" in error for error in result.errors)


def test_batch_validation_rejects_negative_numbers(schema, raw_customers) -> None:
    import batch

    broken = raw_customers.head(20).copy()
    broken.loc[broken.index[0], "MonthlyCharges"] = "-10"
    result = batch.validate_batch(broken, schema)
    assert not result.ok
    assert any("negative" in error.lower() for error in result.errors)


def test_batch_validation_rejects_oversized_uploads(schema, raw_customers) -> None:
    import batch

    oversized = pd.concat([raw_customers] * 2, ignore_index=True)
    result = batch.validate_batch(oversized, schema)
    assert not result.ok
    assert any(str(batch.MAX_ROWS) in error.replace(",", "") for error in result.errors)


def test_batch_validation_warns_but_allows_unknown_categories(schema, raw_customers) -> None:
    """handle_unknown='ignore' absorbs these, so they warn rather than block."""
    import batch

    frame = raw_customers.head(20).copy()
    frame.loc[frame.index[0], "PaymentMethod"] = "Cryptocurrency wallet"
    result = batch.validate_batch(frame, schema)
    assert result.ok
    assert any("PaymentMethod" in warning for warning in result.warnings)


def test_batch_scoring_ranks_by_descending_risk(pipeline, schema, raw_customers) -> None:
    import batch

    queue = batch.score_batch(pipeline, raw_customers.head(300), schema, 0.5, config.ID_COLUMN)
    probabilities = queue["Churn probability"].tolist()
    assert probabilities == sorted(probabilities, reverse=True)
    assert queue["Priority"].tolist() == list(range(1, len(queue) + 1))


def test_batch_scoring_matches_single_row_scoring(pipeline, schema, raw_customers) -> None:
    """Batch and single-record paths must agree, or the queue is misleading."""
    import batch

    sample = raw_customers.head(10)
    queue = batch.score_batch(pipeline, sample, schema, 0.5, config.ID_COLUMN)

    prepared = batch.prepare_batch(sample, schema)
    for position in range(len(sample)):
        single = float(pipeline.predict_proba(prepared.iloc[[position]])[0, 1])
        customer = sample.iloc[position][config.ID_COLUMN]
        from_queue = float(
            queue.loc[queue["Customer"] == customer, "Churn probability"].iloc[0]
        )
        assert from_queue == pytest.approx(single, abs=1e-12)


def test_batch_scoring_is_fast_enough_for_the_interface(pipeline, schema, raw_customers) -> None:
    import batch

    sample = raw_customers.head(1000)
    started = time.perf_counter()
    batch.score_batch(pipeline, sample, schema, 0.5, config.ID_COLUMN)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, f"1,000 rows took {elapsed:.2f}s"


def test_batch_risk_bands_match_the_schema(pipeline, schema, raw_customers) -> None:
    import batch

    tiers = schema["risk_tiers"]
    bands = (float(tiers["low_max_exclusive"]), float(tiers["medium_max_exclusive"]))
    queue = batch.score_batch(
        pipeline, raw_customers.head(400), schema, 0.5, config.ID_COLUMN, bands
    )
    for _, row in queue.iterrows():
        probability = float(row["Churn probability"])
        expected = "Low" if probability < bands[0] else (
            "Medium" if probability < bands[1] else "High"
        )
        assert row["Risk band"] == expected


def test_batch_never_uses_the_identifier_as_a_predictor(schema) -> None:
    assert config.ID_COLUMN not in schema["feature_order"]


# --------------------------------------------------------------------------
# Retention brief and its guardrails
# --------------------------------------------------------------------------


def test_rationale_is_disabled_by_default(monkeypatch) -> None:
    """The layer must ship off, so the Space never depends on a provider."""
    import rationale

    monkeypatch.delenv(rationale.ENABLE_ENV_VAR, raising=False)
    assert rationale.is_enabled() is False


def test_rationale_stays_disabled_without_a_token(monkeypatch) -> None:
    import rationale

    monkeypatch.setenv(rationale.ENABLE_ENV_VAR, "true")
    monkeypatch.delenv(rationale.TOKEN_ENV_VAR, raising=False)
    assert rationale.is_enabled() is False


def test_deterministic_brief_is_returned_when_disabled(monkeypatch) -> None:
    import rationale

    monkeypatch.delenv(rationale.ENABLE_ENV_VAR, raising=False)
    brief = rationale.generate_brief(
        0.42, "Medium", 0.5, [("Contract term", "Month-to-month", 0.67)], 0.2654
    )
    assert brief.generated is False
    assert "22" not in brief.text or "42" in brief.text
    assert "human review" in brief.text.lower()


def test_deterministic_brief_passes_its_own_language_guardrail() -> None:
    """The fallback must satisfy the rules it enforces on generated text."""
    import rationale

    text = rationale.deterministic_brief(
        0.81, "High", 0.5,
        [("Contract term", "Month-to-month", 0.67), ("Internet service", "DSL", -0.62)],
        0.2654,
    )
    assert rationale.check_prohibited(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "This customer will churn next month.",
        "The account is going to cancel shortly.",
        "The score is high because they have a month-to-month contract.",
        "Churn is guaranteed for this profile.",
        "This is caused by the lack of technical support.",
        "They should be offered a discount to stay.",
        "Reduce their price to retain them.",
    ],
)
def test_prohibited_language_is_rejected(text: str) -> None:
    import rationale

    assert rationale.check_prohibited(text), f"Guardrail missed: {text}"


def test_acceptable_language_is_not_rejected() -> None:
    import rationale

    text = (
        "SUMMARY: The model gives this account an estimated churn probability of 81%, "
        "which is above the sample average. FACTORS: a month-to-month contract is "
        "associated with higher scores. QUESTIONS: has the account had recent service "
        "issues? CAVEATS: this is decision support requiring human review."
    )
    assert rationale.check_prohibited(text) == []
    assert rationale.validate_structure(text) == []


def test_structure_validation_rejects_incomplete_output() -> None:
    import rationale

    assert rationale.validate_structure("SUMMARY: too short.")


def test_fact_block_contains_only_computed_values() -> None:
    """The prompt must never carry raw customer attributes."""
    import rationale

    facts = rationale.build_facts(
        0.42, "Medium", 0.5, [("Contract term", "Month-to-month", 0.67)], 0.2654
    )
    assert "42.0%" in facts or "42%" in facts
    assert "Medium" in facts
    # No identifier, and nothing resembling a raw record dump.
    assert "customerID" not in facts
    assert "7590-VHVEG" not in facts


def test_brief_reports_its_provenance() -> None:
    import rationale

    brief = rationale.Brief(text="x", generated=False)
    assert "no ai generation" in brief.provenance.lower()
    assert rationale.Brief(text="x", generated=True).provenance.lower().startswith("ai-generated")


# --------------------------------------------------------------------------
# Revenue-at-risk valuation
# --------------------------------------------------------------------------


def test_expected_remaining_tenure_is_within_the_survival_horizon(survival_reference) -> None:
    import valuation

    for contract in ("Month-to-month", "One year", "Two year"):
        remaining = valuation.expected_remaining_tenure(0.0, contract, survival_reference)
        tau = survival_reference["tau_months"]
        assert 0.0 < remaining <= tau


def test_expected_remaining_tenure_decreases_as_current_tenure_increases(survival_reference) -> None:
    """A customer already further along has, on average, less commercial life left."""
    import valuation

    early = valuation.expected_remaining_tenure(1.0, "Month-to-month", survival_reference)
    late = valuation.expected_remaining_tenure(60.0, "Month-to-month", survival_reference)
    assert late < early


def test_expected_remaining_tenure_is_floored_at_the_horizon(survival_reference) -> None:
    import valuation

    tau = survival_reference["tau_months"]
    remaining = valuation.expected_remaining_tenure(tau, "Two year", survival_reference)
    assert remaining == survival_reference["floor_remaining_months"]


def test_expected_remaining_tenure_returns_none_for_an_unrecognised_contract(
    survival_reference,
) -> None:
    import valuation

    assert valuation.expected_remaining_tenure is not None  # sanity: module imported
    result = valuation.revenue_at_risk(0.5, 70.0, 12.0, "Lifetime", survival_reference)
    assert result is None


def test_revenue_at_risk_matches_manual_multiplication(survival_reference) -> None:
    import valuation

    probability, monthly_charges, tenure, contract = 0.42, 80.0, 10.0, "Month-to-month"
    result = valuation.revenue_at_risk(probability, monthly_charges, tenure, contract, survival_reference)
    remaining = valuation.expected_remaining_tenure(tenure, contract, survival_reference)
    assert result["revenue_at_risk"] == pytest.approx(probability * monthly_charges * remaining, abs=0.01)


def test_batch_scoring_adds_revenue_at_risk_when_reference_supplied(
    pipeline, schema, raw_customers, survival_reference
) -> None:
    import batch

    queue = batch.score_batch(
        pipeline, raw_customers.head(50), schema, 0.5, config.ID_COLUMN,
        survival_reference=survival_reference,
    )
    assert "Revenue at risk" in queue.columns
    assert queue["Revenue at risk"].notna().all()
    assert (queue["Revenue at risk"] >= 0).all()


def test_batch_scoring_omits_revenue_at_risk_without_a_reference(pipeline, schema, raw_customers) -> None:
    import batch

    queue = batch.score_batch(pipeline, raw_customers.head(20), schema, 0.5, config.ID_COLUMN)
    assert "Revenue at risk" not in queue.columns


# --------------------------------------------------------------------------
# Batch-scoring alert webhook
# --------------------------------------------------------------------------


def test_alerts_disabled_by_default(monkeypatch) -> None:
    import alerts

    monkeypatch.delenv(alerts.ENABLE_ENV_VAR, raising=False)
    assert alerts.is_enabled() is False


def test_alerts_stay_disabled_without_a_webhook_url(monkeypatch) -> None:
    import alerts

    monkeypatch.setenv(alerts.ENABLE_ENV_VAR, "true")
    monkeypatch.delenv(alerts.WEBHOOK_ENV_VAR, raising=False)
    assert alerts.is_enabled() is False


def test_alerts_enabled_only_with_both_flag_and_url(monkeypatch) -> None:
    import alerts

    monkeypatch.setenv(alerts.ENABLE_ENV_VAR, "true")
    monkeypatch.setenv(alerts.WEBHOOK_ENV_VAR, "https://hooks.slack.example/T000/B000/xxx")
    assert alerts.is_enabled() is True


def test_alert_message_contains_only_aggregate_counts_never_a_customer_row() -> None:
    """The batch page promises uploaded data is never persisted or forwarded."""
    import alerts

    summary = {
        "total": 100, "flagged": 12, "flagged_share": 0.12, "high": 5, "medium": 7, "low": 88,
        "total_revenue_at_risk": 1234.0, "flagged_revenue_at_risk": 900.0,
    }
    message = alerts.build_message(summary, 0.5)
    assert "customerID" not in message["text"]
    assert "7590-VHVEG" not in message["text"]
    assert "human review" in message["text"].lower()
    assert "900" in message["text"]


def test_send_alert_returns_false_without_a_webhook_configured(monkeypatch) -> None:
    import alerts

    monkeypatch.delenv(alerts.WEBHOOK_ENV_VAR, raising=False)
    assert alerts.send_alert({"text": "test"}) is False


def test_send_alert_catches_a_network_failure_and_returns_false(monkeypatch) -> None:
    """A webhook outage must never break batch scoring."""
    import alerts

    monkeypatch.setenv(alerts.WEBHOOK_ENV_VAR, "https://hooks.slack.example/T000/B000/xxx")

    def _raise(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr("requests.post", _raise)
    assert alerts.send_alert({"text": "test"}) is False


def test_maybe_send_batch_alert_is_a_no_op_when_disabled(monkeypatch) -> None:
    import alerts

    monkeypatch.delenv(alerts.ENABLE_ENV_VAR, raising=False)
    calls = []
    monkeypatch.setattr(alerts, "send_alert", lambda payload: calls.append(payload) or True)
    alerts.maybe_send_batch_alert(
        object(), {"total": 10, "flagged": 5, "high": 3, "medium": 1, "low": 6}, 0.5
    )
    assert calls == []


def test_maybe_send_batch_alert_skips_results_with_no_high_risk_accounts(monkeypatch) -> None:
    import alerts

    monkeypatch.setenv(alerts.ENABLE_ENV_VAR, "true")
    monkeypatch.setenv(alerts.WEBHOOK_ENV_VAR, "https://hooks.slack.example/T000/B000/xxx")
    calls = []
    monkeypatch.setattr(alerts, "send_alert", lambda payload: calls.append(payload) or True)
    alerts.maybe_send_batch_alert(
        object(), {"total": 10, "flagged": 0, "high": 0, "medium": 0, "low": 10}, 0.5
    )
    assert calls == []
