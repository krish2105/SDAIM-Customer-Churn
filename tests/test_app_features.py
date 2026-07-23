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
