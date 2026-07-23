"""Prediction behaviour of the saved pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config


def test_prediction_shape(pipeline: Pipeline, smoke_input: pd.DataFrame) -> None:
    predictions = pipeline.predict(smoke_input)
    assert predictions.shape == (len(smoke_input),)


def test_predicted_classes_are_binary(pipeline: Pipeline, smoke_input: pd.DataFrame) -> None:
    assert set(np.unique(pipeline.predict(smoke_input))).issubset({0, 1})


def test_probabilities_within_unit_interval(pipeline: Pipeline, smoke_input: pd.DataFrame) -> None:
    proba = pipeline.predict_proba(smoke_input)
    assert proba.shape == (len(smoke_input), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_single_row_prediction(pipeline: Pipeline, smoke_input: pd.DataFrame) -> None:
    """The application always scores exactly one row."""
    one = smoke_input.head(1)
    probability = float(pipeline.predict_proba(one)[0, 1])
    assert 0.0 <= probability <= 1.0


def test_unknown_categorical_value_does_not_crash(
    pipeline: Pipeline, smoke_input: pd.DataFrame
) -> None:
    """OneHotEncoder(handle_unknown='ignore') must absorb unseen categories."""
    mutated = smoke_input.head(1).copy()
    mutated.loc[:, "PaymentMethod"] = "Cryptocurrency wallet"
    mutated.loc[:, "Contract"] = "Five year"
    probability = float(pipeline.predict_proba(mutated)[0, 1])
    assert 0.0 <= probability <= 1.0


def test_missing_numeric_value_is_imputed_not_fatal(
    pipeline: Pipeline, smoke_input: pd.DataFrame
) -> None:
    mutated = smoke_input.head(1).copy()
    mutated.loc[:, "TotalCharges"] = np.nan
    probability = float(pipeline.predict_proba(mutated)[0, 1])
    assert 0.0 <= probability <= 1.0


def test_input_column_order_is_enforced(pipeline: Pipeline, smoke_input: pd.DataFrame) -> None:
    """Reordering columns must not silently change the result."""
    baseline = float(pipeline.predict_proba(smoke_input.head(1))[0, 1])
    shuffled = smoke_input.head(1)[list(reversed(config.FEATURE_COLUMNS))]
    reordered = float(pipeline.predict_proba(shuffled)[0, 1])
    assert baseline == pytest.approx(reordered), (
        "The pipeline must select columns by name, not by position"
    )


def test_prediction_is_deterministic(pipeline: Pipeline, smoke_input: pd.DataFrame) -> None:
    first = pipeline.predict_proba(smoke_input)[:, 1]
    second = pipeline.predict_proba(smoke_input)[:, 1]
    assert np.array_equal(first, second)


def test_identifier_column_is_rejected_or_ignored(
    pipeline: Pipeline, smoke_input: pd.DataFrame
) -> None:
    """customerID must never influence a score."""
    with_id = smoke_input.head(1).copy()
    with_id[config.ID_COLUMN] = "9999-ZZZZZ"
    probability = float(pipeline.predict_proba(with_id[config.FEATURE_COLUMNS])[0, 1])
    baseline = float(pipeline.predict_proba(smoke_input.head(1))[0, 1])
    assert probability == pytest.approx(baseline)


def test_threshold_and_class_agree(
    pipeline: Pipeline, smoke_input: pd.DataFrame, model_metadata: dict[str, Any]
) -> None:
    threshold = float(model_metadata["decision_threshold"])
    proba = pipeline.predict_proba(smoke_input)[:, 1]
    derived = (proba >= threshold).astype(int)
    assert np.array_equal(derived, pipeline.predict(smoke_input))


def test_risk_tier_boundaries() -> None:
    assert config.risk_tier(0.0) == "Low"
    assert config.risk_tier(config.RISK_LOW_MAX - 1e-9) == "Low"
    assert config.risk_tier(config.RISK_LOW_MAX) == "Medium"
    assert config.risk_tier(config.RISK_MEDIUM_MAX - 1e-9) == "Medium"
    assert config.risk_tier(config.RISK_MEDIUM_MAX) == "High"
    assert config.risk_tier(1.0) == "High"
