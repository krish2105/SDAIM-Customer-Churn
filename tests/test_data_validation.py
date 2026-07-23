"""Raw dataset contract tests."""

from __future__ import annotations

import pandas as pd

from src import config
from src.data_validation import validate_dataset


def test_raw_dataset_exists() -> None:
    assert config.RAW_DATASET_PATH.is_file(), (
        "data/raw/Telco-Customer-Churn.csv is missing. Restore the official IBM file; "
        "do not substitute another dataset."
    )


def test_schema_columns_exact_and_ordered(raw_frame: pd.DataFrame) -> None:
    assert list(raw_frame.columns) == config.EXPECTED_COLUMNS


def test_row_count(raw_frame: pd.DataFrame) -> None:
    assert len(raw_frame) == config.EXPECTED_ROWS


def test_column_count(raw_frame: pd.DataFrame) -> None:
    assert raw_frame.shape[1] == config.EXPECTED_COLUMN_COUNT


def test_customer_id_is_unique(raw_frame: pd.DataFrame) -> None:
    assert raw_frame[config.ID_COLUMN].is_unique


def test_target_domain(raw_frame: pd.DataFrame) -> None:
    assert set(raw_frame[config.TARGET_COLUMN].unique()) == {"Yes", "No"}


def test_senior_citizen_domain(raw_frame: pd.DataFrame) -> None:
    assert set(raw_frame["SeniorCitizen"].unique()).issubset({"0", "1"})


def test_no_duplicate_rows(raw_frame: pd.DataFrame) -> None:
    assert int(raw_frame.duplicated().sum()) == 0


def test_categorical_domains_match_data_dictionary(raw_frame: pd.DataFrame) -> None:
    for column, expected in config.EXPECTED_CATEGORIES.items():
        observed = set(raw_frame[column].unique())
        assert observed.issubset(set(expected)), f"{column} has unexpected categories"


def test_numeric_columns_have_no_negative_values(raw_frame: pd.DataFrame) -> None:
    for column in config.NUMERIC_FEATURES:
        coerced = pd.to_numeric(raw_frame[column].str.strip(), errors="coerce")
        assert not (coerced.dropna() < 0).any(), f"{column} contains negative values"


def test_total_charges_blanks_are_only_zero_tenure_customers(raw_frame: pd.DataFrame) -> None:
    """The documented blanks must stay explainable, not become arbitrary gaps."""
    blank = raw_frame["TotalCharges"].str.strip() == ""
    assert blank.sum() > 0, "Expected the documented blank TotalCharges values to be present"
    assert set(raw_frame.loc[blank, "tenure"].unique()) == {"0"}


def test_full_validator_passes() -> None:
    result = validate_dataset()
    assert result.passed, f"Validation failed: {[c['check'] for c in result.failures]}"


def test_git_blob_sha_matches_official_file() -> None:
    result = validate_dataset()
    observed = result.summary["git_blob_sha"]
    if observed is None:
        import pytest

        pytest.skip("Git is unavailable, so the blob SHA cannot be verified here.")
    assert observed == config.EXPECTED_GIT_BLOB_SHA
