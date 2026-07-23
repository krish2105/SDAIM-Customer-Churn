"""Shared fixtures.

The smoke-test record is built from the validated raw dataset with the target
column removed, so the predictors are real but no label can leak into them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    from src.data_validation import load_raw_dataframe

    return load_raw_dataframe()


@pytest.fixture(scope="session")
def model_metadata() -> dict[str, Any]:
    if not config.METADATA_PATH.is_file():
        pytest.skip("Model metadata missing — run `make train` first.")
    return json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def feature_schema() -> dict[str, Any]:
    if not config.FEATURE_SCHEMA_PATH.is_file():
        pytest.skip("Feature schema missing — run `make train` first.")
    return json.loads(config.FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pipeline():
    if not config.MODEL_PATH.is_file():
        pytest.skip("Model artifact missing — run `make train` first.")
    import joblib

    return joblib.load(config.MODEL_PATH)


@pytest.fixture(scope="session")
def smoke_input(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Five real customer records, target and identifier removed."""
    frame = raw_frame.head(5).copy()
    frame = frame.drop(columns=[config.TARGET_COLUMN, config.ID_COLUMN])
    for column in config.NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column].str.strip(), errors="coerce")
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(str)
    return frame[config.FEATURE_COLUMNS]
