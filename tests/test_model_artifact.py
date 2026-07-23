"""Tests for the exported model artifact, metadata and feature schema."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from src import config


def test_model_artifact_exists() -> None:
    assert config.MODEL_PATH.is_file(), "Run `make train` to produce the model artifact."


def test_metadata_and_schema_exist() -> None:
    assert config.METADATA_PATH.is_file()
    assert config.FEATURE_SCHEMA_PATH.is_file()
    assert config.MODEL_CARD_PATH.is_file()


def test_pipeline_loads_and_is_a_complete_pipeline(pipeline: Pipeline) -> None:
    assert isinstance(pipeline, Pipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names, "Preprocessing must be inside the saved artifact"
    assert "classifier" in step_names


def test_pipeline_loads_in_a_fresh_process(project_root: Path) -> None:
    """A separate interpreter must be able to load and score the artifact."""
    script = (
        "import json, sys, joblib, pandas as pd;"
        "p = joblib.load(r'{model}');"
        "s = json.load(open(r'{schema}'));"
        "row = {{f['name']: f['default'] for f in s['features']}};"
        "df = pd.DataFrame([row])[s['feature_order']];"
        "prob = float(p.predict_proba(df)[0, 1]);"
        "assert 0.0 <= prob <= 1.0;"
        "print('FRESH_PROCESS_OK')"
    ).format(model=config.MODEL_PATH, schema=config.FEATURE_SCHEMA_PATH)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "FRESH_PROCESS_OK" in completed.stdout


def test_metadata_contains_real_values(model_metadata: dict[str, Any]) -> None:
    for key in (
        "project_name",
        "model_name",
        "model_version",
        "target",
        "positive_class",
        "decision_threshold",
        "random_state",
        "training_timestamp_utc",
        "dataset_git_blob_sha",
        "dataset_rows",
        "dataset_columns",
        "metrics",
    ):
        assert key in model_metadata, f"metadata is missing {key}"

    assert model_metadata["dataset_rows"] == config.EXPECTED_ROWS
    assert model_metadata["dataset_columns"] == config.EXPECTED_COLUMN_COUNT
    assert model_metadata["dataset_git_blob_sha"] == config.EXPECTED_GIT_BLOB_SHA
    assert model_metadata["random_state"] == config.RANDOM_STATE
    assert model_metadata["target"] == config.TARGET_COLUMN


def test_metadata_metrics_are_numeric_and_in_range(model_metadata: dict[str, Any]) -> None:
    """Guards against placeholder strings ever reaching the artifact."""
    for name, value in model_metadata["metrics"].items():
        assert isinstance(value, (int, float)), f"{name} must be a number, not {type(value)}"
        assert 0.0 <= float(value) <= 1.0, f"{name} outside [0, 1]"


def test_metadata_reports_a_real_selected_model(model_metadata: dict[str, Any]) -> None:
    assert model_metadata["model_name"] in {"Logistic Regression", "Random Forest"}

    # Guard against template tokens surviving into a shipped artifact. Matching
    # the specific placeholder strings rather than the bare word "ACTUAL", which
    # legitimately appears in the "not actual observed data" notice.
    placeholders = (
        "ACTUAL_SELECTED_MODEL",
        "ACTUAL_TIMESTAMP",
        "ACTUAL_VERIFIED_SHA",
        "ACTUAL_NUMBER",
        "TODO",
        "UNRESOLVED",
        "<<",
    )
    serialised = str(model_metadata)
    for token in placeholders:
        assert token not in serialised, f"metadata still contains the placeholder {token!r}"


def test_feature_schema_excludes_identifier_and_target(feature_schema: dict[str, Any]) -> None:
    names = [feature["name"] for feature in feature_schema["features"]]
    assert config.ID_COLUMN not in names, "customerID must not be a model feature"
    assert config.TARGET_COLUMN not in names, "Churn must not be a model feature"
    assert config.ID_COLUMN not in feature_schema["feature_order"]
    assert config.TARGET_COLUMN not in feature_schema["feature_order"]


def test_feature_schema_order_matches_training_configuration(
    feature_schema: dict[str, Any],
) -> None:
    assert feature_schema["feature_order"] == config.FEATURE_COLUMNS


def test_feature_schema_categories_cover_the_dataset(feature_schema: dict[str, Any]) -> None:
    by_name = {feature["name"]: feature for feature in feature_schema["features"]}
    for column, expected in config.EXPECTED_CATEGORIES.items():
        observed = set(by_name[column]["categories"])
        assert observed.issubset(set(expected))
        assert observed, f"{column} has no observed training categories"


def test_feature_schema_numeric_bounds_are_non_negative(feature_schema: dict[str, Any]) -> None:
    by_name = {feature["name"]: feature for feature in feature_schema["features"]}
    for column in config.NUMERIC_FEATURES:
        assert by_name[column]["minimum"] >= 0
        assert by_name[column]["maximum"] >= by_name[column]["minimum"]


def test_pipeline_feature_names_exclude_identifier_and_target(pipeline: Pipeline) -> None:
    seen = list(getattr(pipeline, "feature_names_in_", config.FEATURE_COLUMNS))
    assert config.ID_COLUMN not in seen
    assert config.TARGET_COLUMN not in seen
    assert seen == config.FEATURE_COLUMNS
