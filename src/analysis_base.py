"""Shared evaluation context for the post-training analysis modules.

Fairness, calibration, threshold and drift analysis all need the same three
things: the exact train/test split used during training, the deployed pipeline,
and its metadata. Rebuilding that in four places would invite the splits to
drift apart silently, so it is built once here.

The split is reproduced from the immutable raw file using the same seed and
stratification as ``src/train.py``. It is therefore identical to the split the
artifact was fitted on, which is what makes these analyses valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.train import load_model_frame, split_features_target


@dataclass(frozen=True)
class EvaluationContext:
    """Everything the analysis modules need, built once and shared."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: Pipeline
    metadata: dict[str, Any]

    @property
    def test_probabilities(self) -> np.ndarray:
        """Predicted churn probability for each held-out customer."""
        return self.pipeline.predict_proba(self.X_test)[:, 1]

    @property
    def train_probabilities(self) -> np.ndarray:
        """In-sample probabilities. Optimistic — use only where that is stated."""
        return self.pipeline.predict_proba(self.X_train)[:, 1]

    def predictions_at(self, threshold: float) -> np.ndarray:
        return (self.test_probabilities >= threshold).astype(int)


def load_evaluation_context() -> EvaluationContext:
    """Rebuild the training split and load the deployed artifact.

    Raises:
        FileNotFoundError: when the model artifact has not been produced yet.
    """
    if not config.MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {config.MODEL_PATH}. Run `make train` first."
        )

    frame = load_model_frame()
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    pipeline = joblib.load(config.MODEL_PATH)
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))

    return EvaluationContext(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        pipeline=pipeline,
        metadata=metadata,
    )


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Feature names after the ColumnTransformer, in the order the model sees them."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


#: Shared bootstrap settings so every module reports a CI at the same
#: precision from the same number of resamples, rather than each analysis
#: picking its own and making the reports incomparable.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_ALPHA = 0.05


def bootstrap_percentile_ci(
    n: int,
    statistic_fn,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for an arbitrary row-level statistic.

    Args:
        n: number of rows in the population being resampled.
        statistic_fn: called with one integer index array (with replacement,
            length ``n``) per resample; must return a single float.
        n_resamples: bootstrap resamples. 1,000 is enough for a stable 95% CI
            at this sample size without being slow to compute.
        alpha: significance level; 0.05 gives a 95% interval.
        seed: makes the interval reproducible run to run, same convention as
            every other seeded procedure in this project.

    Returns:
        A dict with ``low``, ``high`` (the percentile bounds) and ``estimate``
        (the statistic on the unresampled data, for comparison).
    """
    rng = np.random.default_rng(seed)
    point_estimate = statistic_fn(np.arange(n))
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        samples[i] = statistic_fn(indices)

    # nanpercentile rather than percentile: a statistic_fn may legitimately
    # return NaN for a degenerate resample (e.g. a subgroup entirely absent by
    # chance). Those resamples are excluded rather than poisoning the interval.
    lower = float(np.nanpercentile(samples, 100 * (alpha / 2)))
    upper = float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))
    return {"estimate": float(point_estimate), "low": lower, "high": upper}
