"""Typed structures describing the model input contract.

These structures are the single source of truth shared by training (which
writes ``feature_schema.json``) and the Streamlit application (which reads it
to build controls). Keeping them here prevents the deployed form from drifting
away from the trained pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

from src import config

FeatureKind = Literal["categorical", "numeric"]


@dataclass(frozen=True)
class FeatureSpec:
    """Description of one model predictor."""

    name: str
    kind: FeatureKind
    dtype: str
    label: str
    help_text: str
    group: str
    categories: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    default: Any = None


#: Human-facing labels, help text and UI grouping for every predictor.
#: ``categories`` and numeric bounds are filled in at training time from the
#: observed training split only — never from the test split.
FEATURE_PRESENTATION: dict[str, dict[str, str]] = {
    "gender": {
        "label": "Gender",
        "help_text": "Gender as recorded in the customer sample.",
        "group": "Customer profile",
    },
    "SeniorCitizen": {
        "label": "Senior citizen",
        "help_text": "Whether the customer is classified as a senior citizen.",
        "group": "Customer profile",
    },
    "Partner": {
        "label": "Has partner",
        "help_text": "Whether the customer has a partner.",
        "group": "Customer profile",
    },
    "Dependents": {
        "label": "Has dependents",
        "help_text": "Whether the customer has dependents.",
        "group": "Customer profile",
    },
    "PhoneService": {
        "label": "Phone service",
        "help_text": "Whether the customer subscribes to phone service.",
        "group": "Services",
    },
    "MultipleLines": {
        "label": "Multiple lines",
        "help_text": "Forced to 'No phone service' when phone service is not held.",
        "group": "Services",
    },
    "InternetService": {
        "label": "Internet service",
        "help_text": "Internet product held by the customer.",
        "group": "Services",
    },
    "OnlineSecurity": {
        "label": "Online security",
        "help_text": "Add-on requiring an internet subscription.",
        "group": "Services",
    },
    "OnlineBackup": {
        "label": "Online backup",
        "help_text": "Add-on requiring an internet subscription.",
        "group": "Services",
    },
    "DeviceProtection": {
        "label": "Device protection",
        "help_text": "Add-on requiring an internet subscription.",
        "group": "Services",
    },
    "TechSupport": {
        "label": "Technical support",
        "help_text": "Add-on requiring an internet subscription.",
        "group": "Services",
    },
    "StreamingTV": {
        "label": "Streaming TV",
        "help_text": "Add-on requiring an internet subscription.",
        "group": "Services",
    },
    "StreamingMovies": {
        "label": "Streaming movies",
        "help_text": "Add-on requiring an internet subscription.",
        "group": "Services",
    },
    "Contract": {
        "label": "Contract term",
        "help_text": "Contract length currently held by the customer.",
        "group": "Contract and billing",
    },
    "PaperlessBilling": {
        "label": "Paperless billing",
        "help_text": "Whether the customer receives paperless billing.",
        "group": "Contract and billing",
    },
    "PaymentMethod": {
        "label": "Payment method",
        "help_text": "Method used to settle the account.",
        "group": "Contract and billing",
    },
    "tenure": {
        "label": "Tenure (months)",
        "help_text": "Number of months the customer has been with the company.",
        "group": "Charges and tenure",
    },
    "MonthlyCharges": {
        "label": "Monthly charges",
        "help_text": "Amount billed to the customer each month.",
        "group": "Charges and tenure",
    },
    "TotalCharges": {
        "label": "Total charges",
        "help_text": "Cumulative amount billed to the customer to date.",
        "group": "Charges and tenure",
    },
}

#: Order in which input groups are rendered in the application.
UI_GROUP_ORDER: list[str] = [
    "Customer profile",
    "Services",
    "Contract and billing",
    "Charges and tenure",
]


def build_feature_schema(
    categories: dict[str, list[str]],
    numeric_bounds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Assemble the serialisable feature schema written to ``deploy/artifacts``.

    Args:
        categories: Categories observed in the **training** split only.
        numeric_bounds: ``{"min", "max", "median"}`` per numeric predictor,
            computed on the **training** split only.
    """
    features: list[dict[str, Any]] = []

    for name in config.FEATURE_COLUMNS:
        presentation = FEATURE_PRESENTATION[name]
        if name in config.CATEGORICAL_FEATURES:
            observed = list(categories[name])
            spec = FeatureSpec(
                name=name,
                kind="categorical",
                dtype="string",
                label=presentation["label"],
                help_text=presentation["help_text"],
                group=presentation["group"],
                categories=observed,
                default=observed[0],
            )
        else:
            bounds = numeric_bounds[name]
            spec = FeatureSpec(
                name=name,
                kind="numeric",
                dtype="float64" if name != "tenure" else "int64",
                label=presentation["label"],
                help_text=presentation["help_text"],
                group=presentation["group"],
                minimum=float(bounds["min"]),
                maximum=float(bounds["max"]),
                default=float(bounds["median"]),
            )
        features.append(asdict(spec))

    return {
        "schema_version": "1.0.0",
        "project_name": config.PROJECT_NAME,
        "target": config.TARGET_COLUMN,
        "target_mapping": config.TARGET_MAPPING,
        "positive_class": config.POSITIVE_CLASS,
        "excluded_identifier": config.ID_COLUMN,
        "feature_order": list(config.FEATURE_COLUMNS),
        "numeric_features": list(config.NUMERIC_FEATURES),
        "categorical_features": list(config.CATEGORICAL_FEATURES),
        "ui_group_order": list(UI_GROUP_ORDER),
        "dependency_rules": {
            "phone": {
                "controlling_feature": "PhoneService",
                "controlling_value": "No",
                "dependent_features": ["MultipleLines"],
                "forced_value": config.NO_PHONE_SENTINEL,
            },
            "internet": {
                "controlling_feature": "InternetService",
                "controlling_value": "No",
                "dependent_features": list(config.INTERNET_DEPENDENT_FEATURES),
                "forced_value": config.NO_INTERNET_SENTINEL,
            },
        },
        "risk_tiers": {
            "low_max_exclusive": config.RISK_LOW_MAX,
            "medium_max_exclusive": config.RISK_MEDIUM_MAX,
            "note": (
                "Risk tiers are communication bands for human triage. They are "
                "not independently validated business thresholds."
            ),
        },
        "features": features,
    }


def load_feature_schema(path: Path | None = None) -> dict[str, Any]:
    """Read the persisted feature schema."""
    schema_path = path or config.FEATURE_SCHEMA_PATH
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
