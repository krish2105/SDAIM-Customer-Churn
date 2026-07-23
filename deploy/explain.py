"""Per-prediction contribution analysis (H1-4).

Closes gap G4. A retention specialist cannot act on a bare number; they need to
know which attributes moved it.

**Deliberately not SHAP.** For a linear model the contribution of each encoded
feature to the log-odds is exactly ``coefficient x transformed_value``. That is
not an approximation of the model — it *is* the model, and it reconstructs the
score exactly. A SHAP value for the same model would be an estimate of a quantity
this arithmetic gives in closed form, at the cost of a heavy dependency and an
explanation the team could not defend line by line.

For a non-linear estimator this decomposition does not exist; the module reports
that honestly rather than substituting a different method silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

#: Wording applied to every contribution surface. The model shows association
#: within a fictional training sample; it cannot support a causal claim about an
#: individual, and the interface must not imply otherwise.
CAUSAL_DISCLAIMER = (
    "These are contributions to this model's score, not reasons this customer will "
    "leave. They describe associations learned from the training sample. Changing an "
    "attribute would not necessarily change the customer's behaviour."
)


@dataclass(frozen=True)
class Contribution:
    """One encoded feature's push on the log-odds."""

    feature: str
    display_name: str
    value: str
    contribution: float

    @property
    def direction(self) -> str:
        return "increases" if self.contribution > 0 else "decreases"


@dataclass(frozen=True)
class Explanation:
    """Additive decomposition of a single prediction."""

    supported: bool
    reason: str
    intercept: float
    total_log_odds: float
    probability: float
    contributions: list[Contribution]

    def top(self, n: int = 5) -> tuple[list[Contribution], list[Contribution]]:
        """``(largest increases, largest decreases)``, each already sorted."""
        ordered = sorted(self.contributions, key=lambda c: c.contribution, reverse=True)
        increases = [c for c in ordered if c.contribution > 0][:n]
        decreases = sorted(
            [c for c in ordered if c.contribution < 0], key=lambda c: c.contribution
        )[:n]
        return increases, decreases

    def reconstructs(self, tolerance: float = 1e-8) -> bool:
        """Whether the parts genuinely sum to the whole.

        If this is ever false the decomposition is wrong and must not be shown.
        """
        total = self.intercept + sum(c.contribution for c in self.contributions)
        return bool(abs(total - self.total_log_odds) < tolerance)


def _prettify(encoded_name: str, schema: dict[str, Any]) -> tuple[str, str]:
    """Turn an encoded feature name into ``(display name, value)``.

    The one-hot encoder emits names like ``Contract_Two year``; the underlying
    column is matched by longest prefix so values containing underscores are not
    split in the wrong place.
    """
    labels = {f["name"]: f["label"] for f in schema.get("features", [])}
    candidates = [name for name in labels if encoded_name.startswith(f"{name}_")]
    if candidates:
        column = max(candidates, key=len)
        return labels[column], encoded_name[len(column) + 1:]
    if encoded_name in labels:
        return labels[encoded_name], "numeric"
    return encoded_name, ""


def explain_prediction(pipeline, row: pd.DataFrame, schema: dict[str, Any]) -> Explanation:
    """Decompose one prediction into per-feature log-odds contributions.

    Args:
        pipeline: The fitted pipeline (``preprocessor`` + ``classifier``).
        row: Exactly one row in the training column order.
        schema: The persisted feature schema, for display names.
    """
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    coefficients = getattr(classifier, "coef_", None)
    intercept = getattr(classifier, "intercept_", None)

    if coefficients is None or intercept is None:
        # Tree ensembles and dummies have no additive log-odds decomposition.
        # Report that rather than silently substituting a different method.
        probability = float(pipeline.predict_proba(row)[0, 1])
        return Explanation(
            supported=False,
            reason=(
                f"{type(classifier).__name__} has no additive log-odds decomposition. "
                "Permutation importance would be the appropriate method for this model."
            ),
            intercept=0.0,
            total_log_odds=0.0,
            probability=probability,
            contributions=[],
        )

    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    values = np.asarray(transformed).ravel()

    weights = np.asarray(coefficients).ravel()
    names = list(preprocessor.get_feature_names_out())
    products = weights * values

    intercept_value = float(np.asarray(intercept).ravel()[0])
    total = float(intercept_value + products.sum())
    probability = float(1.0 / (1.0 + np.exp(-total)))

    contributions = []
    for name, product, raw_value in zip(names, products, values):
        # A zero contribution carries no information: either the coefficient is
        # zero or the one-hot column is inactive for this customer.
        if abs(product) < 1e-12:
            continue
        display_name, level = _prettify(name, schema)
        contributions.append(
            Contribution(
                feature=name,
                display_name=display_name,
                value=level if level and level != "numeric" else f"{raw_value:.2f}",
                contribution=float(product),
            )
        )

    return Explanation(
        supported=True,
        reason="",
        intercept=intercept_value,
        total_log_odds=total,
        probability=probability,
        contributions=contributions,
    )


def contribution_summary(explanation: Explanation, n: int = 5) -> pd.DataFrame:
    """Tabular view of the strongest contributions in each direction."""
    increases, decreases = explanation.top(n)
    rows = [
        {
            "Attribute": c.display_name,
            "Value": c.value,
            "Effect on score": "Increases risk" if c.contribution > 0 else "Decreases risk",
            "Log-odds contribution": round(c.contribution, 4),
        }
        for c in [*increases, *decreases]
    ]
    return pd.DataFrame(rows)
