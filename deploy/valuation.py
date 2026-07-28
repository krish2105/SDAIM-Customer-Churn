"""Revenue-at-risk valuation — turns a probability into a dollar exposure.

**Revenue at risk** = churn probability x `MonthlyCharges` x expected remaining
tenure in months. Every factor is either a direct model output or a customer's
own recorded field, except expected remaining tenure, which comes from the
persisted survival lookup (``artifacts/survival_reference.json``, built by
``src/survival.py`` from observed Kaplan-Meier curves by contract type).

This module is intentionally self-contained arithmetic with **no dependency on
`lifelines`** — the survival curves are fitted once, offline, and condensed
into a plain JSON lookup so the runtime image never needs the fitting library.
It duplicates the small residual-life formula in ``src/survival.py`` rather
than importing it, because the deployed application must never import the
training package (see ``tests/test_deployment_files.py``).

**What this is not.** It is not a prediction that this specific dollar amount
will be lost, and it is not a validated ROI figure — no intervention has been
measured to change any outcome. It is a transparent, auditable multiplication
of three already-disclosed numbers, offered as a triage aid: two customers
with the same churn probability do not carry the same commercial stake.
"""

from __future__ import annotations

#: Matches src/survival.py — a customer already at (or past) the observed
#: horizon is floored at one month of remaining value rather than zero.
FLOOR_REMAINING_MONTHS = 1.0
EPSILON = 1e-6


def expected_remaining_tenure(
    current_tenure: float,
    contract: str,
    reference: dict,
) -> float | None:
    """Discrete restricted-mean residual life for one customer.

    Returns ``None`` if *contract* has no entry in *reference* (an unrecognised
    category), so the caller can omit the valuation rather than guess.
    """
    segments = reference.get("segments", {})
    segment = segments.get(contract)
    if segment is None:
        return None

    survival = segment["survival"]
    tau = len(survival) - 1
    t0 = int(round(max(0.0, min(current_tenure, tau))))
    if t0 >= tau:
        return reference.get("floor_remaining_months", FLOOR_REMAINING_MONTHS)

    s_t0 = max(survival[t0], EPSILON)
    remaining = sum(survival[t0 + 1 : tau + 1]) / s_t0
    return max(remaining, reference.get("floor_remaining_months", FLOOR_REMAINING_MONTHS))


def revenue_at_risk(
    probability: float,
    monthly_charges: float,
    tenure: float,
    contract: str,
    reference: dict,
) -> dict | None:
    """Estimated monthly-equivalent revenue exposed by one customer's risk.

    Returns ``None`` when no survival reference is available for *contract*,
    so callers can render "not available" instead of a silently wrong number.
    """
    remaining = expected_remaining_tenure(tenure, contract, reference)
    if remaining is None:
        return None

    return {
        "expected_remaining_months": round(remaining, 1),
        "revenue_at_risk": round(probability * monthly_charges * remaining, 2),
        "monthly_charges": monthly_charges,
        "probability": probability,
    }
