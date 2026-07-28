"""Batch-scoring alert webhook, with the same kill-switch discipline as ``rationale.py``.

Turns a batch-scoring result into an operational signal: when a scored upload
contains High-risk accounts, post a **summary** notification (counts and
aggregate revenue at risk — never a per-customer row) to a Slack-compatible
incoming webhook.

Guardrails, all enforced here rather than assumed:

1. **Kill switch** — disabled unless both ``ENABLE_BATCH_ALERTS`` and
   ``ALERT_WEBHOOK_URL`` are set. The application is fully functional with the
   layer off, and it ships off by default.
2. **Aggregate only** — the payload carries counts and totals, never a
   customer identifier or row. The batch page already promises uploaded data
   is scored in memory and never persisted; forwarding individual customers to
   a third-party webhook would break that promise even with the layer "on".
3. **Never breaks the interface** — every network or configuration failure is
   caught and logged; scoring and the on-screen queue are unaffected either
   way.
4. **Fires once per result, not once per rerun** — Streamlit reruns the whole
   script on every widget interaction, so a request is deduplicated against
   the specific upload and threshold that produced it.
5. **No secret in the repository** — the webhook URL is a Space secret, read
   from the environment, exactly like ``rationale.py``'s ``HF_TOKEN``.

Slack's incoming-webhook payload shape (``{"text": ...}``) is used because it
is the simplest widely-supported format; a Microsoft Teams workflow webhook
expects an Adaptive Card instead and would need its own payload builder.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import streamlit as st

logger = logging.getLogger("churn_app.alerts")

ENABLE_ENV_VAR = "ENABLE_BATCH_ALERTS"
WEBHOOK_ENV_VAR = "ALERT_WEBHOOK_URL"
REQUEST_TIMEOUT_SECONDS = 5

#: Only fire when at least this many High-risk accounts are in the upload.
#: A queue with zero High-risk rows is not an operational event.
MIN_HIGH_RISK_TO_ALERT = 1


def is_enabled() -> bool:
    """Whether the layer is switched on and a destination is configured."""
    flag = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(os.environ.get(WEBHOOK_ENV_VAR))


def build_message(summary: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Slack-compatible payload. Aggregate counts only — never a customer row."""
    lines = [
        "*Customer Churn Intelligence — retention work queue scored*",
        f"Customers scored: *{summary['total']:,}*  ·  "
        f"Flagged for review: *{summary['flagged']:,}* at threshold {threshold:.2f}",
        f"Risk bands — High: *{summary['high']:,}*  ·  Medium: {summary['medium']:,}  ·  "
        f"Low: {summary['low']:,}",
    ]
    if "flagged_revenue_at_risk" in summary:
        lines.append(
            f"Revenue at risk (flagged accounts): *${summary['flagged_revenue_at_risk']:,.0f}*"
        )
    lines.append(
        "_Decision support only — every account requires human review before contact._"
    )
    return {"text": "\n".join(lines)}


def send_alert(payload: dict[str, Any]) -> bool:
    """POST the payload to the configured webhook. Never raises.

    Returns ``False`` on any configuration or network failure, exactly as if
    nothing had been sent — the caller should not surface this to the user.
    """
    webhook_url = os.environ.get(WEBHOOK_ENV_VAR, "")
    if not webhook_url:
        return False

    try:
        import requests  # noqa: PLC0415

        response = requests.post(webhook_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        logger.exception("Batch alert webhook call failed")
        return False
    return True


def _dedupe_key(uploaded_file: Any, summary: dict[str, Any], threshold: float) -> str:
    """One alert per distinct (file, threshold, result), not per Streamlit rerun."""
    name = getattr(uploaded_file, "name", "")
    size = getattr(uploaded_file, "size", 0)
    fingerprint = f"{name}:{size}:{threshold:.2f}:{summary['high']}:{summary['flagged']}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def maybe_send_batch_alert(uploaded_file: Any, summary: dict[str, Any], threshold: float) -> None:
    """Send the alert if enabled, warranted, and not already sent for this result."""
    if not is_enabled() or summary.get("high", 0) < MIN_HIGH_RISK_TO_ALERT:
        return

    key = _dedupe_key(uploaded_file, summary, threshold)
    sent = st.session_state.setdefault("_batch_alerts_sent", set())
    if key in sent:
        return

    if send_alert(build_message(summary, threshold)):
        sent.add(key)
        logger.info("Batch alert sent: high=%d flagged=%d", summary["high"], summary["flagged"])
