"""Customer Churn Intelligence — retention decision-support application.

Runtime behaviour
-----------------
The application loads a pre-trained pipeline from ``artifacts/`` and never
trains anything at startup. Every path is resolved relative to this file so the
container and a local run behave identically.

Outputs are decision support for a human retention specialist. Nothing here
takes, or should trigger, an action affecting a customer.
"""

from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

import batch as batch_module
import charts
import explain as explain_module
import rationale as rationale_module
import theme as theme_module

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
REFERENCE_RATES_PATH = ARTIFACTS_DIR / "reference_rates.json"
MODEL_CARD_PATH = ARTIFACTS_DIR / "model_card.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("churn_app")

#: Shown to the user when anything fails. Deliberately free of paths, stack
#: frames and internal identifiers — the detail goes to the server log only.
USER_SAFE_ERROR = (
    "The prediction service is temporarily unavailable. Please try again, and "
    "contact the analytics team if the problem persists."
)

#: Two documented demonstration profiles. Neither asserts what the model will
#: return: they are input presets, and the outcome is whatever the model scores.
DEMO_CASES: dict[str, dict[str, Any]] = {
    "Case A — established two-year contract": {
        "description": (
            "A long-tenure customer on a two-year contract paying by bank transfer, "
            "holding support and security add-ons. Used to demonstrate the form with a "
            "settled account profile."
        ),
        "values": {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "Yes",
            "Dependents": "Yes",
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "tenure": 68.0,
            "MonthlyCharges": 65.0,
            "TotalCharges": 4420.0,
        },
    },
    "Case B — new month-to-month fibre customer": {
        "description": (
            "A recently acquired customer on a month-to-month fibre contract paying by "
            "electronic check, with no support or security add-ons. Used to demonstrate "
            "the form with an early-life account profile."
        ),
        "values": {
            "gender": "Male",
            "SeniorCitizen": "0",
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "tenure": 2.0,
            "MonthlyCharges": 95.0,
            "TotalCharges": 190.0,
        },
    },
}

SENIOR_DISPLAY = {"0": "No", "1": "Yes"}


# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts() -> dict[str, Any]:
    """Load the pipeline and its companions once per process.

    Raises:
        FileNotFoundError: when a required artifact is absent. The caller turns
            this into a user-safe message; the detail stays in the log.
    """
    missing = [
        path.name
        for path in (MODEL_PATH, METADATA_PATH, FEATURE_SCHEMA_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))

    reference: dict[str, Any] = {}
    if REFERENCE_RATES_PATH.is_file():
        reference = json.loads(REFERENCE_RATES_PATH.read_text(encoding="utf-8"))

    model_card = ""
    if MODEL_CARD_PATH.is_file():
        model_card = MODEL_CARD_PATH.read_text(encoding="utf-8")

    logger.info(
        "Artifacts loaded: model=%s version=%s features=%d",
        metadata.get("model_name"),
        metadata.get("model_version"),
        len(schema.get("feature_order", [])),
    )
    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "schema": schema,
        "reference": reference,
        "model_card": model_card,
    }


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------


def active_theme() -> str:
    return "dark" if st.session_state.get("dark_mode", False) else "light"


def active_threshold(metadata: dict[str, Any]) -> float:
    """Current decision threshold: the user's choice, or the documented default.

    Exposed in the interface because `reports/threshold_analysis.md` shows the
    optimum depends entirely on a business cost ratio that was never supplied.
    Hiding the assumption behind a constant would misrepresent it as a finding.
    """
    default = float(metadata.get("decision_threshold", 0.5))
    return float(st.session_state.get("decision_threshold", default))


def risk_tier(probability: float, schema: dict[str, Any]) -> str:
    tiers = schema.get("risk_tiers", {})
    low_max = float(tiers.get("low_max_exclusive", 0.40))
    medium_max = float(tiers.get("medium_max_exclusive", 0.70))
    if probability < low_max:
        return "Low"
    if probability < medium_max:
        return "Medium"
    return "High"


def recommended_action(tier: str) -> str:
    """Cautious, non-committal review guidance. No pricing or contract action."""
    return {
        "Low": (
            "No immediate retention action is indicated by this score. The account may "
            "be left to routine monitoring and periodic reporting."
        ),
        "Medium": (
            "This score may support adding the account to a periodic review queue so a "
            "retention specialist can look at the wider relationship history."
        ),
        "High": (
            "This score may support prioritising the account for a structured human "
            "retention review. A specialist should confirm the picture against account "
            "history and customer contact records before anything is offered."
        ),
    }[tier]


def render_kpis(metadata: dict[str, Any]) -> None:
    """Measured facts about the deployed model. No projected business impact."""
    metrics = metadata.get("metrics", {})
    cards = [
        ("Deployed model", escape(str(metadata.get("model_name", "—"))), "Selected by cross-validation"),
        ("Test ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}", "Held-out set, measured once"),
        ("Test recall (churn)", f"{metrics.get('recall', float('nan')):.3f}", "Share of actual churners identified"),
        ("Decision threshold", f"{metadata.get('decision_threshold', 0.5):.2f}", "Documented default, not optimised"),
    ]
    html = ['<div class="cci-kpis">']
    for label, value, note in cards:
        html.append(
            f'<div class="cci-kpi"><div class="cci-kpi-label">{escape(label)}</div>'
            f'<div class="cci-kpi-value">{value}</div>'
            f'<div class="cci-kpi-note">{escape(note)}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def section_title(title: str, hint: str = "") -> None:
    """Header row inside a bordered section container.

    Sections use ``st.container(border=True)`` rather than a raw wrapper div:
    Streamlit closes unbalanced HTML at the end of each markdown block, so a
    hand-written opening ``<div>`` would never actually enclose the widgets.
    """
    hint_html = f'<span class="cci-section-hint">{escape(hint)}</span>' if hint else ""
    st.markdown(
        f'<div class="cci-section-title">{escape(title)}{hint_html}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------


def apply_demo_case(case_name: str, schema: dict[str, Any]) -> None:
    """Write a demonstration profile into widget state."""
    values = DEMO_CASES[case_name]["values"]
    for feature in schema["features"]:
        name = feature["name"]
        if name in values:
            st.session_state[f"input_{name}"] = values[name]


def initialise_state(schema: dict[str, Any]) -> None:
    for feature in schema["features"]:
        key = f"input_{feature['name']}"
        if key not in st.session_state:
            st.session_state[key] = feature["default"]


def categorical_control(feature: dict[str, Any], disabled_value: str | None = None) -> str:
    """Render one selectbox, honouring a forced dependency value.

    Widget values live in session state under ``input_<name>``, so no default is
    passed here — that would conflict with state written by the demo loader.
    """
    name = feature["name"]
    key = f"input_{name}"
    options: list[str] = list(feature["categories"])

    if disabled_value is not None:
        # Render the forced value only, without a state key, so the user's own
        # earlier choice is preserved for when the dependency is released.
        st.selectbox(
            feature["label"],
            options=[disabled_value],
            index=0,
            disabled=True,
            help=f"{feature['help_text']} Locked by the service selection above.",
        )
        return disabled_value

    format_func = (lambda value: SENIOR_DISPLAY[value]) if name == "SeniorCitizen" else str
    return st.selectbox(
        feature["label"],
        options=options,
        key=key,
        format_func=format_func,
        help=feature["help_text"],
    )


def numeric_control(feature: dict[str, Any]) -> float:
    """Render one number input. Negative values are impossible by construction."""
    name = feature["name"]
    key = f"input_{name}"
    step = 1.0 if name == "tenure" else 0.05
    # The upper bound is generous rather than clamped to the training maximum:
    # a genuinely larger value should still be enterable, and the plausibility
    # notes flag it instead of silently truncating it.
    maximum = float(feature["maximum"]) * 3 if feature.get("maximum") else 1e6
    return st.number_input(
        feature["label"],
        min_value=0.0,
        max_value=maximum,
        step=step,
        key=key,
        help=(
            f"{feature['help_text']} Observed in training between "
            f"{feature['minimum']:.2f} and {feature['maximum']:.2f}."
        ),
        format="%.0f" if name == "tenure" else "%.2f",
    )


def render_inputs(schema: dict[str, Any]) -> dict[str, Any]:
    """Render every predictor control and return the collected raw values."""
    features = {feature["name"]: feature for feature in schema["features"]}
    values: dict[str, Any] = {}

    # --- Customer profile -------------------------------------------------
    with st.container(border=True):
        section_title("Customer profile", "Demographic attributes as recorded")
        columns = st.columns(2)
        for index, name in enumerate(["gender", "SeniorCitizen", "Partner", "Dependents"]):
            with columns[index % 2]:
                values[name] = categorical_control(features[name])

    # --- Services ---------------------------------------------------------
    with st.container(border=True):
        section_title("Services", "Add-ons follow the subscription they depend on")
        columns = st.columns(2)
        with columns[0]:
            values["PhoneService"] = categorical_control(features["PhoneService"])
        with columns[1]:
            phone_locked = (
                schema["dependency_rules"]["phone"]["forced_value"]
                if values["PhoneService"] == "No"
                else None
            )
            values["MultipleLines"] = categorical_control(features["MultipleLines"], phone_locked)

        values["InternetService"] = categorical_control(features["InternetService"])
        internet_locked = (
            schema["dependency_rules"]["internet"]["forced_value"]
            if values["InternetService"] == "No"
            else None
        )
        addons = schema["dependency_rules"]["internet"]["dependent_features"]
        columns = st.columns(2)
        for index, name in enumerate(addons):
            with columns[index % 2]:
                values[name] = categorical_control(features[name], internet_locked)

        if internet_locked:
            st.caption(
                "Internet service is set to 'No', so every internet add-on is locked to "
                "'No internet service' — the only combination present in the training data."
            )
        if phone_locked:
            st.caption(
                "Phone service is set to 'No', so multiple lines is locked to 'No phone service'."
            )

    # --- Contract and billing --------------------------------------------
    with st.container(border=True):
        section_title("Contract and billing", "Commercial terms of the account")
        columns = st.columns(2)
        with columns[0]:
            values["Contract"] = categorical_control(features["Contract"])
            values["PaperlessBilling"] = categorical_control(features["PaperlessBilling"])
        with columns[1]:
            values["PaymentMethod"] = categorical_control(features["PaymentMethod"])

    # --- Charges and tenure ----------------------------------------------
    with st.container(border=True):
        section_title("Charges and tenure", "Numeric values must be zero or greater")
        columns = st.columns(3)
        for index, name in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
            with columns[index]:
                values[name] = numeric_control(features[name])

    return values


def validate_inputs(values: dict[str, Any], schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(blocking_errors, non_blocking_notes)``."""
    errors: list[str] = []
    notes: list[str] = []

    for name in schema["numeric_features"]:
        value = values.get(name)
        if value is None or float(value) < 0:
            errors.append(f"{name} must be zero or greater.")

    tenure = float(values.get("tenure", 0))
    monthly = float(values.get("MonthlyCharges", 0))
    total = float(values.get("TotalCharges", 0))

    if tenure == 0 and total > 0:
        notes.append(
            "Tenure is 0 but total charges are above 0. In the training sample, "
            "zero-tenure customers had no total charges recorded yet."
        )
    if tenure >= 1 and total < monthly * 0.5:
        notes.append(
            "Total charges look low relative to monthly charges and tenure. "
            "Check the figures before relying on the score."
        )
    return errors, notes


def build_input_frame(values: dict[str, Any], schema: dict[str, Any]) -> pd.DataFrame:
    """One row in exactly the training column order and dtypes."""
    row = {name: values[name] for name in schema["feature_order"]}
    frame = pd.DataFrame([row], columns=schema["feature_order"])
    for name in schema["categorical_features"]:
        frame[name] = frame[name].astype(str)
    for name in schema["numeric_features"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
    return frame


# --------------------------------------------------------------------------
# Result rendering
# --------------------------------------------------------------------------


def render_result(
    probability: float,
    prediction: int,
    values: dict[str, Any],
    metadata: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    tier = risk_tier(probability, schema)
    foreground, background = theme_module.risk_colours(active_theme(), tier)
    threshold = active_threshold(metadata)
    default_threshold = float(metadata.get("decision_threshold", 0.5))
    threshold_note = (
        f"{threshold:.2f}" if abs(threshold - default_threshold) < 1e-9
        else f"{threshold:.2f} (adjusted from {default_threshold:.2f})"
    )
    class_label = "Likely to churn" if prediction == 1 else "Not likely to churn"

    st.markdown(
        f"""
<div class="cci-result">
  <div class="cci-result-head">
    <span class="cci-result-tier" style="color:{foreground};background:{background};
      border:1px solid {foreground};">{tier} risk — for human review</span>
    <div class="cci-result-prob" style="color:{foreground};">{probability:.1%}</div>
    <p class="cci-result-caption">Estimated probability that this customer churns</p>
  </div>
  <div class="cci-result-body">
    <dl>
      <div class="cci-result-row"><dt>Predicted class</dt><dd>{escape(class_label)}</dd></div>
      <div class="cci-result-row"><dt>Decision threshold</dt><dd>{threshold_note}</dd></div>
      <div class="cci-result-row"><dt>Risk band</dt><dd>{tier}</dd></div>
      <div class="cci-result-row"><dt>Contract</dt><dd>{escape(str(values['Contract']))}</dd></div>
      <div class="cci-result-row"><dt>Tenure</dt><dd>{float(values['tenure']):.0f} months</dd></div>
      <div class="cci-result-row"><dt>Model version</dt>
        <dd>{escape(str(metadata.get('model_version', '—')))}</dd></div>
    </dl>
    <div class="cci-action"><strong>Suggested review action.</strong>
      {escape(recommended_action(tier))}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Risk bands are communication aids for triage, not independently validated business "
        "thresholds. **This probability is a model score, not a validated frequency:** "
        "calibration analysis shows the model is over-confident about churn because it was "
        "trained with balanced class weights to favour recall. Use the ranking with more "
        "confidence than the magnitude."
    )


def render_context_charts(
    probability: float,
    values: dict[str, Any],
    schema: dict[str, Any],
    reference: dict[str, Any],
    explanation: Any | None = None,
) -> None:
    tiers = schema.get("risk_tiers", {})
    tab_why, tab_position, tab_segments, tab_population = st.tabs(
        ["Why this score", "Risk position", "Segment context", "Scored population"]
    )

    with tab_why:
        if explanation is None:
            st.info("Contribution analysis is not available for this prediction.")
        elif not explanation.supported:
            st.info(
                "This model type has no additive contribution decomposition, so none is "
                "shown. Showing an approximation without saying so would be misleading."
            )
        else:
            figure = charts.contribution_chart(explanation, active_theme())
            if figure is None:
                st.info("No attribute moved this score materially.")
            else:
                st.pyplot(figure, width="stretch")
                st.caption(explain_module.CAUSAL_DISCLAIMER)
                with st.expander("How this is calculated", expanded=False):
                    st.markdown(
                        "For a logistic regression the contribution of each encoded feature to "
                        "the log-odds is exactly `coefficient x transformed value`. This is not "
                        "an approximation of the model — it **is** the model, and the "
                        "contributions plus the intercept reconstruct the score exactly. A test "
                        "asserts that reconstruction on every run.\n\n"
                        f"Intercept `{explanation.intercept:+.4f}` + contributions = "
                        f"log-odds `{explanation.total_log_odds:+.4f}` -> probability "
                        f"`{explanation.probability:.4f}`."
                    )

    with tab_position:
        st.pyplot(
            charts.risk_position_chart(
                probability,
                active_theme(),
                float(tiers.get("low_max_exclusive", 0.40)),
                float(tiers.get("medium_max_exclusive", 0.70)),
            ),
            width="stretch",
        )

    with tab_segments:
        if not reference:
            st.info("Segment reference statistics are not available in this build.")
        else:
            selections = {
                column: str(values[column])
                for column in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]
                if column in values
            }
            figure = charts.segment_reference_chart(selections, reference, active_theme())
            if figure is None:
                st.info("No matching segment statistics for the current selection.")
            else:
                st.pyplot(figure, width="stretch")
                st.caption(
                    "Descriptive churn rates observed in the training split for each selected "
                    "segment. These are sample averages, not statements about this customer, "
                    "and they do not isolate the effect of any single attribute."
                )

    with tab_population:
        figure = charts.score_distribution_chart(probability, reference, active_theme())
        if figure is None:
            st.info("Score distribution reference is not available in this build.")
        else:
            st.pyplot(figure, width="stretch")
            st.caption(
                "Distribution of scores the deployed model produced on the held-out test set. "
                "It shows where this customer sits relative to a scored population; it is not "
                "a ranking of live customers."
            )


def render_model_information(metadata: dict[str, Any], model_card: str) -> None:
    with st.expander("Model information and limitations", expanded=False):
        metrics = metadata.get("metrics", {})
        st.markdown(
            f"""
**Model** {metadata.get('model_name', '—')} &nbsp;·&nbsp;
**Version** {metadata.get('model_version', '—')} &nbsp;·&nbsp;
**Trained (UTC)** {metadata.get('training_timestamp_utc', '—')} &nbsp;·&nbsp;
**Seed** {metadata.get('random_state', '—')}

**Held-out test performance** — accuracy {metrics.get('accuracy', 0):.4f} ·
precision {metrics.get('precision', 0):.4f} · recall {metrics.get('recall', 0):.4f} ·
F1 {metrics.get('f1', 0):.4f} · ROC-AUC {metrics.get('roc_auc', 0):.4f}

**Selection rule.** {metadata.get('selection_rule', '—')}

**{metadata.get('selection_justification', '')}**
"""
        )
        st.markdown(
            """
#### Limitations

- Trained on IBM's **fictional** telecommunications sample. Performance on any live
  population is unknown and must be revalidated before operational use.
- A single cross-section with no time dimension, so drift and seasonality cannot be assessed.
- The positive class is a minority, which limits the precision attainable at high recall.
- The 0.5 threshold is a documented default, not a cost-optimised operating point.
- `gender` and `SeniorCitizen` are among the predictors. **A fairness audit has been
  performed** (`reports/fairness_report.md`), optimising for equal opportunity. No material
  disparity was found on `gender`. A material disparity was found on `SeniorCitizen`, driven
  largely by a genuine base-rate difference between the groups. Removing both attributes was
  measured to cost almost no accuracy and is recommended before operational use.
- The model shows association within the sample, never causation. It cannot say what would
  happen if this customer's contract or services were changed.
- Per-prediction contributions are shown and are exact for this linear model, but they
  describe association within the training sample, not causation.

#### Governance

This model must not autonomously change prices, terminate or modify contracts, deny
service, target customers unfairly, or make any financial or customer-treatment decision.
Every output requires human review before a customer is contacted.
"""
        )
    # Kept as a sibling rather than nested inside the expander above: Streamlit
    # does not render an expander inside an expander correctly.
    if model_card:
        with st.expander("Full model card", expanded=False):
            st.markdown(model_card)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Customer Churn Intelligence",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False

    try:
        artifacts = load_artifacts()
    except Exception:  # noqa: BLE001 - user must never see the technical detail
        logger.exception("Artifact loading failed")
        st.markdown(theme_module.build_css("light"), unsafe_allow_html=True)
        st.title("Customer Churn Intelligence")
        st.error(USER_SAFE_ERROR)
        st.stop()
        return

    metadata = artifacts["metadata"]
    schema = artifacts["schema"]
    reference = artifacts["reference"]
    initialise_state(schema)

    if "decision_threshold" not in st.session_state:
        st.session_state["decision_threshold"] = float(
            metadata.get("decision_threshold", 0.5)
        )

    st.markdown(theme_module.build_css(active_theme()), unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("### Churn Intelligence")
        st.caption("Retention decision support")
        st.toggle(
            "Dark mode",
            key="dark_mode",
            help="Switches the interface and charts between the light and dark palettes.",
        )
        st.divider()

        st.markdown("**Decision threshold**")
        default_threshold = float(metadata.get("decision_threshold", 0.5))
        st.slider(
            "Flag a customer at or above",
            min_value=0.05,
            max_value=0.95,
            step=0.01,
            key="decision_threshold",
            help=(
                "The threshold that converts a probability into a flag. The cost-optimal "
                "value depends on how many unnecessary reviews one missed churner is worth "
                "— a business input this project was never given. Move it to see the "
                "sensitivity."
            ),
        )
        if abs(st.session_state["decision_threshold"] - default_threshold) > 1e-9:
            st.caption(
                f"Adjusted from the documented default of {default_threshold:.2f}. "
                "Lower catches more churners and creates more unnecessary reviews."
            )
        else:
            st.caption(
                f"Documented default ({default_threshold:.2f}) — an assumption, not an optimum."
            )

        st.divider()

        st.markdown("**Demonstration cases**")
        case_name = st.selectbox(
            "Load a documented profile",
            options=list(DEMO_CASES),
            help="Fills the form with a documented input profile. The prediction is "
            "whatever the model returns for it — no outcome is pre-set.",
        )
        st.caption(DEMO_CASES[case_name]["description"])
        if st.button("Load this profile", width="stretch"):
            apply_demo_case(case_name, schema)
            st.rerun()

        st.divider()
        st.markdown(
            f'<span class="cci-version-caption">Application version '
            f'{escape(str(metadata.get("model_version", "—")))}</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Model: {metadata.get('model_name', '—')}  \n"
            f"Trained (UTC): {metadata.get('training_timestamp_utc', '—')}"
        )

    # ---------------- Header ----------------
    st.markdown(
        f"""
<div class="cci-header">
  <h1>Customer Churn Intelligence</h1>
  <p class="cci-purpose">Estimates whether a customer shows elevated churn risk, so retention
  specialists can prioritise accounts for a structured human review.</p>
  <div class="cci-badges">
    <span class="cci-badge">Model <strong>{escape(str(metadata.get('model_name', '—')))}</strong></span>
    <span class="cci-badge">Version <strong>{escape(str(metadata.get('model_version', '—')))}</strong></span>
    <span class="cci-badge">Dataset <strong>IBM Telco sample (fictional)</strong></span>
    <span class="cci-badge">Decision support <strong>Human review required</strong></span>
  </div>
  <p class="cci-deployment-note">Application version
  {escape(str(metadata.get('model_version', '—')))} — automated deployment verification</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="cci-notice">
  <strong>Data and use disclaimer.</strong> This application is trained on IBM's publicly
  published Telco Customer Churn sample, which represents a <strong>fictional</strong>
  telecommunications company. It is not actual observed commercial customer data. Outputs
  are decision support only: they may support prioritisation for human review and must not
  be used to change prices, modify or terminate contracts, deny service, or make any
  customer-treatment decision automatically.
</div>
""",
        unsafe_allow_html=True,
    )

    render_kpis(metadata)
    st.write("")

    # ---------------- Body ----------------
    single_tab, batch_tab = st.tabs(
        ["Single customer assessment", "Batch scoring — retention work queue"]
    )

    with batch_tab:
        render_batch_page(artifacts, metadata, schema)

    with single_tab:
        render_single_assessment(artifacts, metadata, schema, reference)


def render_single_assessment(
    artifacts: dict[str, Any],
    metadata: dict[str, Any],
    schema: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """One customer at a time: form, result, and contribution analysis."""
    input_column, result_column = st.columns([1.55, 1], gap="large")

    with input_column:
        values = render_inputs(schema)
        errors, notes = validate_inputs(values, schema)
        predict_clicked = st.button(
            "Generate churn assessment",
            type="primary",
            width="stretch",
            disabled=bool(errors),
        )
        for message in errors:
            st.error(message)
        for message in notes:
            st.warning(message)

    # Scoring happens outside the column contexts so the result can be kept in
    # session state. Without this, an unrelated rerun — switching the theme, for
    # instance — would silently discard the assessment on screen.
    failed = False
    if predict_clicked and not errors:
        try:
            frame = build_input_frame(values, schema)
            probability = float(artifacts["pipeline"].predict_proba(frame)[0, 1])
            prediction = int(probability >= active_threshold(metadata))
            explanation = explain_module.explain_prediction(
                artifacts["pipeline"], frame, schema
            )
            if explanation.supported and not explanation.reconstructs():
                # The decomposition must equal the model exactly. If it does not,
                # something is wrong and showing it would mislead.
                logger.error("Contribution decomposition failed to reconstruct the score")
                explanation = None
        except Exception:  # noqa: BLE001
            logger.exception("Prediction failed for one submitted record")
            st.session_state.pop("last_assessment", None)
            failed = True
        else:
            logger.info(
                "Scored one record: probability=%.6f prediction=%d", probability, prediction
            )
            st.session_state["last_assessment"] = {
                "probability": probability,
                "prediction": prediction,
                "values": dict(values),
                "explanation": explanation,
            }

    assessment = st.session_state.get("last_assessment")

    with result_column:
        if failed:
            st.error(USER_SAFE_ERROR)
        elif assessment:
            if assessment["values"] != values:
                st.info(
                    "The inputs have changed since this assessment was produced. "
                    "Select 'Generate churn assessment' again to refresh it."
                )
            render_result(
                assessment["probability"],
                int(assessment["probability"] >= active_threshold(metadata)),
                assessment["values"],
                metadata,
                schema,
            )
        else:
            st.markdown(
                """
<div class="cci-empty">
  <strong>No assessment yet</strong>
  Complete the customer details and select <em>Generate churn assessment</em>.
  The result shows a probability, a risk band and a suggested review action for a
  retention specialist to confirm.
</div>
""",
                unsafe_allow_html=True,
            )

    if assessment and not failed:
        st.write("")
        render_retention_brief(assessment, metadata, schema, reference)

    # Context charts sit full width beneath both columns: inside the narrow
    # result column the axis labels become unreadable.
    if assessment and not failed:
        st.write("")
        render_context_charts(
            assessment["probability"],
            assessment["values"],
            schema,
            reference,
            assessment.get("explanation"),
        )

    st.write("")
    render_model_information(metadata, artifacts["model_card"])

    st.markdown(
        f"""
<div class="cci-footnote">
  Customer Churn Intelligence and Retention Decision-Support Platform ·
  application version {escape(str(metadata.get('model_version', '—')))} ·
  model {escape(str(metadata.get('model_name', '—')))} ·
  dataset: IBM Telco Customer Churn sample (fictional company, Apache-2.0 repository).
  Predictions are decision support and require human review. No revenue, churn-reduction
  or return-on-investment effect has been measured or is claimed.
</div>
""",
        unsafe_allow_html=True,
    )


def render_retention_brief(
    assessment: dict[str, Any],
    metadata: dict[str, Any],
    schema: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """Written brief for the specialist, generated or templated."""
    probability = assessment["probability"]
    tier = risk_tier(probability, schema)
    threshold = active_threshold(metadata)
    contributions = rationale_module.contributions_from_explanation(
        assessment.get("explanation")
    )
    overall = reference.get("overall_training_churn_rate") if reference else None

    with st.expander("Retention review brief", expanded=False):
        try:
            brief = rationale_module.generate_brief(
                probability, tier, threshold, contributions, overall
            )
        except Exception:  # noqa: BLE001
            logger.exception("Retention brief generation failed")
            st.error(USER_SAFE_ERROR)
            return

        if brief.generated:
            st.caption(
                "⚠️ **AI-generated** from pre-computed model outputs. The model that wrote this "
                "text made no inference about the customer — it only rendered numbers this "
                "application had already calculated. Verify before use."
            )
        else:
            st.caption(
                f"Deterministic template — no AI generation. {brief.fallback_reason}"
            )

        st.markdown(brief.text)

        st.caption(
            "This brief supports a human review. It is not an instruction, it recommends no "
            "commercial action, and it must not be shown to a customer."
        )


# --------------------------------------------------------------------------
# Batch scoring — retention work queue
# --------------------------------------------------------------------------


def render_batch_page(
    artifacts: dict[str, Any],
    metadata: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Score a whole customer book and return a prioritised review queue."""
    st.markdown(
        """
<div class="cci-notice">
  <strong>Retention work queue.</strong> Upload a customer file to score every record at once
  and receive a queue ranked by estimated churn risk. The ranking is decision support for
  prioritising human review — it is not an instruction to contact anyone, and no customer
  should be acted on without a specialist confirming the account first.
  <br><br>
  Uploaded data is scored in memory and <strong>never written to disk or retained</strong>
  after your session ends.
</div>
""",
        unsafe_allow_html=True,
    )

    threshold = active_threshold(metadata)
    tiers = schema.get("risk_tiers", {})
    bands = (
        float(tiers.get("low_max_exclusive", 0.40)),
        float(tiers.get("medium_max_exclusive", 0.70)),
    )

    with st.expander("Required file format", expanded=False):
        st.markdown(
            f"A CSV containing one row per customer and every model predictor as a column, "
            f"named exactly as below. A `{schema.get('excluded_identifier', 'customerID')}` "
            f"column is optional but recommended — it is carried through for traceability and "
            f"is **never** used as a predictor. Maximum {batch_module.MAX_ROWS:,} rows."
        )
        st.code(", ".join(schema["feature_order"]), language="text")
        st.caption(
            "The raw project dataset `data/raw/Telco-Customer-Churn.csv` already has this "
            "shape and can be uploaded directly to see the queue populated."
        )

    uploaded = st.file_uploader(
        "Customer file (CSV)",
        type=["csv"],
        help="Scored in memory. Nothing is stored.",
    )

    if uploaded is None:
        st.markdown(
            """
<div class="cci-empty">
  <strong>No file uploaded</strong>
  Upload a customer CSV to produce a ranked retention work queue.
</div>
""",
            unsafe_allow_html=True,
        )
        return

    try:
        frame = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        logger.exception("Uploaded file could not be parsed as CSV")
        st.error(
            "That file could not be read as a CSV. Check that it is comma-separated and "
            "has a header row."
        )
        return

    validation = batch_module.validate_batch(frame, schema)
    for message in validation.warnings:
        st.warning(message)
    if not validation.ok:
        for message in validation.errors:
            st.error(message)
        st.info("Fix the issues above and upload the file again. Nothing was scored.")
        return

    try:
        with st.spinner(f"Scoring {len(frame):,} customers…"):
            queue = batch_module.score_batch(
                artifacts["pipeline"],
                frame,
                schema,
                threshold,
                validation.identifier_column,
                bands,
            )
    except Exception:  # noqa: BLE001
        logger.exception("Batch scoring failed")
        st.error(USER_SAFE_ERROR)
        return

    summary = batch_module.queue_summary(queue, threshold)
    logger.info(
        "Batch scored: rows=%d flagged=%d threshold=%.2f",
        summary["total"], summary["flagged"], threshold,
    )

    cards = [
        ("Customers scored", f"{summary['total']:,}", "In this upload"),
        ("Flagged for review", f"{summary['flagged']:,}",
         f"{summary['flagged_share'] * 100:.1f}% at threshold {threshold:.2f}"),
        ("High risk band", f"{summary['high']:,}", "Probability ≥ 0.70"),
        ("Medium risk band", f"{summary['medium']:,}", "0.40 ≤ probability < 0.70"),
    ]
    html = ['<div class="cci-kpis">']
    for label, value, note in cards:
        html.append(
            f'<div class="cci-kpi"><div class="cci-kpi-label">{escape(label)}</div>'
            f'<div class="cci-kpi-value">{escape(value)}</div>'
            f'<div class="cci-kpi-note">{escape(note)}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.write("")

    display = queue.copy()
    display["Churn probability"] = display["Churn probability"].map(lambda v: f"{v:.1%}")
    st.dataframe(display, width="stretch", hide_index=True, height=460)

    st.download_button(
        "Download the ranked queue (CSV)",
        data=queue.to_csv(index=False).encode("utf-8"),
        file_name="retention_work_queue.csv",
        mime="text/csv",
        width="stretch",
    )

    st.caption(
        "Ranked by estimated churn probability at the current decision threshold. Adjust the "
        "threshold in the sidebar to change how many accounts are flagged. Every account in "
        "this queue requires human review before any customer is contacted, and the ranking "
        "reflects association within a fictional training sample rather than certainty about "
        "any individual."
    )


if __name__ == "__main__":
    main()
