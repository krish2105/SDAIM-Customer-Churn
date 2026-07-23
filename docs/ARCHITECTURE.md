# Architecture

Three flows: how the model is produced, how a prediction is served, and how a change
reaches the deployed application.

---

## 1. Training flow

```mermaid
flowchart TD
    A["data/raw/Telco-Customer-Churn.csv<br/>immutable, blob SHA verified"] --> B["src/data_validation.py<br/>30 contract checks"]
    B -->|any check fails| STOP["Training refuses to start<br/>error report written"]
    B -->|all checks pass| C["src/train.py :: load_model_frame<br/>strip + coerce TotalCharges<br/>SeniorCitizen to string"]
    C --> D["split_features_target<br/>drop customerID, map Churn to 0/1"]
    D --> E["train_test_split<br/>80/20 stratified, seed 42<br/>ONE split, before any fit"]
    E --> F["X_train, y_train"]
    E --> G["X_test, y_test<br/>held out, untouched"]

    F --> H["Stratified 5-fold CV<br/>on the training split only"]
    H --> I1["Logistic Regression<br/>impute + scale + one-hot"]
    H --> I2["Random Forest<br/>impute + one-hot"]
    H --> I3["DummyClassifier<br/>reference, not a candidate"]
    I1 --> J["select_model<br/>rule fixed in code before<br/>any test result exists"]
    I2 --> J
    J --> K["Selected model refitted<br/>on the full training split"]

    K --> L["Evaluate ONCE on the held-out set"]
    G --> L
    L --> M["reports/<br/>model_comparison.csv<br/>executive_model_summary.md<br/>figures + tables"]
    K --> N["deploy/artifacts/<br/>model_pipeline.joblib<br/>model_metadata.json<br/>feature_schema.json<br/>reference_rates.json<br/>model_card.md"]

    style STOP fill:#fdecea,stroke:#b4232b,color:#7a1a1f
    style G fill:#eef3fb,stroke:#1e40af,color:#12306e
    style N fill:#e4f5ea,stroke:#15803d,color:#0f5228
```

**The leakage control is structural, not procedural.** Every imputer, encoder and scaler
lives inside the `Pipeline`, so `cross_validate` refits them from scratch on each fold. No
statistic computed on the test split can reach a fitted parameter.

---

## 2. Inference flow

```mermaid
flowchart LR
    subgraph Container["Docker container, non-root, port 7860"]
        direction TB
        S1["app.py starts<br/>NO training at startup"] --> S2["load_artifacts()<br/>@st.cache_resource<br/>loaded once per process"]
        S2 --> S3["feature_schema.json<br/>builds every control"]
    end

    U["Retention specialist"] --> F1["Form: 4 grouped sections<br/>19 predictors"]
    S3 --> F1
    F1 --> F2["Dependency rules enforced<br/>no phone to No phone service<br/>no internet to No internet service"]
    F2 --> F3["Validation<br/>numeric values >= 0<br/>plausibility notes"]
    F3 --> F4["One-row DataFrame<br/>exact training column order"]
    F4 --> P["pipeline.predict_proba"]
    S2 --> P
    P --> R1["Probability"]
    P --> R2["Class at threshold 0.50"]
    R1 --> R3["Risk band<br/>Low / Medium / High"]
    R3 --> R4["Cautious review action"]
    R1 --> C1["Context charts<br/>from reference_rates.json"]
    R4 --> H["HUMAN REVIEW<br/>required before any<br/>customer is contacted"]
    C1 --> H

    style H fill:#fbf0df,stroke:#b45309,color:#6b3406
    style P fill:#eef3fb,stroke:#1e40af,color:#12306e
```

Errors are handled at two points — artifact load and prediction. Both log the technical
detail server-side and show the user a message containing no path, stack frame or internal
identifier.

---

## 3. CI/CD flow

```mermaid
flowchart TD
    D["Developer commits<br/>tests + secret scan run locally first"] --> PR{"Pull request<br/>or push to main?"}

    PR -->|pull request| CI["ci.yml"]
    PR -->|push to main| CI
    CI --> CI1["checkout@v6 with LFS"]
    CI1 --> CI2["setup-python@v6, Python 3.11"]
    CI2 --> CI3["install requirements-dev.txt"]
    CI3 --> CI4["validate dataset --strict-sha"]
    CI4 --> CI5["compileall src tests deploy"]
    CI5 --> CI6["pytest"]
    CI6 --> CI7["scan_secrets.sh"]
    CI7 --> CI8["verify deployment files<br/>and Space metadata"]

    PR -->|push to main touching deploy/**| DEP["deploy.yml"]
    DEP --> V["job: validate<br/>dataset + pytest + secret scan"]
    V -->|fails| X["Deployment never runs"]
    V -->|passes| G1["job: deploy<br/>needs: validate"]
    G1 --> G2{"HF_SPACE_ID variable<br/>and HF_TOKEN secret<br/>both configured?"}
    G2 -->|no| X2["Fail with an actionable message<br/>token value never printed"]
    G2 -->|yes| G3["huggingface/hub-sync@v0.2.1<br/>repo_type: space<br/>space_sdk: docker<br/>subdirectory: deploy"]
    G3 --> HF["Hugging Face Space<br/>rebuilds the Docker image"]
    HF --> OBS["Build and live app must be<br/>OBSERVED before any claim<br/>of successful deployment"]

    style X fill:#fdecea,stroke:#b4232b,color:#7a1a1f
    style X2 fill:#fdecea,stroke:#b4232b,color:#7a1a1f
    style OBS fill:#fbf0df,stroke:#b45309,color:#6b3406
```

Only `deploy/` is synchronised. Training code, the raw dataset, notebooks, tests and
reports stay in GitHub and never enter the Space.

---

## 3b. Post-training analysis flow

Added in Horizon 1 and 2. Every analysis rebuilds the **same** train/test split from the
immutable raw file, so none can silently evaluate against a different partition than the one
the model was fitted on.

```mermaid
flowchart LR
    A["data/raw (immutable)"] --> B["src/analysis_base.py<br/>rebuilds the exact split<br/>seed 42, stratified"]
    M["deploy/artifacts/<br/>model_pipeline.joblib"] --> B

    B --> F["src/fairness.py<br/>subgroup metrics<br/>+ counterfactual retrain"]
    B --> C["src/calibration.py<br/>reliability, Brier, ECE"]
    B --> T["src/threshold.py<br/>cost-ratio sweep"]
    B --> D["src/drift.py<br/>baseline + PSI + chi-squared"]

    F --> R1["reports/fairness_report.md"]
    C --> R2["reports/calibration_report.md"]
    T --> R3["reports/threshold_analysis.md"]
    D --> R4["reports/drift_report.md"]

    F -.->|regenerates| MC["model_card.md<br/>fairness section written<br/>FROM the audit"]
    D --> BL["artifacts/drift_baseline.json"]

    TR["src/tracking.py<br/>MLflow, SQLite"] --> REG["Registry<br/>alias: production<br/>manual rollback"]

    style B fill:#eef3fb,stroke:#1e40af,color:#12306e
    style MC fill:#e4f5ea,stroke:#15803d,color:#0f5228
```

**Why the model card is generated from the audit.** Version 1.0.0 stated "no fairness audit
has been carried out". Once the audit existed, that claim was false — and a hand-edited card
would drift again. The card's fairness section is now written from
`reports/tables/fairness_report.json`, and a test asserts the card cannot claim the audit is
missing while the report exists.

---

## 3c. Extended inference surface

```mermaid
flowchart TD
    U["Retention specialist"] --> MODE{Single record<br/>or whole book?}

    MODE -->|single| S1["Form: 19 predictors<br/>dependency rules enforced"]
    S1 --> P["pipeline.predict_proba"]
    P --> R["Probability · class · risk band<br/>· cautious review action"]
    P --> E["deploy/explain.py<br/>coefficient x value<br/>reconstructs score EXACTLY"]
    E --> CH["Contribution chart<br/>+ causal disclaimer"]
    R --> BR["deploy/rationale.py<br/>retention brief"]
    E --> BR

    MODE -->|batch| B1["deploy/batch.py<br/>CSV upload"]
    B1 --> B2["Validate vs feature_schema<br/>BEFORE scoring anything"]
    B2 -->|errors| B3["Per-column errors<br/>nothing scored"]
    B2 -->|ok| B4["Score all rows<br/>rank by probability"]
    B4 --> B5["Work queue + CSV export<br/>never written to disk"]

    TH["Threshold slider<br/>default 0.50"] --> R
    TH --> B4

    BR --> G{"Generation enabled?"}
    G -->|no, the default| GT["Deterministic template"]
    G -->|yes| GL["LLM renders computed facts only"]
    GL --> GV{"Structure + language<br/>guardrails pass?"}
    GV -->|no| GT
    GV -->|yes| GO["Labelled AI-generated"]

    R --> H["HUMAN REVIEW REQUIRED"]
    B5 --> H
    GT --> H
    GO --> H

    style H fill:#fbf0df,stroke:#b45309,color:#6b3406
    style B3 fill:#fdecea,stroke:#b4232b,color:#7a1a1f
    style GT fill:#e4f5ea,stroke:#15803d,color:#0f5228
```

The LLM never sees the customer record. It receives only values already computed
deterministically, and the deterministic template is the default rather than a degraded mode.

---

## 4. Repository layout and responsibilities

| Path | Responsibility |
|---|---|
| `data/raw/` | Immutable audited evidence. Never written to. |
| `data/processed/` | Reproducible split exports, for traceability only. Never read back by training. |
| `src/config.py` | Single source of truth for paths, the dataset contract and modelling constants. |
| `src/schemas.py` | The model input contract shared by training and the application. |
| `src/data_validation.py` | 30 contract checks; the gate that training refuses to bypass. |
| `src/eda.py` | Source of truth for every quantitative statement in the report. |
| `src/train.py` | Split, cross-validate, select, evaluate, export. |
| `src/evaluate.py` | Metrics, figures and the executive summary. Reusable without retraining. |
| `src/analysis_base.py` | Shared evaluation context — one split definition for all four analyses. |
| `src/fairness.py` | Subgroup audit and the counterfactual retrain (H1-1). |
| `src/calibration.py` | Reliability, Brier, ECE and calibrator comparison (H1-2). |
| `src/threshold.py` | Cost-ratio sensitivity sweep (H1-3). |
| `src/drift.py` | Baseline profile and the validated detectors (H2-2). |
| `src/tracking.py` | MLflow tracking and registry. Development only (H2-1). |
| `deploy/explain.py` | Exact log-odds decomposition (H1-4). |
| `deploy/batch.py` | Batch validation, scoring and work-queue construction (H2-3). |
| `deploy/rationale.py` | Guardrailed retention brief, disabled by default (H2-4). |
| `deploy/` | Exactly what ships to the Space: app, theme, charts, Dockerfile, artifacts. |
| `tests/` | 52 tests over the data contract, the artifact, prediction behaviour and deployment config. |
| `.github/workflows/` | CI and deployment. |
| `docs/` | Audit trail, decisions, architecture, security, report scaffolding. |

## 5. Boundaries deliberately drawn

- **`deploy/` never imports `src/`.** A test enforces it. The container carries no training
  code, so the image stays small and the deployed surface is minimal.
- **`data/raw/` is written by nothing.** Validation reads it; EDA and training copy from it.
- **The application never trains.** It loads one artifact and scores one row.
- **The model never acts.** It returns a number, a band and a suggestion; a person decides.
- **MLflow never reaches the container.** It is a development dependency; a test asserts the
  application does not import it, so the runtime image stays small.
- **The LLM never reasons.** It renders values the deterministic pipeline already produced.
