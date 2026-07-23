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
