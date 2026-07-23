# Customer Churn Intelligence and Retention Decision-Support Platform

End-to-end machine-learning project: validated data, a leakage-safe pipeline, two compared
models, a saved artifact, a professional Streamlit application, Docker packaging, and
automated deployment to a Hugging Face Space via GitHub Actions.

> **Data notice.** Trained on IBM's publicly published Telco Customer Churn sample, which
> represents a **fictional** telecommunications company. It is not actual observed
> commercial customer data. No revenue, churn-reduction or return-on-investment effect has
> been measured, and none is claimed.

---

## 1. Executive summary

A telecommunications organisation has no consistent way to judge which customer accounts
deserve a retention specialist's attention. This project delivers a decision-support
application that scores one customer record and returns a churn probability, a Low /
Medium / High risk band, and a cautious suggested review action.

The selected model — **Logistic Regression** inside a complete scikit-learn pipeline —
achieves **0.8414 ROC-AUC** and **0.7834 recall** on a held-out test set of 1,409
customers that no model saw during fitting or selection. It identifies **293 of the 374**
actual churners in that set and misses 81.

Recall is weighted deliberately: a missed churner receives no review at all, while an
unnecessary review costs specialist time and is recoverable.

The application demonstrates technical feasibility and **may support prioritisation** of
accounts for human review. It decides nothing.

## 2. Business problem

| Aspect | Detail |
|---|---|
| Problem | No consistent, repeatable basis for judging which accounts are at elevated churn risk |
| Decision supported | Prioritising customer accounts for a structured human retention review |
| Stakeholders | Chief Customer Officer, Chief Marketing Officer, retention leadership, customer-service managers, data and analytics teams, model-risk and governance reviewers |
| Outputs | Predicted class, churn probability, risk band, cautious review action, model version, stated limitations |
| Explicitly out of scope | Changing prices, terminating or modifying contracts, denying service, any automated customer-treatment decision |

## 3. Dataset provenance

| Property | Value |
|---|---|
| Dataset | IBM Telco Customer Churn sample |
| Repository | https://github.com/IBM/telco-customer-churn-on-icp4d |
| Licence | Apache-2.0 |
| Rows × columns | 7,043 × 21 |
| Verified Git blob SHA | `3de7a612d1609f25f21a455bda77948729369002` |
| Target | `Churn` (Yes → 1, No → 0) |
| Excluded identifier | `customerID` |

The file in `data/raw/` is byte-identical to the official published file and is never
modified. Validation reads it; all cleaning happens in code.

**The brief asks for a "real-world dataset".** This sample is officially published,
realistically structured and fully traceable, but it represents a fictional company.
Instructor confirmation is recorded as pending in `PROJECT_INPUTS.md`; see decision D-01
in `docs/DECISIONS.md`.

## 4. Target and features

19 predictors — 16 categorical and 3 numeric:

| Group | Features |
|---|---|
| Customer profile | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Contract and billing | `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Charges and tenure | `tenure`, `MonthlyCharges`, `TotalCharges` |

`SeniorCitizen` is treated as a binary category, not a magnitude. `customerID` never enters
the feature matrix.

## 5. Architecture

```
Raw CSV (immutable)
   → 30-check validation gate  (training refuses to proceed on failure)
   → one stratified 80/20 split, before any transformer is fitted
   → Pipeline[ ColumnTransformer(impute+scale | impute+one-hot) → estimator ]
   → 5-fold stratified CV on the training split only → model selection
   → single held-out evaluation → artifacts + reports
   → Streamlit app loads exactly that artifact
   → Docker image → Hugging Face Space (GitHub Actions)
```

Mermaid diagrams for the training, inference and CI/CD flows are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 6. Repository structure

```
.
├── README.md                     This file
├── LICENSE_NOTICE.md             Dataset and dependency licensing
├── PROJECT_INPUTS.md             Known and unresolved project values
├── SOURCE_MANIFEST.json          Dataset provenance contract
├── Makefile                      All commands, runnable from the root
├── requirements-dev.txt          Development and analysis dependencies
├── data/
│   ├── raw/                      Immutable audited dataset
│   └── processed/                Reproducible split exports (git-ignored)
├── src/                          config, schemas, validation, EDA, train, evaluate
├── notebooks/                    01_eda_and_modeling.ipynb (executed)
├── reports/                      figures/, tables/, model_comparison.csv,
│                                 eda_observations.md, executive_model_summary.md
├── deploy/                       Everything that ships to the Space
│   ├── app.py, theme.py, charts.py
│   ├── .streamlit/config.toml
│   ├── Dockerfile, requirements.txt, README.md, .dockerignore
│   └── artifacts/                model_pipeline.joblib, model_metadata.json,
│                                 feature_schema.json, reference_rates.json, model_card.md
├── tests/                        52 tests
├── scripts/                      bootstrap, validate, eda, train, test, app,
│                                 docker, secret scan, release verification
├── docs/                         Audit trail and all supporting documentation
└── .github/workflows/            ci.yml, deploy.yml
```

## 7. Local setup

Requires **Python 3.11** (matching the container runtime) and, optionally, Docker.

```bash
make bootstrap
```

This creates `.venv`, installs dependencies, creates output directories and validates the
dataset. It installs nothing globally and does not touch Homebrew or your system Python.

Then activate the environment in your own shell:

```bash
source .venv/bin/activate
```

## 8. Commands

All commands run from the repository root.

```bash
make validate      # Validate the raw dataset against the documented contract
make eda           # Regenerate 11 figures, 13 tables and the observations document
make train         # Train, compare, select and export the pipeline and artifacts
make notebook      # Regenerate and execute notebooks/01_eda_and_modeling.ipynb
make test          # Run the full pytest suite
make app           # Run the Streamlit application on http://localhost:8501
make docker-build  # Build the deployment image
make docker-run    # Build, run and health-check on http://localhost:7860
make secret-scan   # Scan the project for credential patterns
make verify        # Run every non-interactive local quality gate
```

## 9. Model comparison and final result

Cross-validation on the training split only; test columns from the held-out 20%.

| Model | Role | CV ROC-AUC | CV F1 | CV recall | Test ROC-AUC | Test F1 | Test recall | Test precision | Test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **selected** | 0.8460 ± 0.0124 | 0.6286 | 0.8013 | 0.8414 | 0.6130 | 0.7834 | 0.5034 | 0.7374 |
| Random Forest | candidate | 0.8454 ± 0.0091 | 0.6296 | 0.7632 | 0.8417 | 0.6349 | 0.7834 | 0.5337 | 0.7608 |
| Dummy (stratified) | baseline | 0.5065 ± 0.0171 | 0.2762 | 0.2776 | 0.5163 | 0.2903 | 0.2914 | 0.2891 | 0.6217 |

**Selection.** The rule was fixed in code before any test result existed: mean CV ROC-AUC,
then CV F1, then CV recall, with a 0.01 tie tolerance. The candidates tied on ROC-AUC (gap
0.0005) and F1 (gap 0.0010), so the rule fell through to CV recall, where Logistic
Regression led 0.8013 to 0.7632.

**Stated plainly:** on the held-out set the Random Forest edges ahead on accuracy,
precision and F1, with identical recall and effectively identical ROC-AUC. The pre-declared
rule was still followed, because reselecting after seeing test results would invalidate the
held-out evaluation. See decision D-08 in `docs/DECISIONS.md`.

**Confusion matrix — selected model, held-out test set (n = 1,409):**

| | Predicted: retained | Predicted: churn |
|---|---:|---:|
| **Actual: retained** | 746 | 289 |
| **Actual: churn** | 81 | 293 |

- 81 of 374 actual churners are missed (21.7%) — the costly error.
- 289 of 582 flagged accounts would have stayed (49.7%) — recoverable cost.
- 582 of 1,409 customers are flagged for review (41.3%) — a bounded workload.

**Why accuracy alone is insufficient.** The sample is 26.54% churners, so always predicting
"retained" scores 73.46% accuracy while finding nobody at risk. The dummy baseline confirms
it: 0.6217 accuracy with 0.2914 recall and 0.5163 ROC-AUC — no better than chance at
ranking.

## 10. Key EDA findings

Every figure is computed by `src/eda.py`; none is hand-entered. Full detail in
[reports/eda_observations.md](reports/eda_observations.md).

- **Churn rate 26.54%** (1,869 of 7,043) — imbalanced, which drives the choice of metrics.
- **Contract term separates most strongly:** month-to-month 42.71%, one year 11.27%, two
  year 2.83%.
- **Payment method:** electronic check 45.29% against 15.24% for automatic credit card.
- **Internet service:** fibre optic 41.89%, DSL 18.96%, none 7.40%.
- **Technical support:** absent 41.64%, present 15.17%.
- **Tenure:** churners average 17.98 months against 37.57 for retained customers;
  correlation with the churn flag −0.352.
- **Missing values:** exactly 11, all in `TotalCharges`, all with `tenure = 0` — customers
  who have not completed a billing cycle. Structurally meaningful, not random.

These are associations within the sample. None is evidence of causation, and the add-on
comparisons are partly confounded by the `No internet service` level appearing in every
add-on column.

## 11. Deployment architecture

```
GitHub repository (main)
   └── push touching deploy/**
         └── .github/workflows/deploy.yml
               ├── job: validate  — dataset, pytest, secret scan
               └── job: deploy    — needs: validate
                     ├── verify HF_SPACE_ID variable and HF_TOKEN secret exist
                     ├── verify the deployment package is complete
                     └── huggingface/hub-sync@v0.2.1 (subdirectory: deploy)
                           └── Hugging Face Docker Space rebuilds → port 7860
```

The Space runs `deploy/Dockerfile`: `python:3.11-slim`, non-root user, only the runtime
files, Streamlit on port 7860.

## 12. GitHub Actions

**`ci.yml`** — on pull requests and pushes to `main`: checkout with LFS, Python 3.11,
install dev dependencies, validate the dataset with `--strict-sha`, `compileall` over
`src tests deploy`, run pytest, run the secret scan, verify every required deployment file
and the Space README metadata.

**`deploy.yml`** — on pushes to `main` touching `deploy/**` (and `workflow_dispatch`):
a `validate` job the deploy job depends on; `permissions: contents: read`; a concurrency
group preventing overlapping deployments; explicit configuration checks that fail with an
actionable message; then the Space sync.

The token is read only as `${{ secrets.HF_TOKEN }}`. It is never echoed, never
interpolated into a URL, and never written to the job summary. A test asserts the workflows
reference no secret other than `HF_TOKEN` and no variable other than `HF_SPACE_ID`.

**A successful sync is not a successful deployment.** The Space build happens afterwards
and must be observed directly.

## 13. Security model

No credential exists anywhere in this repository. Full detail in
[docs/SECURITY.md](docs/SECURITY.md).

- `HF_TOKEN` as a GitHub Actions **secret**; `HF_SPACE_ID` as a repository **variable**.
- Fine-grained Hugging Face token scoped to write to the single target Space.
- `scripts/scan_secrets.sh` runs locally, in CI and again before deployment; it prints file
  paths and pattern categories only, never the matched text.
- Container runs as a non-root user from a slim base with only runtime files.
- The application shows no stack trace, path or internal identifier to the user; technical
  detail goes to the server log.
- No external font, script or stylesheet is loaded — the page is entirely self-contained.

## 14. Limitations

- Trained on a **fictional** sample; performance on any live population is unknown and must
  be revalidated before operational use.
- A single cross-section with no time dimension: drift and seasonality cannot be assessed,
  and no monitoring baseline exists.
- Class imbalance (26.54% positive) limits the precision attainable at high recall — at
  0.7834 recall, roughly half the flagged accounts are false positives.
- The 0.50 threshold is a documented default. No cost matrix was supplied, so no
  cost-sensitive optimisation was performed.
- Risk bands are communication aids for triage, not validated business thresholds.
- `gender` and `SeniorCitizen` are predictors and **no formal fairness audit has been
  carried out**. One is required before any operational use.
- The model produces no explanation of an individual prediction.
- Predictions describe association within the sample, never causation.

## 15. Governance

The model must not autonomously change prices, terminate or modify contracts, deny service,
target customers unfairly, or make any financial or customer-treatment decision. Every
output requires human review before a customer is contacted. See
[deploy/artifacts/model_card.md](deploy/artifacts/model_card.md).

## 16. Links

| Resource | Status |
|---|---|
| GitHub repository | `<<UNRESOLVED — add once the repository exists and has been verified>>` |
| Hugging Face Space | `<<UNRESOLVED — add once the Space is created and its build has been observed>>` |
| Successful CI run | `<<UNRESOLVED — add the real run URL>>` |
| Successful deployment run | `<<UNRESOLVED — add the real run URL>>` |

No link is recorded here until it has been opened and confirmed to work.

## 17. Team contributions

| Member | Student ID | Contribution | Sign-off |
|---|---|---|---|
| `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | |
| `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | |
| `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | |
| `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | `<<UNRESOLVED>>` | |

Complete this table before submission. Suggested contribution areas: data validation and
EDA; preprocessing and modelling; application and front end; Docker and CI/CD; security and
documentation; report and demonstration.

## 18. Documentation index

| Document | Purpose |
|---|---|
| [docs/INPUT_AUDIT.md](docs/INPUT_AUDIT.md) | What was found, what was missing, what was verified |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phases, acceptance criteria, command sequence |
| [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md) | Commands run, results, decisions, open items |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 20 decisions with alternatives and consequences |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Training, inference and CI/CD diagrams |
| [docs/SECURITY.md](docs/SECURITY.md) | Credentials, scanning, rotation, container hardening |
| [docs/TEST_PLAN.md](docs/TEST_PLAN.md) | What is tested, and what is deliberately not |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Eleven documented failure modes and fixes |
| [docs/SCREENSHOT_CHECKLIST.md](docs/SCREENSHOT_CHECKLIST.md) | Every evidence screenshot, with captions |
| [docs/DELIVERABLES_CHECKLIST.md](docs/DELIVERABLES_CHECKLIST.md) | Complete / pending status of every deliverable |
| [docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md) | 33-section report scaffold |
| [docs/DEMONSTRATION_SCRIPT.md](docs/DEMONSTRATION_SCRIPT.md) | Five-minute demo plus viva questions |
| [docs/QUALITY_GATE_RESULTS.md](docs/QUALITY_GATE_RESULTS.md) | Actual recorded gate results |
