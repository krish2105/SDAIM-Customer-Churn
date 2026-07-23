<div align="center">

# 📉 Customer Churn Intelligence

### Retention Decision-Support Platform

**End-to-end machine learning — from a provenance-verified dataset to an automatically deployed, live web application.**

[![Live App](https://img.shields.io/badge/▶_Live_App-krish21may--churn.hf.space-1E40AF?style=for-the-badge)](https://krish21may-churn.hf.space)
[![Hugging Face Space](https://img.shields.io/badge/🤗_Space-krish21may%2Fchurn-FF9D00?style=for-the-badge)](https://huggingface.co/spaces/krish21may/churn)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/CI-passing-2EA043?logo=githubactions&logoColor=white)
![Tests](https://img.shields.io/badge/tests-115_passing-2EA043)
![License](https://img.shields.io/badge/dataset-Apache--2.0-blue)

</div>

> [!NOTE]
> **Data disclaimer.** Trained on IBM's publicly published Telco Customer Churn sample, which represents a **fictional** telecommunications company. It is not actual observed commercial customer data. This platform demonstrates technical feasibility and **may support prioritisation** of accounts for human review. No revenue, churn-reduction or return-on-investment effect has been measured, and none is claimed.

---

## 🔗 Quick links

| Resource | Link |
|---|---|
| 🟢 **Live application** | **https://krish21may-churn.hf.space** |
| 🤗 **Hugging Face Space** | https://huggingface.co/spaces/krish21may/churn |
| 🐙 **GitHub repository** | https://github.com/krish2105/SDAIM-Customer-Churn |
| ✅ **Successful CI run** | [Actions run 30019016553](https://github.com/krish2105/SDAIM-Customer-Churn/actions/runs/30019016553) |
| 🚀 **Successful deploy run** | [Actions run 30020847651](https://github.com/krish2105/SDAIM-Customer-Churn/actions/runs/30020847651) |
| 📄 **Full report** | [`Customer_Churn_Intelligence_Final_Report.pdf`](Customer_Churn_Intelligence_Final_Report.pdf) |

---

## 🎯 What this is

A telecommunications retention team has limited time and no consistent basis for deciding **which customer accounts deserve attention first**. This platform scores a customer and returns a churn probability, a Low / Medium / High risk band, and a cautious suggested review action — **decision support for a human specialist, never an automated decision.**

<div align="center">

| Metric (held-out test set, 1,409 customers) | Value |
|:---|:---:|
| **ROC-AUC** | **0.8414** |
| **Recall (churn)** | **0.7834** |
| Precision (churn) | 0.5034 |
| F1 (churn) | 0.6130 |
| Accuracy | 0.7374 |

*Selected model: Logistic Regression · identifies 293 of 374 actual churners · misses 81*

</div>

Recall is weighted deliberately: a missed churner receives no review at all and the opportunity is lost silently, whereas an unnecessary review costs a specialist's time and is recoverable.

---

## ✨ Highlights

- 🔐 **Provenance proven, not assumed** — the dataset's Git blob SHA is verified against the official IBM publication on every run, locally and in CI.
- 🧪 **Unbiased evaluation** — the model-selection rule was fixed in code *before* the test set was examined.
- ⚖️ **Fairness audited with a measured counterfactual** — removing the protected attributes was found to cost only +0.0008 ROC-AUC.
- 📊 **Calibration predicted, then confirmed** — the model was expected to be over-confident (balanced class weights) and is, by 0.15.
- 🔎 **Exact explainability** — per-prediction log-odds contributions that reconstruct the model to floating-point precision (not SHAP).
- 📦 **Batch scoring** — upload a customer book, get a ranked retention work queue; 1,000 rows in 0.01 s.
- 🤖 **Fully automated deployment** — push to `main` → validate → sync → the Space rebuilds itself, verified end-to-end with a visible-change test.
- ✅ **115 automated tests · 10/10 quality gates · £0 running cost.**

---

## 🏗️ Architecture

```
 Raw CSV (immutable, SHA-verified)
      │
      ▼  30-check validation gate — training refuses to proceed on failure
 One stratified 80/20 split  ── before any transformer is fitted (leakage-safe)
      │
      ▼  Pipeline[ impute + scale + one-hot → estimator ]  (one artifact)
 5-fold CV on the training split only → model selection → single held-out eval
      │
      ▼
 Streamlit app  ──►  Docker image  ──►  Hugging Face Space
                                          ▲
                        GitHub Actions ───┘  (automatic on push to main)
```

Full training, inference and CI/CD diagrams: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 📂 Repository structure

```
SDAIM-Customer-Churn/
├── data/raw/                    Immutable, SHA-verified dataset
├── src/                         Validation, EDA, training, evaluation, and the
│                                governance analyses (fairness, calibration,
│                                threshold, drift, tracking, features, tuning)
├── notebooks/                   01_eda_and_modeling.ipynb  (executed)
├── reports/                     Figures, tables and analysis reports
├── deploy/                      ◄── everything that ships to the Space
│   ├── app.py  theme.py  charts.py  explain.py  batch.py  rationale.py
│   ├── Dockerfile  requirements.txt  README.md
│   └── artifacts/              model_pipeline.joblib + metadata, schema, card
├── tests/                       115 automated tests
├── scripts/                     bootstrap · validate · train · test · docker ·
│                                secret-scan · release-verify · report · shots
├── docs/                        Audit trail and all supporting documentation
└── .github/workflows/          ci.yml  ·  deploy.yml
```

The brief's suggested minimal set (`app.py`, the model artifact, `requirements.txt`, `README.md`, `.github/workflows/deploy.yml`) all live inside `deploy/` — kept separate so training code never reaches the container.

---

## 🚀 Quick start

**Requires Python 3.11** (matching the container runtime). Docker is optional.

```bash
make bootstrap        # create .venv, install deps, validate the dataset
source .venv/bin/activate
```

Then, all from the repository root:

```bash
make validate         # verify the dataset against its documented contract
make eda              # regenerate figures, tables and observations
make train            # train, compare, select and export the pipeline
make analysis         # fairness, calibration, threshold and drift reports
make test             # run the full pytest suite (115 tests)
make app              # run the app locally at http://localhost:8501
make docker-run       # build, run and health-check at http://localhost:7860
make verify           # every non-interactive local quality gate (10)
```

Run `make help` for the full list.

---

## 📈 Model comparison

Cross-validation on the training split only; test columns from the held-out 20%.

| Model | Role | CV ROC-AUC | Test ROC-AUC | Test recall | Test F1 | Test accuracy |
|---|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **selected** | 0.8460 ± 0.0124 | **0.8414** | 0.7834 | 0.6130 | 0.7374 |
| Random Forest | candidate | 0.8454 ± 0.0091 | 0.8417 | 0.7834 | 0.6349 | 0.7608 |
| Dummy (stratified) | baseline | 0.5065 | 0.5163 | 0.2914 | 0.2903 | 0.6217 |

**Selection.** The candidates tied on cross-validated ROC-AUC (gap 0.0005) and F1 (gap 0.0010), so the pre-declared rule fell through to CV recall, where Logistic Regression led 0.8013 to 0.7632. On the held-out set the Random Forest is marginally better on three metrics — the model was *not* switched, because reselecting after seeing test results would invalidate the evaluation. This is documented openly in [`docs/DECISIONS.md`](docs/DECISIONS.md) (D-08).

**Why not accuracy alone?** The sample is 26.54% churners, so always predicting "retained" scores 73.46% accuracy while finding nobody at risk. The dummy baseline confirms it.

---

## 🔬 Beyond the brief — governance analyses

Each of these closes a limitation the project published about itself.

| Analysis | Finding | Report |
|---|---|---|
| **Fairness audit** | No material disparity on `gender`. Material on `SeniorCitizen`, driven largely by a genuine base-rate gap. Removal recommended (costs +0.0008 ROC-AUC). | [`fairness_report.md`](reports/fairness_report.md) |
| **Calibration** | Over-confident by 0.15; isotonic cuts ECE 0.1503 → 0.0194 with ROC-AUC unchanged. | [`calibration_report.md`](reports/calibration_report.md) |
| **Threshold** | Cost-ratio sensitivity curve; the deployed 0.50 is optimal at ≈3:1. | [`threshold_analysis.md`](reports/threshold_analysis.md) |
| **Drift apparatus** | Validated both ways: stable on control, alert on a simulated shift. | [`drift_report.md`](reports/drift_report.md) |
| **Feature engineering + tuning** | Tried, measured, **not adopted** — best gain +0.0022, below the 0.005 bar. | [`tuning_experiment.md`](reports/tuning_experiment.md) |
| **Experiment tracking** | MLflow runs, model registry, documented rollback. | [`tracking_report.md`](reports/tracking_report.md) |

---

## 🤖 Automated deployment

Pushing a change to `main` that touches `deploy/**` triggers the deployment workflow with **no manual step**:

```
push to main (deploy/**)
   └─ job: validate ── dataset check · 115 tests · secret scan
        └─ (only if green) job: deploy
             ├─ verify HF_SPACE_ID variable + HF_TOKEN secret exist
             ├─ verify the deployment package is complete
             └─ huggingface/hub-sync ──► Space rebuilds its Docker image
```

The token is read only as `${{ secrets.HF_TOKEN }}` — never echoed, never in a URL, never in a log. Verified end-to-end by a **visible-change test**: version `1.0.0 → 1.1.0` was pushed and appeared live automatically ([run 30023117028](https://github.com/krish2105/SDAIM-Customer-Churn/actions/runs/30023117028)).

---

## 🔒 Security & governance

- No credential exists anywhere in this repository. `HF_TOKEN` is a GitHub Actions **secret**; `HF_SPACE_ID` is a repository **variable**.
- A secret scanner runs locally, in CI, and before deployment — and was **itself verified** against synthetic credentials.
- The container runs as a **non-root** user from a slim base with only runtime files.
- The model must **never** autonomously change prices, modify contracts, deny service or make any customer-treatment decision. **Every output requires human review.**

Full detail: [`docs/SECURITY.md`](docs/SECURITY.md) · [`deploy/artifacts/model_card.md`](deploy/artifacts/model_card.md).

---

## ⚠️ Limitations

- Trained on a **fictional** sample; live performance is unknown.
- A single cross-section — no real drift can be observed (the detector is apparatus, validated on a simulated shift).
- Probabilities are **not calibrated** (over-confident by ~0.15); trust the ranking more than the magnitude.
- The 0.50 threshold is an assumption, not an optimum.
- Association, never causation. No business impact has been measured — that would require an A/B evaluation.

---

## 👥 Team

| Member | Student ID | Primary contribution |
|---|---|---|
| **Krishna Mathur** | AS25DXB018 | Platform, Docker, CI/CD, security and deployment |
| **Yash Petkar** | AS25DXB021 | Modelling rigour — fairness, calibration, tracking |
| **Atharva Soundankar** | AS25DXB020 | Application, explainability, batch scoring, drift |

**Module:** SDAIM (Term 3) · **Instructor:** JP Aggarwal

---

## 📚 Documentation index

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Training, inference and CI/CD diagrams |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 29 design decisions with alternatives |
| [`docs/SECURITY.md`](docs/SECURITY.md) | Credentials, scanning, container hardening |
| [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) | What is tested, and what is deliberately not |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Documented failure modes and fixes |
| [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) | Three-horizon roadmap (H1 & H2 delivered) |
| [`docs/QUALITY_GATE_RESULTS.md`](docs/QUALITY_GATE_RESULTS.md) | Recorded results of all 10 gates |
| [`Customer_Churn_Intelligence_Final_Report.pdf`](Customer_Churn_Intelligence_Final_Report.pdf) | The full submission report |

---

<div align="center">

**Customer Churn Intelligence and Retention Decision-Support Platform** · Version 1.1.0

Built with reproducibility, provenance and governance as first-class concerns.

*Dataset: IBM Telco Customer Churn sample (fictional company, Apache-2.0). Decision support only — human review required.*

</div>
