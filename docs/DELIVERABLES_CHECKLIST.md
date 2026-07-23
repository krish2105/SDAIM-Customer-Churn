# Deliverables Checklist

Status is factual: **Complete** means it exists and was verified in this build.
**Pending** means it requires platform credentials or evidence that only you can produce.

---

## A. Local deliverables — all complete

| # | Deliverable | Location | Status | Verification |
|---|---|---|---|---|
| 1 | Official validated CSV | `data/raw/Telco-Customer-Churn.csv` | ✅ Complete | Git blob SHA matches `3de7a61…9002`; 30 checks pass |
| 2 | Data dictionary / provenance | `docs/DATA_DICTIONARY.csv`, `SOURCE_MANIFEST.json`, `LICENSE_NOTICE.md` | ✅ Complete | Category domains cross-checked against the data |
| 3 | Reproducible EDA script | `src/eda.py` | ✅ Complete | `make eda` regenerates every artifact |
| 4 | EDA notebook | `notebooks/01_eda_and_modeling.ipynb` | ✅ Complete | Generated and **executed**; 33 cells, no errors |
| 5 | EDA figures and observations | `reports/figures/` (11 PNG), `reports/eda_observations.md` | ✅ Complete | Every stated number computed by code |
| 6 | Leakage-safe preprocessing | `src/train.py` | ✅ Complete | Single split before any fit; all transformers inside the Pipeline |
| 7 | Two trained substantive models | Logistic Regression, Random Forest | ✅ Complete | Plus a stratified baseline for context |
| 8 | Model-comparison table | `reports/model_comparison.csv` | ✅ Complete | CV and held-out metrics, both models and the baseline |
| 9 | Selected Joblib pipeline | `deploy/artifacts/model_pipeline.joblib` | ✅ Complete | 8.4 KB; loads and scores in a fresh process |
| 10 | Metadata and feature schema | `model_metadata.json`, `feature_schema.json` | ✅ Complete | Real measured values; placeholder tokens rejected by test |
| 11 | Model card | `deploy/artifacts/model_card.md` | ✅ Complete | Use, out-of-scope uses, provenance, limits, ethics, human review |
| 12 | Streamlit application | `deploy/app.py` + `theme.py` + `charts.py` | ✅ Complete | Verified in browser, both themes, prediction confirmed |
| 13 | Runtime requirements | `deploy/requirements.txt` | ✅ Complete | Pinned; coverage of every runtime import asserted by test |
| 14 | Dockerfile | `deploy/Dockerfile` | ✅ Complete | Built, run, health check passed |
| 15 | Space README | `deploy/README.md` | ✅ Complete | Valid front matter: `sdk: docker`, `app_port: 7860` |
| 16 | Automated tests | `tests/` | ✅ Complete | **52 passed** |
| 17 | CI workflow | `.github/workflows/ci.yml` | ✅ Complete | All eight required steps |
| 18 | Deployment workflow | `.github/workflows/deploy.yml` | ✅ Complete | Validation gate, least privilege, concurrency, config checks |
| 19 | Security and secret-scan controls | `scripts/scan_secrets.sh`, `docs/SECURITY.md`, `.gitignore` | ✅ Complete | Scanner verified against synthetic credentials |
| 20 | Root README | `README.md` | ✅ Complete | All 18 required sections |
| 21 | Report template | `docs/REPORT_TEMPLATE.md` | ✅ Complete | 33 sections; measured values pre-filled |
| 22 | Screenshot checklist | `docs/SCREENSHOT_CHECKLIST.md` | ✅ Complete | 37 figures with captions and proof statements |
| 23 | Demonstration script | `docs/DEMONSTRATION_SCRIPT.md` | ✅ Complete | Five-minute script plus 10 viva questions |
| 24 | Troubleshooting guide | `docs/TROUBLESHOOTING.md` | ✅ Complete | 15 documented failure modes |
| 25 | Deliverables checklist | This file | ✅ Complete | — |
| 26 | Quality-gate results | `docs/QUALITY_GATE_RESULTS.md` | ✅ Complete | Actual recorded output |
| 27 | Manual commands for external setup | `README.md` §8, `docs/IMPLEMENTATION_PLAN.md` phases 9–11 | ✅ Complete | Exact commands provided, none executed |

## A2. Horizon 1 and 2 deliverables — all complete

| # | Deliverable | Location | Status | Verification |
|---|---|---|---|---|
| 28 | Fairness audit | `src/fairness.py`, `reports/fairness_report.md` | ✅ Complete | 2 attributes, 3 criteria, counterfactual retrain measured |
| 29 | Calibration analysis | `src/calibration.py`, `reports/calibration_report.md` | ✅ Complete | Over-confidence of 0.15 measured; isotonic cuts ECE to 0.0194 |
| 30 | Threshold analysis | `src/threshold.py`, `reports/threshold_analysis.md` | ✅ Complete | Cost-ratio curve, 8 ratios; app exposes the control |
| 31 | Per-prediction explainability | `deploy/explain.py` | ✅ Complete | Reconstructs the model exactly, asserted on 25 customers |
| 32 | Experiment tracking + registry | `src/tracking.py`, `reports/tracking_report.md` | ✅ Complete | 3 runs, registry alias, documented rollback |
| 33 | Drift apparatus | `src/drift.py`, `reports/drift_report.md` | ✅ Complete | Validated both ways: stable on control, alert on shift |
| 34 | Batch scoring work queue | `deploy/batch.py` | ✅ Complete | 1,000 rows in 0.01 s; export; nothing written to disk |
| 35 | Guardrailed retention brief | `deploy/rationale.py` | ✅ Complete | Disabled by default; fallback passes its own filter |
| 36 | Improvement plan | `docs/IMPROVEMENT_PLAN.md` | ✅ Complete | Three horizons; H1 and H2 marked delivered |

## B. Supporting documentation — all complete

| Deliverable | Location | Status |
|---|---|---|
| Input audit | `docs/INPUT_AUDIT.md` | ✅ Complete |
| Implementation plan | `docs/IMPLEMENTATION_PLAN.md` | ✅ Complete |
| Implementation log | `docs/IMPLEMENTATION_LOG.md` | ✅ Complete |
| Design decisions | `docs/DECISIONS.md` | ✅ Complete — 20 decisions |
| Architecture with Mermaid diagrams | `docs/ARCHITECTURE.md` | ✅ Complete — training, inference, CI/CD |
| Security documentation | `docs/SECURITY.md` | ✅ Complete |
| Test plan | `docs/TEST_PLAN.md` | ✅ Complete — 106 tests across six modules |
| Improvement plan | `docs/IMPROVEMENT_PLAN.md` | ✅ Complete — H1 and H2 delivered |

---

## C. External deliverables — six of eight verified

Six have now been completed and verified by direct observation on the real platforms. The
remaining two are yours: capturing the screenshots and writing the report.

| # | Deliverable | Status | How to complete | Evidence to record |
|---|---|---|---|---|
| 1 | GitHub repository URL | ✅ **Complete** | Create the repository, add the remote, push `main` | The URL, opened and confirmed |
| 2 | Real commit hashes | ✅ **Complete** | Push; read the SHAs from GitHub | Full SHAs from the platform |
| 3 | Successful CI run | ✅ **Complete** | Triggered automatically by the push | Run URL, run ID, timestamp, all steps green |
| 4 | Hugging Face Space URL | ✅ **Complete** | Create a Docker Space; set `HF_SPACE_ID` | The Space URL |
| 5 | Successful Space build | ✅ **Complete** | Triggered by the deployment workflow | Build log screenshot |
| 6 | Visible automated version update | ✅ **Complete** | Follow `docs/DEMONSTRATION_SCRIPT.md` | Second run + 1.1.0 live |
| 7 | Authentic screenshots | ⬜ Pending | `docs/SCREENSHOT_CHECKLIST.md` figures 23–37 | Genuine captures, checked for credentials |
| 8 | Final PDF report | ⬜ Pending | Complete `docs/REPORT_TEMPLATE.md` | Every `<<UNRESOLVED>>` replaced |

**All 37 figures** of the screenshot checklist can now be captured: the application runs
locally and in Docker, the repository and both workflow runs are public, and the Space is
live at <https://krish21may-churn.hf.space>.

---

## D. Unresolved placeholders

Every one of these is deliberately marked rather than invented. All deployment-related
placeholders have now been resolved with verified values; only academic and instructor
inputs remain.

| Placeholder | Files | Needed for |
|---|---|---|
| Course, group number, member names, student IDs, instructor, submission date | `PROJECT_INPUTS.md`, `README.md` §17, `docs/REPORT_TEMPLATE.md` §1, §32 | Report cover and team table |
| Instructor confirmation on the fictional-data question | `PROJECT_INPUTS.md`, report §6 | Academic assurance |
| Report format, presentation duration, repository visibility | `PROJECT_INPUTS.md` | Submission compliance |

---

## E. Requirements traceability against the instructor brief

| Brief requirement | Where satisfied | Status |
|---|---|---|
| Task 1.1 — dataset selection, loading, understanding, EDA | `src/eda.py`, `reports/`, notebook | ✅ |
| Task 1.2 — preprocessing, encoding, scaling, split, leakage prevention | `src/train.py` | ✅ |
| Task 1.3 — train and compare ≥ 2 models, justify metrics and selection | `src/train.py`, `reports/model_comparison.csv`, D-06/D-08 | ✅ |
| Task 1.4 — save the model with preprocessing (or the whole pipeline) | `model_pipeline.joblib` — complete pipeline | ✅ |
| Task 2.1 — interactive UI with inputs, predict action, clear output, probability | `deploy/app.py` | ✅ |
| Task 2.2 — install dependencies, run locally, test inputs, verify preprocessing and loading | `make app`, 52 tests, browser verification | ✅ |
| Task 3.1 — create a Space with all required files | `krish21may/churn` live | ✅ **Verified** |
| Task 3.2 — build and test on the platform | Space built, RUNNING, prediction confirmed | ✅ **Verified** |
| Task 4.1 — GitHub repository with application and deployment files | https://github.com/krish2105/SDAIM-Customer-Churn | ✅ **Verified** |
| Task 4.2 — HF token stored securely, never hard-coded | Fine-grained token scoped to the single Space, stored as the `HF_TOKEN` Actions secret | ✅ **Verified** |
| Task 4.3 — workflow on push to main that syncs and updates the Space | Runs 30020847651 and 30023117028 both green | ✅ **Verified** |
| Task 4.4 — visible change deploys automatically and is verified | 1.0.0 → 1.1.0 confirmed live after an automatic redeploy | ✅ **Verified** |
| Part 5 — report with all listed sections and screenshots | `docs/REPORT_TEMPLATE.md`, `docs/SCREENSHOT_CHECKLIST.md` | ✅ Scaffolded; ⬜ evidence pending |

---

## F. Final pre-submission gate

- [ ] All `<<UNRESOLVED>>` placeholders replaced with verified values
- [ ] Repository pushed; commit SHAs recorded from the platform
- [ ] CI run green, URL recorded
- [ ] Space created, built and live; URL recorded
- [ ] Visible-change test completed and verified in the live app
- [ ] All 37 screenshots captured and checked for credentials
- [ ] Report completed and exported
- [ ] `make verify` re-run and `docs/QUALITY_GATE_RESULTS.md` refreshed
- [ ] Team contribution table completed and signed
- [ ] No claim of measured revenue, churn reduction or ROI anywhere
- [ ] Dataset described as fictional everywhere it appears
