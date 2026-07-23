# Final Report Template

Complete this to produce the submission report. Measured values are **pre-filled** — they
come from the actual training run and must not be altered. Anything marked
`<<UNRESOLVED>>` needs authentic evidence from you.

> **Rule.** Never write a URL, commit SHA, run ID, timestamp or screenshot you have not
> personally verified.

---

## 1. Cover page

- **Title:** Customer Churn Intelligence and Retention Decision-Support Platform
- **Subtitle:** End-to-End ML Model Deployment using Hugging Face and GitHub Actions
- **Course / module:** SDAIM (Term 3)
- **Instructor:** JP Aggarwal
- **Members and student IDs:**
  - Krishna Mathur — `AS25DXB018`
  - Yash Petkar — `AS25DXB021`
  - Atharva Soundankar — `AS25DXB020`
- **GitHub repository:** https://github.com/krish2105/SDAIM-Customer-Churn
- **Hugging Face Space:** https://huggingface.co/spaces/krish21may/churn
- **Live application:** https://krish21may-churn.hf.space

---

## 2. Executive summary

*Half a page. Adapt:*

This project delivers a customer-retention decision-support application. A machine-learning
pipeline trained on IBM's published Telco Customer Churn sample estimates the probability
that a customer will churn, and presents it as a risk band with a suggested review action
for a retention specialist.

The selected model, Logistic Regression within a complete scikit-learn pipeline, achieves
**0.8414 ROC-AUC** and **0.7834 recall** on a held-out test set of 1,409 customers,
identifying 293 of 374 actual churners. The application is packaged as a Docker container,
deployed to a Hugging Face Space, and redeployed automatically by GitHub Actions on every
change to the deployment package.

The dataset represents a **fictional** telecommunications company. The work demonstrates
technical feasibility and may support prioritisation of accounts for human review. No
revenue, churn-reduction or return-on-investment effect has been measured or is claimed.

---

## 3. Business context and objective

- Retention specialists have limited time and no consistent basis for prioritising accounts.
- Objective: a repeatable, documented risk estimate that supports — never replaces — human
  judgement.
- Scope boundary: the model informs prioritisation only.

---

## 4. Stakeholders and decision supported

| Stakeholder | Interest |
|---|---|
| Chief Customer Officer | Consistency and defensibility of retention prioritisation |
| Chief Marketing Officer | Targeting efficiency of retention activity |
| Retention leadership | Bounded, prioritised review workload |
| Customer-service managers | Practical guidance at account level |
| Data and analytics | Reproducibility and maintainability |
| Model risk and governance | Documented limitations and human-review controls |

**Decision supported:** prioritising customer accounts for a structured human retention
review.
**Explicitly not supported:** pricing, contract changes, service denial, any automated
customer-treatment decision.

---

## 5. Dataset description and official source

| Property | Value |
|---|---|
| Dataset | IBM Telco Customer Churn sample |
| Repository | https://github.com/IBM/telco-customer-churn-on-icp4d |
| Licence | Apache-2.0 |
| Rows × columns | 7,043 × 21 |
| Target | `Churn` (Yes → 1, No → 0) |
| Excluded identifier | `customerID` |
| Verified Git blob SHA | `3de7a612d1609f25f21a455bda77948729369002` |

*Insert Figure 1 (source) and Figure 2 (integrity verification).*

---

## 6. Fictional-data and instructor-confirmation note

The brief asks for a "real-world dataset". The IBM sample is officially published,
realistically structured and fully traceable, but represents a **fictional** company rather
than observed commercial customer data.

This project describes it accurately throughout. Instructor confirmation was requested and
is recorded as `<<UNRESOLVED: confirmed / not confirmed>>`. See decision D-01 in
`docs/DECISIONS.md`.

---

## 7. Data quality assessment

30 automated checks, all passing:

| Aspect | Finding |
|---|---|
| Rows / columns | 7,043 × 21, exact names and order |
| Duplicate rows | 0 |
| Duplicate `customerID` | 0 |
| Target domain | Exactly {Yes, No} |
| `SeniorCitizen` domain | {0, 1} |
| Missing values | 11, all in `TotalCharges`, all with `tenure = 0` |
| Negative values | None |
| Category domains | All match the IBM data dictionary |
| Observed ranges | tenure 0–72; MonthlyCharges 18.25–118.75; TotalCharges 18.80–8,684.80 |

The 11 blanks belong to customers who have not completed a billing cycle — structurally
meaningful, not random. Handled by median imputation **inside** the pipeline, fitted on the
training split only. The raw file is never modified.

*Insert Figure 4.*

---

## 8. Exploratory data analysis

Describe the method: `src/eda.py` is the single source of truth; the notebook imports the
same functions; every quantitative statement is computed, none hand-entered. 11 figures and
13 tables produced.

*Insert Figures 3, 5, 6a–6d.*

---

## 9. Key observations

| Observation | Evidence |
|---|---|
| Churn rate 26.54% (1,869 of 7,043) | Figure 5 |
| Contract term separates most strongly: 42.71% / 11.27% / 2.83% | Figure 6a |
| Electronic check 45.29% vs automatic credit card 15.24% | Figure 6c |
| Fibre optic 41.89%, DSL 18.96%, none 7.40% | Figure 6b |
| No technical support 41.64% vs 15.17% with it | `reports/tables/churn_rate_by_techsupport.csv` |
| Churners average 17.98 months tenure vs 37.57 retained | Figure 6d |
| tenure–churn correlation −0.352 | `reports/tables/correlation_matrix.csv` |

**Caveats to state explicitly.** These are associations within the sample, not causal
effects. The add-on comparisons are partly confounded because "No internet service" appears
as a level in every add-on column. `tenure` and `TotalCharges` correlate at 0.826, which is
expected and recorded as a known limitation.

---

## 10. Preprocessing

1. Strip whitespace and coerce `TotalCharges` with `errors="coerce"`.
2. Map `Churn`: Yes → 1, No → 0.
3. Exclude `customerID` from the features.
4. Treat `SeniorCitizen` as a binary category, not a magnitude.
5. Numeric branch: median imputation, plus standard scaling for the linear model.
6. Categorical branch: most-frequent imputation, one-hot encoding with
   `handle_unknown="ignore"`.
7. Preprocessing and estimator combined in a single `Pipeline`.

---

## 11. Data-leakage prevention

The control is structural, not procedural:

- The split happens **once**, before any transformer is fitted:
  `train_test_split(test_size=0.20, random_state=42, stratify=y)`.
- Every imputer, encoder and scaler lives **inside** the `Pipeline`, so cross-validation
  refits them from scratch on each fold.
- The held-out set is used **once**, after selection.
- Nothing is imputed or encoded before splitting. Imputing `TotalCharges` beforehand would
  compute the median using test rows — precisely the mistake the design prevents.
- Stratification preserves the class balance: 0.2654 in both splits.

---

## 12. Modelling methodology

- 80/20 stratified split, `random_state=42`.
- Stratified 5-fold cross-validation on the training split only.
- `class_weight="balanced"` on both candidates.
- Selection rule fixed **in code before any test result existed**: CV ROC-AUC → CV F1 → CV
  recall, with a 0.01 tie tolerance.
- Held-out evaluation performed once, after selection.

---

## 13. Models compared

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, balanced class weights, impute + scale + one-hot |
| Random Forest | 400 trees, `min_samples_leaf=5`, balanced class weights, impute + one-hot |
| Dummy (stratified) | Reference point only; never eligible for selection |

---

## 14. Evaluation metrics and business meaning

| Metric | Why it matters here |
|---|---|
| Recall | Share of actual churners identified. A miss means no review happens at all — the costly error |
| Precision | Share of flagged accounts that really churn. Governs wasted specialist time |
| F1 | Balance of the two on the minority class |
| ROC-AUC | Ranking quality independent of threshold — the right measure for a prioritisation tool |
| Accuracy | Reported for completeness; **insufficient alone** on a 26.54% positive rate |

**Why accuracy alone fails.** Always predicting "retained" scores 73.46% while finding
nobody at risk. The dummy baseline makes this concrete: 0.6217 accuracy with 0.2914 recall
and 0.5163 ROC-AUC — no better than chance at ranking.

---

## 15. Performance comparison

| Model | Role | CV ROC-AUC | CV F1 | CV recall | Test ROC-AUC | Test F1 | Test recall | Test precision | Test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **selected** | 0.8460 ± 0.0124 | 0.6286 | 0.8013 | 0.8414 | 0.6130 | 0.7834 | 0.5034 | 0.7374 |
| Random Forest | candidate | 0.8454 ± 0.0091 | 0.6296 | 0.7632 | 0.8417 | 0.6349 | 0.7834 | 0.5337 | 0.7608 |
| Dummy (stratified) | baseline | 0.5065 ± 0.0171 | 0.2762 | 0.2776 | 0.5163 | 0.2903 | 0.2914 | 0.2891 | 0.6217 |

**Confusion matrix — selected model, held-out set (n = 1,409):**

| | Predicted: retained | Predicted: churn |
|---|---:|---:|
| **Actual: retained** | 746 | 289 |
| **Actual: churn** | 81 | 293 |

- 81 of 374 churners missed (21.7%) — unrecoverable.
- 289 of 582 flagged would have stayed (49.7%) — recoverable.
- 582 of 1,409 flagged (41.3%) — a bounded review workload.

*Insert Figures 7, 8, 9, 10.*

---

## 16. Selected-model justification

The candidates tied on cross-validated ROC-AUC (gap 0.0005) and F1 (gap 0.0010), both inside
the 0.01 tolerance, so the pre-declared rule fell through to CV recall, where Logistic
Regression led 0.8013 to 0.7632. Recall breaks the tie because a missed churner costs more
than an unnecessary review.

**Stated openly:** on the held-out set the Random Forest is marginally better on accuracy,
precision and F1, with identical recall and effectively identical ROC-AUC. The model was not
switched, because reselecting after seeing test results would invalidate the held-out
evaluation. See decision D-08.

---

## 17. Saved pipeline and reproducibility

- Complete `Pipeline` — preprocessing and estimator — in `deploy/artifacts/model_pipeline.joblib` (8.4 KB).
- `model_metadata.json`: model name, version, timestamp, seed, dataset blob SHA, metrics,
  confusion matrix, environment versions.
- `feature_schema.json`: predictors, dtypes, observed training categories, numeric bounds,
  target mapping, excluded identifier, display labels, UI grouping.
- `model_card.md`: intended use, out-of-scope uses, provenance, metrics, limitations, ethics,
  human-review requirement.
- Reproducible with `make bootstrap && make validate && make train` at `random_state=42`.
- Verified to load and score in a fresh Python process.

*Insert Figure 11.*

---

## 18. Streamlit application design

- Loads artifacts via `pathlib` relative to `app.py`, cached with `st.cache_resource`, and
  **never trains at startup**.
- 19 inputs in four groups; dependency rules enforced (no phone → `No phone service`; no
  internet → `No internet service`).
- Outputs: predicted class, probability, risk band, cautious review action, model version.
- Token-based light/dark themes with an in-app toggle; charts follow the active theme.
- Three context charts: risk position, segment reference rates, scored-population position.
- No stack trace, path or identifier is ever shown to the user.

*Insert Figures 13, 14, 15, 16, 17, 18, 19, 20, 21.*

---

## 19. Local testing

- **106 automated tests** across the data contract, artifact integrity, prediction behaviour,
  deployment configuration, the analysis modules and the application features — all passing.
- Streamlit smoke test: server starts and answers `/_stcore/health` with HTTP 200.
- Manual testing of both demonstration profiles and both themes.
- Quality gate results recorded in `docs/QUALITY_GATE_RESULTS.md`.

*Insert Figure 12.*

---

## 20. Docker packaging

- `python:3.11-slim`, non-root `appuser`, `/app` working directory, port 7860.
- Only runtime files copied; training code, raw data, notebooks and tests excluded.
- Pinned runtime dependencies matching the training environment.
- Built and run locally: health check passed, and the container reproduced the **identical**
  prediction to the local environment.

*Insert Figure 22.*

---

## 21. GitHub repository and version control

- Repository: https://github.com/krish2105/SDAIM-Customer-Churn (public)
- Default branch: `main`
- Meaningful staged commits, each made only after tests and the secret scan passed.
- Commit history (read from the platform, newest first):

| SHA | Commit |
|---|---|
| `d90bf8b675bb` | Bring documentation level with the implemented work |
| `26660b3fcad0` | Sync test plan and quality-gate records with the expanded suite |
| `27e09b025288` | Implement improvement plan Horizons 1 and 2 |
| `73d38f40324c` | Correct deliverables checklist status after deployment |
| `549dc6a005e4` | Record visible-change deployment test as passed |
| `561ce33b724f` | Release version 1.1.0 with automated deployment verification caption |
| `91d7f8279ac4` | Record verified deployment evidence |
| `591d0d1a0696` | Add C-level improvement implementation plan |
| `9f5100648213` | Complete professional documentation and tests |
| `82f7b3dc74de` | Add CI and Hugging Face deployment workflows |
| `0ec91eb52e05` | Add Streamlit Docker deployment application |
| `999ed6a1e844` | Train and compare churn classification models |
| `f96beea075be` | Add reproducible EDA and data validation |
| `d9dd086c10ef` | Initialize validated churn project structure |

*Insert Figures 23, 24.*

---

## 22. Hugging Face Space configuration

- Space ID: `krish21may/churn`
- SDK: Docker; `app_port: 7860`
- Visibility: Public
- Contents: only the `deploy/` package.

*Insert Figures 30, 31.*

---

## 23. GitHub Actions CI/CD workflow

**`ci.yml`** — checkout with LFS, Python 3.11, install dev dependencies, validate the
dataset with `--strict-sha`, `compileall`, pytest, secret scan, deployment-file and Space
metadata verification.

**`deploy.yml`** — triggered by pushes to `main` touching `deploy/**`, plus
`workflow_dispatch`. A `validate` job the deploy job depends on; `permissions: contents:
read`; a concurrency group; explicit configuration checks; then
`huggingface/hub-sync@v0.2.1` with `repo_type: space`, `space_sdk: docker`,
`subdirectory: deploy`.

The token is read only as `${{ secrets.HF_TOKEN }}` and never printed. A test asserts no
other secret or variable is referenced.

*Insert Figures 25, 26, 27, 28.*

---

## 24. Initial deployment evidence

| Item | Value |
|---|---|
| Workflow run URL | https://github.com/krish2105/SDAIM-Customer-Churn/actions/runs/30020847651 |
| Run ID | `30020847651` (validate + sync jobs both green) |
| Space build result | Succeeded — runtime stage `RUNNING` |
| Live Space URL | https://huggingface.co/spaces/krish21may/churn |
| Live prediction verified | Yes — 22.9%, Low risk band |
| Commit SHA deployed | `591d0d1a069604dbff9daca53c22f946a051589b` |

*Insert Figures 29, 32, 33.*

---

## 25. Visible-change deployment evidence

| Item | Value |
|---|---|
| Change made | Application version 1.0.0 → 1.1.0 plus a visible caption |
| Second run URL | https://github.com/krish2105/SDAIM-Customer-Churn/actions/runs/30023117028 |
| Trigger | **Automatic** — the `deploy/**` path filter, no manual dispatch |
| Space rebuild result | Succeeded |
| Version 1.1.0 visible in the live app | Yes — header badge and caption both read 1.1.0 |
| Predictions still working | Yes — 22.9%, Low risk band |
| Commit SHA | `561ce33b724fc147678e97419115860371cbcd21` |

*Insert Figures 34, 35, 36, 37.*

Procedure: `docs/DEMONSTRATION_SCRIPT.md`.

---

## 25b. Post-deployment analysis — Horizon 1 and 2

Added after the initial deployment. Each item closes a limitation this project had already
published about itself, which is the strongest available framing: *we identified it, then we
fixed it.*

### Fairness audit (`reports/fairness_report.md`)

Optimises for **equal opportunity** — equal recall across groups — because the harm that
matters is an at-risk customer being missed.

| Attribute | Finding |
|---|---|
| `gender` | No disparity exceeded the 0.10 materiality convention on any criterion |
| `SeniorCitizen` | Material on all four criteria — **but** seniors churn at 44.14% against 23.25%, and the model finds at-risk seniors *better* (recall 0.9286 vs 0.7319). The group under-served on the governing criterion is **non-seniors**. The real burden is the false-positive gap (0.5000 vs 0.2492) |

**Counterfactual.** Removing both attributes costs **+0.0008 ROC-AUC** and **+0.0027 recall**
— inside the ±0.0124 CV standard deviation. **Removal is recommended.** A group-specific
threshold was rejected outright as direct differential treatment.

*Insert Figures 38, 39.*

### Calibration (`reports/calibration_report.md`)

| Variant | Brier | ECE | ROC-AUC | Mean predicted |
|---|---:|---:|---:|---:|
| Deployed (uncalibrated) | 0.1688 | 0.1503 | 0.8414 | 0.4157 |
| Isotonic | 0.1388 | 0.0194 | 0.8413 | 0.2671 |
| Sigmoid | 0.1383 | 0.0254 | 0.8416 | 0.2678 |

Observed base rate 0.2654. The model is **over-confident about churn by 0.15** — the
predicted consequence of `class_weight="balanced"`, chosen deliberately to raise recall.
Ranking is unaffected: ROC-AUC moves by 0.0003.

*Insert Figure 40.*

### Threshold analysis (`reports/threshold_analysis.md`)

| Cost ratio | Optimal threshold | Recall | Precision |
|---:|---:|---:|---:|
| 1:1 | 0.73 | 0.5695 | 0.6574 |
| 3:1 | 0.44 | 0.8422 | 0.4817 |
| 5:1 | 0.33 | 0.9171 | 0.4403 |
| 20:1 | 0.12 | 0.9866 | 0.3545 |

The deployed 0.50 is cost-optimal at roughly **3:1** — using it was always an implicit
assertion about relative costs, now made explicit.

*Insert Figures 41, 45.*

### Drift apparatus (`reports/drift_report.md`)

**No real drift can be observed** — the dataset is a single cross-section. The apparatus was
built and validated **both ways**: STABLE on the unshifted holdout (0 of 19 flagged), ALERT
on a simulated acquisition campaign (15 of 19, 7 at alert).

*Insert Figure 42.*

### Explainability, batch scoring, tracking and the retention brief

- **Contributions** are the exact log-odds decomposition, reconstructing the score to float
  precision (asserted on 25 customers). Not SHAP — for a linear model this *is* the model.
- **Batch scoring** ranks 1,000 customers in 0.01 s with a CSV export; nothing written to disk.
- **MLflow** tracks 3 runs with nested per-candidate runs, a registered model and a
  documented manual rollback.
- **The retention brief** confines the LLM to rendering pre-computed values. Structured
  output, prohibited-language filter, provenance labelling, kill switch, disabled by default.

*Insert Figures 43, 44, 46, 47.*

---

## 26. Security controls

- No credential anywhere in the repository.
- `HF_TOKEN` as an Actions secret; `HF_SPACE_ID` as a repository variable.
- Fine-grained token scoped to write to one Space only.
- Secret scanner locally, in CI, and before deployment — printing paths and categories only.
- Container: non-root, slim base, runtime files only, no secrets in the image.
- Application: no stack traces, no user input interpolated into HTML, no external resources,
  XSRF protection enabled, usage stats disabled.
- Known gaps stated: no dependency or image vulnerability scanning, no signed commits.

Full detail: `docs/SECURITY.md`.

---

## 27. Governance and ethical limitations

The model must not autonomously change prices, terminate or modify contracts, deny service,
target customers unfairly, or make any financial or customer-treatment decision. Every
output requires human review before a customer is contacted.

`gender` and `SeniorCitizen` are predictors and **no fairness audit has been performed**.
This is a genuine gap and must be closed before operational use. Risk-driven retention
offers can create differential treatment between customers and need model-risk review.

The model produces no explanation of an individual prediction, so it cannot supply a
customer-facing reason.

---

## 28. Risks and limitations

1. Fictional sample — live performance unknown, revalidation required.
2. Single cross-section, no time dimension — drift cannot be assessed, no monitoring baseline.
3. Class imbalance limits precision at high recall (≈0.50 at 0.78 recall).
4. Threshold 0.50 is a documented default; no cost matrix was supplied.
5. Risk bands are communication aids, not validated thresholds.
6. No fairness audit.
7. Association, never causation.
8. Probability calibration has not been verified.

---

## 29. Future enhancements

1. Fairness audit across demographic attributes.
2. Threshold optimisation once real intervention costs exist.
3. Calibration analysis and, if needed, recalibration.
4. Dependency (`pip-audit`) and container (Trivy) scanning in CI.
5. Batch scoring for a full customer book, not one record at a time.
6. Monitoring for input drift and performance decay once live data exists.
7. Per-prediction explanations, subject to a governance review of how they would be used.
8. Model registry and versioned rollback.

---

## 30. Conclusion

The project delivers a complete, reproducible pipeline from a provenance-verified dataset to
an automatically deployed application. The selected model reaches 0.8414 ROC-AUC and 0.7834
recall on held-out data, identifying 293 of 374 churners.

Equally important is what it does not do. It makes no financial claim, automates no
customer-treatment decision, and states its limitations — including the absent fairness
audit — in the application itself as well as the report. It demonstrates technical
feasibility and may support prioritisation for human review.

---

## 31. References

1. IBM. *Telco Customer Churn sample.* https://github.com/IBM/telco-customer-churn-on-icp4d (Apache-2.0)
2. IBM Documentation. *Samples: Telco customer churn.* https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. Streamlit documentation. https://docs.streamlit.io
5. Hugging Face Spaces — Docker SDK. https://huggingface.co/docs/hub/spaces-sdks-docker
6. GitHub Actions documentation. https://docs.github.com/actions
7. `huggingface/hub-sync` action. https://github.com/huggingface/hub-sync
8. Mitchell, M. et al. (2019). Model Cards for Model Reporting. *FAT\* '19*, 220–229.
9. W3C. *Web Content Accessibility Guidelines (WCAG) 2.1.* https://www.w3.org/TR/WCAG21/

---

## 32. Team contribution table

| Member | Student ID | Primary contribution | Sign-off |
|---|---|---|---|
| Krishna Mathur | `AS25DXB018` | Platform, deployment, CI/CD and security | |
| Yash Petkar | `AS25DXB021` | Modelling rigour: fairness, calibration and tracking | |
| Atharva Soundankar | `AS25DXB020` | Application, explainability and batch scoring | |

All three members contributed to the report, the evidence pack and the demonstration.

---

## 33. Appendices

- **A.** Repository structure (README §6)
- **B.** Full model card (`deploy/artifacts/model_card.md`)
- **C.** EDA observations (`reports/eda_observations.md`)
- **D.** Executive model summary (`reports/executive_model_summary.md`)
- **E.** Design decisions (`docs/DECISIONS.md`)
- **F.** Test plan (`docs/TEST_PLAN.md`)
- **G.** Security documentation (`docs/SECURITY.md`)
- **H.** Quality gate results (`docs/QUALITY_GATE_RESULTS.md`)
- **I.** Troubleshooting guide (`docs/TROUBLESHOOTING.md`)
- **J.** Workflow files (`.github/workflows/`)

---

## Pre-submission checklist

- [ ] Every `<<UNRESOLVED>>` replaced with a verified value
- [ ] Every URL opened and confirmed working
- [ ] Every commit SHA, run ID and timestamp copied from the real platform
- [ ] Every screenshot genuine, and checked for credentials before inclusion
- [ ] No claim of measured revenue, churn reduction or ROI anywhere
- [ ] Dataset described as fictional in every place it appears
- [ ] Team contribution table completed and signed
- [ ] `docs/DELIVERABLES_CHECKLIST.md` fully ticked
