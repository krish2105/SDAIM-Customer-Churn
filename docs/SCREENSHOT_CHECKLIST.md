# Screenshot Checklist

Every screenshot the report needs, with a figure number, a caption you can paste, and a
statement of what it proves.

**Rules for all screenshots**

1. Capture the **real** screen. Never mock one up, crop misleadingly, or reuse a screenshot
   from a different run.
2. **Before capturing, check the frame for credentials.** Token values, private URLs, email
   addresses, browser bookmarks, other tabs, notification popups. A secret in a screenshot
   is a secret in your submission.
3. Include enough context to identify the source — the URL bar for web pages, the command
   for terminal output.
4. Capture at a readable resolution. Full-window beats full-screen.
5. Save to `docs/screenshots/` using the figure number, e.g. `fig-07-model-comparison.png`.
   (`.gitignore` excludes a top-level `screenshots/` directory but **not** `docs/screenshots/`,
   so report evidence is versioned deliberately.)

---

## A. Dataset and EDA

### Figure 1 — Official dataset source
**Capture.** The IBM repository page at
`https://github.com/IBM/telco-customer-churn-on-icp4d`, URL bar visible.
**Caption.** *Figure 1: Official IBM repository for the Telco Customer Churn sample, showing the Apache-2.0 licence and the published data file.*
**Proves.** The dataset provenance is official and traceable, not an anonymous copy.

### Figure 2 — Dataset integrity verification
**Capture.** Terminal running:
```bash
git hash-object data/raw/Telco-Customer-Churn.csv
make validate
```
**Caption.** *Figure 2: Dataset integrity verification. The Git blob SHA matches the expected value `3de7a612d1609f25f21a455bda77948729369002`, and all 30 contract checks pass.*
**Proves.** The exact official file revision was used — the strongest possible provenance claim.

### Figure 3 — Shape and data types
**Capture.** Notebook cells showing `frame.shape` and the dtypes table.
**Caption.** *Figure 3: Dataset shape (7,043 × 21) and column data types after the documented type corrections.*
**Proves.** The data matches the documented contract.

### Figure 4 — Missing-value analysis
**Capture.** The notebook cells showing the missing-value summary and the `tenure = 0` finding, or `reports/figures/02_missing_values.png`.
**Caption.** *Figure 4: Missing values after coercion — 11 rows, all in `TotalCharges`, all with `tenure = 0`. The blanks are structurally meaningful rather than random.*
**Proves.** Data quality was investigated, not merely patched.

### Figure 5 — Target distribution
**Capture.** `reports/figures/01_target_distribution.png`.
**Caption.** *Figure 5: Target class distribution — 1,869 churned (26.54%) against 5,174 retained. The imbalance is why accuracy alone is not an acceptable selection metric.*
**Proves.** The imbalance that justifies the metric strategy.

### Figure 6 — Key EDA charts
**Capture.** `06_churn_rate_by_contract.png`, `07_churn_rate_by_internetservice.png`, `08_churn_rate_by_paymentmethod.png`, `03_tenure_by_churn.png`. Combine as 6a–6d.
**Caption.** *Figure 6: Churn rate by contract term (42.71% month-to-month against 2.83% two-year), internet service, payment method, and tenure distribution by churn status.*
**Proves.** Substantive exploratory analysis with clear, honest findings.

---

## B. Modelling

### Figure 7 — Model comparison
**Capture.** The `make train` terminal output showing both CV and held-out sections, or `reports/model_comparison.csv` opened.
**Caption.** *Figure 7: Cross-validated and held-out comparison of Logistic Regression, Random Forest and the stratified baseline. Selection used cross-validation only; the test set was scored once, afterwards.*
**Proves.** Two substantive models were genuinely compared against a baseline.

### Figure 8 — Confusion matrix
**Capture.** `reports/figures/12_confusion_matrix_logistic_regression.png`.
**Caption.** *Figure 8: Confusion matrix for the selected model on the held-out test set — 293 true positives, 81 false negatives, 289 false positives, 746 true negatives.*
**Proves.** The error profile behind the business interpretation.

### Figure 9 — ROC curves
**Capture.** `reports/figures/13_roc_curves.png`.
**Caption.** *Figure 9: ROC curves on the held-out test set. Both candidates reach ≈0.841 AUC; the stratified baseline sits on the diagonal at 0.516.*
**Proves.** Both models rank far better than chance, and the baseline does not.

### Figure 10 — Classification report
**Capture.** `reports/tables/classification_report_logistic_regression.json`, or the terminal metrics line.
**Caption.** *Figure 10: Per-class classification report for the selected model — recall 0.7834 and precision 0.5034 on the churn class.*
**Proves.** Per-class performance, not just headline accuracy.

### Figure 11 — Saved artifacts
**Capture.** `ls -lh deploy/artifacts/`.
**Caption.** *Figure 11: Exported artifacts — the complete pipeline (8.4 KB), metadata, feature schema, reference rates and model card.*
**Proves.** The complete pipeline was saved, and the size shows Git LFS is unnecessary.

### Figure 12 — Automated test results
**Capture.** Terminal running `make test`, showing `52 passed`.
**Caption.** *Figure 12: Full automated test suite — 52 tests covering the data contract, artifact integrity, prediction behaviour and deployment configuration.*
**Proves.** The project is verified automatically, not just by inspection.

---

## C. Local application

### Figure 13 — Application home screen
**Capture.** `make app` → `http://localhost:8501`, top of the page: header, badges, disclaimer, KPI cards.
**Caption.** *Figure 13: Application home screen showing the business purpose, the deployed model and version, the fictional-data disclaimer, and measured held-out performance.*
**Proves.** A professional interface that states its provenance and limits up front.

### Figure 14 — Complete input form
**Capture.** All four input sections. Scroll or stitch if needed.
**Caption.** *Figure 14: Input controls for all 19 predictors, grouped into customer profile, services, contract and billing, and charges and tenure.*
**Proves.** Every model predictor is exposed, coherently organised.

### Figure 15 — Dependency enforcement
**Capture.** Set **Internet service** to `No`; capture the locked add-ons and the caption beneath them.
**Caption.** *Figure 15: Logical dependency enforcement — with no internet service, every add-on is locked to "No internet service", the only combination present in the training data.*
**Proves.** The form cannot produce a combination the model never saw.

### Figure 16 — Low-probability example
**Capture.** Load "Case A — established two-year contract" and generate. Capture the result card.
**Caption.** *Figure 16: Assessment for a long-tenure two-year-contract profile. The model returns a low churn probability and a Low risk band; no immediate action is indicated.*
**Proves.** The model discriminates, and the output is cautious.
**Record the actual probability shown — do not predict it in advance.**

### Figure 17 — High-probability example
**Capture.** Load "Case B — new month-to-month fibre customer" and generate.
**Caption.** *Figure 17: Assessment for a recently acquired month-to-month fibre profile, returning a materially higher probability and a suggested review action.*
**Proves.** Different inputs produce meaningfully different assessments.
**Record the actual probability shown.**

### Figure 18 — Probability, risk tier and disclaimer
**Capture.** The result card plus the caption stating the bands are not validated thresholds.
**Caption.** *Figure 18: Output detail — probability, predicted class, decision threshold, risk band, suggested review action, and the statement that risk bands are communication aids rather than validated business thresholds.*
**Proves.** Uncertainty and governance are communicated, not buried.

### Figure 19 — Context charts
**Capture.** All three tabs: Risk position, Segment context, Scored population.
**Caption.** *Figure 19: Decision-support context — the score within the risk bands, training-sample churn rates for the selected segments, and the score's position within the held-out test distribution.*
**Proves.** The interface supports interpretation rather than presenting a bare number.

### Figure 20 — Light and dark themes
**Capture.** The same screen in both modes, side by side.
**Caption.** *Figure 20: Token-based light and dark themes with an in-app toggle. Charts follow the active theme; both palettes were checked against WCAG AA contrast.*
**Proves.** Deliberate, accessible interface design.

### Figure 21 — Model information and limitations
**Capture.** The expanded "Model information and limitations" section.
**Caption.** *Figure 21: In-application disclosure of metrics, the selection rule, limitations including the absent fairness audit, and the governance constraints.*
**Proves.** Limitations are disclosed to the user, not only in the report.

### Figure 22 — Local Docker run
**Capture.** `make docker-run` output showing the health check passing, plus the app at `http://localhost:7860`.
**Caption.** *Figure 22: Deployment image built and run locally. The container answers `/_stcore/health` with HTTP 200 and reproduces the same prediction as the local environment.*
**Proves.** The container is genuinely runnable before it reaches the Space.

---

## D. GitHub

### Figure 23 — Repository structure
**Capture.** The repository root file listing on GitHub, URL bar visible.
**Caption.** *Figure 23: Repository structure — source, deployment package, tests, documentation and workflows.*
**Proves.** The project is organised and version-controlled.

### Figure 24 — Commit history
**Capture.** The commits page showing the staged, meaningful commits.
**Caption.** *Figure 24: Commit history showing incremental development, each commit made only after tests and the secret scan passed.*
**Proves.** Genuine version-control practice, not a single dump commit.

### Figure 25 — CI workflow file
**Capture.** `.github/workflows/ci.yml` rendered on GitHub.
**Caption.** *Figure 25: Continuous-integration workflow — dataset validation, syntax compilation, tests, secret scan and deployment-file verification.*
**Proves.** Automated quality control on every push and pull request.

### Figure 26 — Deployment workflow file
**Capture.** `.github/workflows/deploy.yml` rendered on GitHub.
**Caption.** *Figure 26: Deployment workflow — a validation job the deployment depends on, least-privilege permissions, a concurrency group, explicit configuration checks, and the Space sync.*
**Proves.** Deployment is gated on validation and follows least privilege.

### Figure 27 — `HF_TOKEN` secret (name only)
**Capture.** Settings → Secrets and variables → Actions → Secrets, showing the **name** `HF_TOKEN`.
**Caption.** *Figure 27: The Hugging Face token stored as an encrypted GitHub Actions secret. Only the name is visible; the value cannot be displayed after creation.*
**Proves.** The token is stored correctly and is not in the code.
**⚠ Never capture the token value. If you ever see it on screen, do not screenshot it.**

### Figure 28 — `HF_SPACE_ID` variable
**Capture.** The Variables tab showing `HF_SPACE_ID` and its value.
**Caption.** *Figure 28: The target Space configured as a repository variable. It is a public identifier, so it is a variable rather than a secret.*
**Proves.** The secret/variable distinction was applied deliberately.

### Figure 29 — First successful workflow run
**Capture.** The completed Actions run, all steps green, run number and timestamp visible.
**Caption.** *Figure 29: First successful automated deployment run, completed at `<<UNRESOLVED: timestamp>>` for commit `<<UNRESOLVED: SHA>>`.*
**Proves.** The pipeline runs end to end on the real platform.
**Record the real run URL, run ID, commit SHA and timestamp.**

---

## E. Hugging Face

### Figure 30 — Space settings
**Capture.** The Space settings or file listing, showing the Docker SDK.
**Caption.** *Figure 30: Hugging Face Space configured with the Docker SDK, containing only the deployment package.*
**Proves.** The Space is configured as designed, with no training code or raw data.

### Figure 31 — Successful Docker build
**Capture.** The Space Build logs ending in a successful build.
**Caption.** *Figure 31: Successful Space container build from `deploy/Dockerfile`.*
**Proves.** The image builds on the platform, not only locally.

### Figure 32 — Live application
**Capture.** The running Space, `huggingface.co` URL visible.
**Caption.** *Figure 32: The deployed application running publicly at `<<UNRESOLVED: Space URL>>`.*
**Proves.** The deployment is real and reachable.

### Figure 33 — Live prediction
**Capture.** A completed assessment in the deployed app.
**Caption.** *Figure 33: A live prediction produced by the deployed application, returning a probability, a risk band and a suggested review action.*
**Proves.** The deployed model loads and scores correctly in production.

---

## F. Visible-change deployment test

### Figure 34 — Version-change commit
**Capture.** The commit diff raising the version to 1.1.0 and adding the caption.
**Caption.** *Figure 34: Visible change committed for the automated-deployment test — application version raised to 1.1.0 with a visible caption.*
**Proves.** A deliberate, traceable change was made.

### Figure 35 — Second successful workflow run
**Capture.** The second Actions run, green, showing it is a distinct run.
**Caption.** *Figure 35: Second automated deployment run triggered by the version change, completed at `<<UNRESOLVED: timestamp>>`.*
**Proves.** Deployment automation works repeatedly, not once by luck.

### Figure 36 — Version 1.1.0 live
**Capture.** The deployed app showing version 1.1.0 and the new caption.
**Caption.** *Figure 36: The deployed application showing version 1.1.0 after automated redeployment, confirming the change reached production without manual intervention.*
**Proves.** **The core requirement of Part 4 of the brief** — a pushed change deploys automatically.

### Figure 37 — Predictions still working after redeployment
**Capture.** A prediction in the updated app, version 1.1.0 visible in the same frame.
**Caption.** *Figure 37: Predictions functioning correctly in version 1.1.0 after automated redeployment.*
**Proves.** The redeployment did not break the application.

---

---

## G. Horizon 1 and 2 evidence

These cover the improvement work. All are capturable now from `reports/` and the live Space.

### Figure 38 — Fairness audit
**Capture.** `reports/figures/15_fairness_seniorcitizen.png` and `15_fairness_gender.png`.
**Caption.** *Figure 38: Fairness criteria by subgroup on the held-out test set. No material disparity on `gender`. A material disparity on `SeniorCitizen`, shown alongside the group base rates that make the three criteria mutually exclusive.*
**Proves.** The audit was performed and reported per criterion, not summarised into a single verdict.

### Figure 39 — The counterfactual that changed the recommendation
**Capture.** The "Decision" section of `reports/fairness_report.md`.
**Caption.** *Figure 39: Cost of removing the protected attributes — +0.0008 ROC-AUC, inside the cross-validation standard deviation. The keep-or-remove decision was made after measuring, and removal is now recommended.*
**Proves.** An evidence-based governance decision rather than an asserted one. **Strong viva material.**

### Figure 40 — Calibration
**Capture.** `reports/figures/16_calibration_curve.png`.
**Caption.** *Figure 40: Reliability diagram. The deployed model is over-confident about churn (mean predicted 0.4157 against a 0.2654 base rate, ECE 0.1503) — the predicted consequence of balanced class weights. Isotonic calibration reduces ECE to 0.0194 while leaving ROC-AUC unchanged.*
**Proves.** Probabilities were validated as probabilities, and the cause of the miscalibration is understood rather than merely observed.

### Figure 41 — Threshold sensitivity
**Capture.** `reports/figures/17_threshold_analysis.png`.
**Caption.** *Figure 41: Threshold trade-off and the cost-minimising threshold against the miss-to-review cost ratio. The deployed 0.50 is optimal at roughly 3:1; the business supplies the ratio, the model supplies the curve.*
**Proves.** The threshold assumption is quantified and exposed rather than defaulted to silently.

### Figure 42 — Drift detector validated both ways
**Capture.** `reports/figures/18_drift_detection.png` plus the `make drift` terminal output showing both control and shifted status.
**Caption.** *Figure 42: Drift detection. The detector reports STABLE on the unshifted holdout (0 of 19 features flagged) and ALERT on a simulated acquisition campaign (15 of 19, 7 at alert).*
**Proves.** The apparatus works in both directions. A detector never shown to fire — or one that always fires — is not evidence of anything.

### Figure 43 — Per-prediction contributions
**Capture.** The "Why this score" tab in the live application.
**Caption.** *Figure 43: Exact log-odds contribution breakdown for a single prediction, with the causal-language disclaimer. For a linear model this decomposition is the model, not an approximation of it.*
**Proves.** A specialist can see what moved the score, without the interface implying causation.

### Figure 44 — Batch scoring work queue
**Capture.** The "Batch scoring" tab after uploading `data/raw/Telco-Customer-Churn.csv` (first 1,000 rows).
**Caption.** *Figure 44: Retention work queue — customers ranked by estimated churn risk with risk bands and CSV export. Uploaded data is scored in memory and never written to disk.*
**Proves.** The tool prioritises a book of accounts, not just one record.

### Figure 45 — Adjustable decision threshold
**Capture.** The sidebar slider at 0.50 and again at a different value, with the result changing.
**Caption.** *Figure 45: The decision threshold exposed as an adjustable control, labelled as an assumption rather than an optimum.*
**Proves.** The assumption is visible to the user, not buried in a report.

### Figure 46 — MLflow tracked runs and registry
**Capture.** `make mlflow-ui` → http://127.0.0.1:5000, showing the runs list and the registered model.
**Caption.** *Figure 46: MLflow experiment tracking with nested per-candidate runs and the registered model carrying the `production` alias.*
**Proves.** Comparability across runs and a rollback path — what a seed alone cannot provide.

### Figure 47 — Retention brief with provenance
**Capture.** The "Retention review brief" expander in the live app.
**Caption.** *Figure 47: Retention brief. AI generation is disabled by default, so the deterministic template is shown with its provenance stated explicitly.*
**Proves.** The GenAI layer is governed: labelled, optional, and safe by default.

---

## Completion tracker

| Range | Figures | Status |
|---|---|---|
| Dataset and EDA | 1–6 | Reproducible now from `reports/` |
| Modelling | 7–12 | Reproducible now |
| Local application | 13–22 | Reproducible now (app and container both run) |
| GitHub | 23–29 | Capturable now — repo public, runs green |
| Hugging Face | 30–33 | Capturable now — Space live |
| Visible-change test | 34–37 | Capturable now — the test has been performed |
| Horizon 1 and 2 evidence | 38–47 | Capturable now |

**All 47 figures are capturable now.** The repository, both workflow runs, the Space and the
visible-change test are all complete and public; the analysis figures are in `reports/`.
