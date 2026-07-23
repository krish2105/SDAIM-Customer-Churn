# Customer Churn Intelligence — Improvement Implementation Plan

**Prepared for:** Chief Customer Officer, Chief Data & Analytics Officer, Model Risk & Governance
**Prepared by:** Krishna Mathur (AS25DXB018) · Yash Petkar (AS25DXB021) · Atharva Soundankar (AS25DXB020)
**Module:** SDAIM (Term 3) · **Instructor:** JP Aggarwal
**Baseline version:** 1.0.0 → **delivered at 1.1.0**
**Status:** Horizons 1 and 2 **complete**. Horizon 3 documented and deliberately not scheduled.

---

## 1. Executive summary

Version 1.0.0 is delivered, verified and deployed to source control. It does what it claims:
it scores a customer record, returns a calibrated-in-name-only probability, a risk band and a
cautious review action, and it redeploys automatically. Ten of ten local quality gates pass
and the first CI run on GitHub succeeded.

That is a working prototype, not a decision system a retention function could adopt. Three
things stand between the two, and all three are already documented as limitations rather than
discovered late:

1. **The numbers are not yet decision-grade.** Probabilities are unvalidated, the 0.50
   threshold is arbitrary, and no fairness audit exists despite `gender` and `SeniorCitizen`
   being predictors. A governance reviewer would stop here.
2. **There is no operational memory.** No experiment tracking, no model registry, no drift
   detection, no retraining trigger. A second model version cannot currently be compared to
   the first except by reading a CSV.
3. **The unit of work is one customer at a time.** A retention manager needs a ranked book of
   accounts, not a form.

This plan closed those three gaps across three horizons. **Horizon 1 is what earns marks.
Horizon 2 is what a hiring manager sees. Horizon 3 is what a telco would actually need.**

**Horizons 1 and 2 have been implemented and are in the repository.** Every item below is
marked with its delivered state and the measured result. Horizon 3 remains documented and
unscheduled, which is a deliberate decision rather than an omission.

**Direct cost: £0.** Every component is open-source or covered by the existing Hugging Face
Pro subscription. The LLM layer is capped by design and ships disabled.

### Headline outcomes

| Delivered | Measured result |
|---|---|
| Fairness audit (H1-1) | No material disparity on `gender`. Material disparity on `SeniorCitizen`, driven largely by a genuine base-rate gap (0.4414 vs 0.2325). **Removing both protected attributes costs only +0.0008 ROC-AUC** — removal now recommended. |
| Calibration (H1-2) | Model is **over-confident by 0.15** (mean predicted 0.4157 vs base rate 0.2654); ECE 0.1503. Isotonic calibration cuts ECE to 0.0194 with ROC-AUC unchanged. |
| Threshold analysis (H1-3) | Optimal threshold published as a curve against cost ratio: 0.73 at 1:1 down to 0.12 at 20:1. The deployed 0.50 is optimal at roughly 3:1. |
| Explainability (H1-4) | Exact log-odds decomposition — reconstructs the model to float precision, asserted by test on 25 customers. |
| Deployment (H1-5) | Live at <https://krish21may-churn.hf.space>, with the visible-change test passed. |
| MLflow (H2-1) | 3 runs tracked, model registered with a `production` alias and a documented rollback. |
| Drift apparatus (H2-2) | **Validated both ways**: stable on unshifted holdout (0/19 flagged), alert on a simulated campaign (15/19 flagged, 7 at alert). |
| Batch scoring (H2-3) | 1,000 customers ranked in **0.01 s**; batch and single-record paths asserted identical. |
| GenAI brief (H2-4) | Guardrailed, provenance-labelled, **disabled by default**, deterministic fallback that passes its own language checks. |
| Test suite | **52 → 106 tests**, all passing. |

---

## 2. Current state — verified baseline

Stated from measured evidence, not aspiration.

| Dimension | Status | Evidence |
|---|---|---|
| Data integrity | **Strong** | Git blob SHA verified; 30 contract checks; raw file immutable |
| Model performance | **Adequate** | Test ROC-AUC 0.8414, recall 0.7834, precision 0.5034, F1 0.6130 |
| Methodology | **Strong** | Single pre-fit split; all transformers in-pipeline; selection rule fixed in code before test evaluation |
| Reproducibility | **Strong** | `random_state=42`; container reproduces the local prediction exactly (22.9% both) |
| Test coverage | **Good** | 52 tests across data contract, artifact, prediction, deployment config |
| CI/CD | **Working** | First CI run on GitHub: **success**. Deploy workflow failed at the config gate by design and correctly skipped the sync |
| Security | **Good** | No credentials in repo; scanner verified against synthetic credentials |
| **Calibration** | **Absent** | Probabilities are displayed to 1 decimal place and have never been validated |
| **Threshold** | **Arbitrary** | 0.50 default; no cost matrix |
| **Fairness** | **Absent** | `gender`, `SeniorCitizen` are predictors; no subgroup analysis |
| **Explainability** | **Absent** | No per-prediction reasoning |
| **Experiment tracking** | **Absent** | Results live in a CSV |
| **Model registry** | **Absent** | One artifact, overwritten in place |
| **Monitoring / drift** | **Absent** | No baseline, no time dimension in the data |
| **Batch scoring** | **Absent** | One record per interaction |

### Deployment status

| Item | State |
|---|---|
| GitHub repo | `krish2105/SDAIM-Customer-Churn` — public, 6 commits pushed, CI green |
| Hugging Face account | `krish21may` (Pro) |
| Space for this project | **Not yet created** — this is the critical path blocker |
| `HF_TOKEN` secret | Not set |
| `HF_SPACE_ID` variable | Not set |

**Note the username asymmetry:** GitHub is `krish2105`, Hugging Face is `krish21may`. The
`HF_SPACE_ID` must use the *Hugging Face* handle.

---

## 3. Gap analysis — what a reviewer would challenge

| # | Gap | Who challenges it | Severity | Horizon |
|---|---|---|---|---|
| G1 | No fairness audit on protected-adjacent attributes | Governance, examiner, ethics | **Critical** | H1 |
| G2 | Probabilities never validated as probabilities | Model risk, examiner | **Critical** | H1 |
| G3 | Threshold not tied to any business cost | CCO, examiner | **High** | H1 |
| G4 | No per-prediction explanation | Retention specialist, regulator | **High** | H1 |
| G5 | No experiment tracking or model registry | Data science lead | **High** | H2 |
| G6 | No drift detection or retraining trigger | Operations | **Medium** | H2 |
| G7 | Single-record scoring only | Retention manager | **Medium** | H2 |
| G8 | No live monitoring or alerting | Operations | **Medium** | H3 |
| G9 | No formal model-risk documentation pack | Compliance | **Medium** | H3 |
| G10 | Static dataset; no feedback loop from outcomes | CDAO | **Low (blocked)** | H3 |

G10 is blocked by the dataset itself — a fictional cross-section with no time dimension. It is
listed for completeness and explicitly deferred.

---

## 4. Horizon 1 — Decision-grade rigour (Weeks 1–2)

**Objective:** make the numbers defensible. Every item here closes a limitation this project
has already published about itself, which is the strongest possible framing in a viva: *we
identified it, then we fixed it.*

### H1-1 · Fairness audit — **do this first**

| | |
|---|---|
| **Owner** | Member A (Modelling) |
| **Effort** | 8 h |
| **Deliverable** | `src/fairness.py`, `reports/fairness_report.md`, 2 figures |
| **Risk if skipped** | The single most likely viva challenge. Currently unanswerable. |

**Approach.** Compute per-subgroup confusion matrices and metrics for `gender` (Female/Male)
and `SeniorCitizen` (0/1) on the held-out test set, then report the three standard fairness
criteria and state which one you are optimising for and why:

- **Demographic parity** — equal flag rates across groups
- **Equal opportunity** — equal *recall* across groups (usually the right one here: it asks
  whether at-risk customers are found equally well regardless of group)
- **Predictive parity** — equal *precision* across groups

Implement with plain `sklearn.metrics` grouped by attribute. Do **not** add Fairlearn or AIF360
for two attributes with two levels each — the dependency cost exceeds the benefit and you
would be unable to explain the internals in a viva.

**Decision to make and document:** if a material disparity appears, choose one of — (a) remove
the attribute and re-measure the performance cost, (b) keep it and document the disparity with
a mitigation, (c) apply a group-specific threshold. **Option (c) is legally hazardous** (it is
differential treatment by protected characteristic) and should be recommended against
explicitly. Recording *why you rejected it* is worth as much as the audit.

**Acceptance criteria.** Subgroup metrics table in the report; a stated fairness criterion with
justification; a documented decision with its performance cost quantified; the model card
updated to replace "no fairness audit has been carried out" with the actual finding.

---

### H1-2 · Probability calibration

| | |
|---|---|
| **Owner** | Member A (Modelling) |
| **Effort** | 6 h |
| **Deliverable** | Calibration curve figure, Brier score, ECE, `reports/calibration_report.md` |
| **Risk if skipped** | The app shows "22.9%" to a decision-maker. If that is not a real 22.9%, the interface is misleading. |

**Approach.** Plot a reliability diagram (`sklearn.calibration.calibration_curve`) on the
held-out set, report the **Brier score** and **Expected Calibration Error**, then decide
whether to wrap the model in `CalibratedClassifierCV`.

**Important nuance to state explicitly:** logistic regression is usually *already*
well-calibrated — it optimises log-loss directly — but you applied `class_weight="balanced"`,
which deliberately distorts the predicted probabilities upward for the minority class. So this
model is **probably mis-calibrated, and predictably so.** Measuring that and explaining *why*
is a stronger result than a flat calibration curve would have been.

If calibration is poor, fit `CalibratedClassifierCV(method="isotonic", cv=5)` on the training
split only and compare Brier scores before/after. **Do not calibrate on the test set.**

**Acceptance criteria.** Reliability diagram before (and after, if calibrated); Brier and ECE
reported; a stated decision; if calibrated, the artifact regenerated and all 52 tests still
passing.

---

### H1-3 · Cost-sensitive threshold optimisation

| | |
|---|---|
| **Owner** | Member B (Business analysis) |
| **Effort** | 6 h |
| **Deliverable** | Cost-curve figure, `reports/threshold_analysis.md`, threshold control in the app |
| **Risk if skipped** | "Why 0.50?" currently has no answer beyond "it is the default". |

**Approach.** You cannot invent real costs — and must not. Instead, build a **parameterised
cost model** and sweep it. Define:

```
Cost(threshold) = FN(threshold) × C_miss + FP(threshold) × C_review
```

Sweep the ratio `C_miss / C_review` from 1:1 to 20:1 and plot the cost-minimising threshold
against that ratio. Then state: *"at a 5:1 cost ratio the optimal threshold is X; at 10:1 it is
Y. The business must supply the ratio; the model supplies the curve."*

This is the professionally correct answer to an unanswerable question, and it is far stronger
than picking a number. Add a threshold slider to the app (defaulting to 0.50) so the
sensitivity is demonstrable live.

**Acceptance criteria.** Cost curve figure; a table of optimal threshold per cost ratio; the
app exposes an adjustable threshold with the risk bands recomputing; documentation states the
ratio is a business input, not a modelling output.

---

### H1-4 · Per-prediction explainability

| | |
|---|---|
| **Owner** | Member C (Application) |
| **Effort** | 10 h |
| **Deliverable** | Contribution chart per prediction, `deploy/explain.py` |
| **Risk if skipped** | A retention specialist cannot act on a bare number. |

**Approach — deliberately not SHAP.** For a logistic regression the contribution of each
feature to the log-odds is **exactly** `coefficient × scaled_value`. This is not an
approximation of the model; it *is* the model. It is faster, has zero extra dependencies, and
is defensible under questioning in a way a SHAP plot often is not.

Extract coefficients from the fitted pipeline, map them back through the one-hot encoder's
`get_feature_names_out()`, and render the top 5 positive and top 5 negative contributors as a
horizontal bar chart in the existing theme.

**Guardrails that must ship with it:** label the chart *"contribution to this model's score"*,
never *"reasons this customer will churn"*. Add a caption stating these are associations within
the training sample, not causes, and that changing a feature would not necessarily change the
customer's behaviour.

If the Random Forest is later selected, switch to permutation importance rather than SHAP for
the same reasons.

**Acceptance criteria.** Contribution chart renders for every prediction in both themes;
contributions sum (in log-odds space) to the model output minus the intercept — assert this in
a test; causal-language guardrail visible on screen.

---

### H1-5 · Hugging Face Space deployment — **critical path**

| | |
|---|---|
| **Owner** | Member D (Platform) |
| **Effort** | 3 h |
| **Deliverable** | Live Space, working automated deployment, visible-change proof |
| **Risk if skipped** | **Parts 3 and 4 of the brief cannot be marked.** Everything else is worth less. |

This is the highest-value three hours in the entire plan. Full runbook in §8.

---

### Horizon 1 summary

| Item | Owner | Hours | Closes |
|---|---|---:|---|
| H1-1 Fairness audit | A | 8 | G1 |
| H1-2 Calibration | A | 6 | G2 |
| H1-3 Threshold optimisation | B | 6 | G3 |
| H1-4 Explainability | C | 10 | G4 |
| H1-5 Space deployment | D | 3 | Brief Parts 3–4 |
| Report and evidence capture | All | 12 | — |
| **Total** | | **45 h** | |

---

## 5. Horizon 2 — MLOps maturity and platform (Weeks 3–4)

**Objective:** demonstrate that this is engineering, not a notebook. This is the horizon a
hiring manager reads.

### H2-1 · MLflow experiment tracking and model registry

| | |
|---|---|
| **Owner** | Member A |
| **Effort** | 10 h |
| **Deliverable** | `mlruns/` tracked experiments, registry with staged versions |

**Approach.** Add `mlflow` to dev requirements only — **not** to the runtime image. Wrap
`run_training()` to log parameters, metrics, the confusion matrix, figures and the dataset blob
SHA per run. Register the selected model with a version and a stage (`Staging` → `Production`).

**Deliberate scope limit:** run MLflow with the local file backend. Do **not** stand up a
tracking server, a database or a cloud backend for a four-week academic project — the
demonstration value is identical and the operational cost is not.

**The point to make in the report:** the current project already achieves reproducibility
through seeds and pinned versions. MLflow adds *comparability across runs* and *rollback*,
which seeds alone cannot give you. Be precise about which problem it solves; claiming it
"adds reproducibility" would be wrong.

**Acceptance criteria.** Three or more runs logged and comparable in the MLflow UI; the
selected model registered with a version; a documented rollback procedure; runtime image size
unchanged.

---

### H2-2 · Drift detection baseline

| | |
|---|---|
| **Owner** | Member B |
| **Effort** | 10 h |
| **Deliverable** | `src/drift.py`, drift report, CI check |

**Honesty constraint that shapes this task:** the dataset is a single cross-section with no
time dimension. **You cannot observe real drift.** Any claim to be "monitoring drift" would be
false. What you *can* do — and should frame explicitly as such — is build the **apparatus** and
prove it works on a simulated shift.

**Approach.** Compute and persist a training-split baseline profile (per-feature distributions,
category frequencies, prediction distribution). Implement two detectors:

- **Numeric drift:** Population Stability Index, with the conventional 0.1 / 0.25 warning and
  alert bands. State that these thresholds are industry convention, not derived here.
- **Categorical drift:** chi-squared test against baseline frequencies.

Then **prove it fires:** construct a deliberately shifted sample (e.g. resample to 80%
month-to-month contracts) and show the detector alerting. A detector never shown to fire is not
evidence of anything — the same standard already applied to the secret scanner in v1.0.0.

Evaluate `evidently` before hand-rolling. If it adds a heavy dependency tree for two detectors
you can implement in 60 lines, hand-roll and document the decision.

**Acceptance criteria.** Baseline profile artifact; PSI and chi-squared implemented; a test
asserting the detector fires on a synthetic shift and stays silent on a resample of the
training distribution; documentation stating no real drift has been observed.

---

### H2-3 · Batch scoring and a retention work-queue

| | |
|---|---|
| **Owner** | Member C |
| **Effort** | 12 h |
| **Deliverable** | CSV upload → ranked, exportable work queue in the app |

**Approach.** This is the highest business-value feature in the plan and the one that most
changes what the tool *is*: from a calculator into a prioritisation instrument.

Add a second page: upload a CSV, validate it against `feature_schema.json` (reusing the
existing contract — reject with a clear per-column error rather than failing mid-scoring),
score all rows, and present a sortable table ranked by probability with risk bands, plus a CSV
export.

**Guardrails:** cap the upload size (5,000 rows) with a clear message; state on screen that the
ranking is decision support requiring human review; do not persist uploaded data.

**Acceptance criteria.** A 1,000-row file scores in under 10 seconds; schema violations produce
actionable per-column errors; export works; uploaded data is never written to disk.

---

### H2-4 · GenAI retention-rationale layer

| | |
|---|---|
| **Owner** | Member D |
| **Effort** | 12 h |
| **Deliverable** | LLM-generated review brief, with guardrails and a kill switch |

**This is the highest-risk item in the plan, and the framing matters more than the code.**

An LLM that "explains why the customer will churn" would be inventing causal claims the model
cannot support — precisely the failure this project has been careful to avoid everywhere else.
Adding it naively would undermine the governance story that is the project's strongest feature.

**The defensible design:** the LLM is a **writing tool, not a reasoning tool.** It receives
*only* structured facts already computed deterministically — the probability, the risk band,
the top contributing features from H1-4, and the segment reference rates — and its sole job is
to render them as a short, cautiously-worded review brief for a specialist.

Mandatory guardrails:

1. **Grounding:** the prompt carries only computed values. The LLM never sees the raw customer
   record and is never asked to predict anything.
2. **Structured output:** constrain to a fixed schema (summary, factors, suggested questions,
   caveats). Validate before display; on validation failure, fall back to the deterministic
   template.
3. **Prohibited-language check:** post-generate, reject output containing causal or
   deterministic phrasing ("will churn", "because", "guaranteed"). Regenerate once, then fall
   back.
4. **Provenance:** label every generated brief as AI-generated on screen.
5. **Kill switch:** a config flag disables the layer entirely, and the app must be fully
   functional with it off. Ship with it **off by default** in the Space.
6. **No new secrets in the repo:** the API token is a Space secret, never committed.

Use HF Inference Providers via the existing Pro subscription. Cap tokens per request, and
handle provider failure by falling back to the deterministic template — never by showing an
error.

**Acceptance criteria.** Kill switch verified — the app works identically with the layer off; a
test asserting the prohibited-language filter rejects a causal-phrasing sample; brief labelled
AI-generated; documented statement that the LLM performs no inference about the customer.

---

### Horizon 2 summary

| Item | Owner | Hours | Closes |
|---|---|---:|---|
| H2-1 MLflow tracking + registry | A | 10 | G5 |
| H2-2 Drift apparatus | B | 10 | G6 |
| H2-3 Batch scoring queue | C | 12 | G7 |
| H2-4 GenAI rationale layer | D | 12 | — (differentiator) |
| Integration, testing, docs | All | 15 | — |
| **Total** | | **59 h** | |

---

## 6. Horizon 3 — Production path (post-submission)

Documented for completeness and to demonstrate strategic thinking. **Not scheduled inside the
four weeks.** Presenting a realistic "what we did not build and why" is itself a mark of
seniority.

| Initiative | Rationale | Indicative effort |
|---|---|---|
| FastAPI prediction service | Decouples the model from the UI; lets other systems consume it | 2 weeks |
| Postgres + feature store | Persist scores, outcomes and a genuine feedback loop | 3 weeks |
| Scheduled retraining with automated gates | Retrain on a trigger; block promotion on a metric regression | 2 weeks |
| Live monitoring and alerting | Drift and performance decay alerting to a real channel | 2 weeks |
| A/B evaluation of retention interventions | **The only way to make a genuine ROI claim** | 1 quarter |
| Model-risk documentation pack | Formal validation report for a model-risk committee | 2 weeks |
| Dependency and container scanning (`pip-audit`, Trivy) | Closes the known security gap in `docs/SECURITY.md` | 3 days |

**The strategic point for the conclusion of your report:** every claim about business value
remains unproven until the A/B evaluation exists. Until a control group has been observed, this
platform demonstrates *technical feasibility* and *may support prioritisation*. That is the
honest ceiling of the current evidence, and stating it is a strength rather than a hedge.

---

## 7. Team allocation and schedule

The team is **three people**, so the four workstreams are distributed as follows.

| Member | Student ID | Workstream | Horizon 1 | Horizon 2 |
|---|---|---|---|---|
| **Krishna Mathur** | AS25DXB018 | Platform, CI/CD, security | Space deployment, visible-change test | GenAI brief and its guardrails |
| **Yash Petkar** | AS25DXB021 | Modelling rigour | Fairness audit, calibration | MLflow tracking and registry |
| **Atharva Soundankar** | AS25DXB020 | Application and analysis | Explainability, threshold analysis | Batch scoring work queue, drift apparatus |

All three contributed to the report, the evidence pack and the demonstration.

| Milestone | Status |
|---|---|
| Space live; deployment verified end to end | ✅ Complete |
| Visible-change test (1.0.0 → 1.1.0) | ✅ Complete |
| Fairness, calibration, threshold, explainability | ✅ Complete |
| MLflow, drift, batch scoring, GenAI brief | ✅ Complete |
| Report, screenshots, rehearsal | ⬜ Outstanding — the team's remaining work |

**Hard rule carried over from v1.0.0:** no commit without `make test` and `make secret-scan`
passing. Every new feature ships with tests, or it does not ship.

---

## 8. Hugging Face deployment runbook

Do this **first**. Everything else is worth less until it is done.

### Step 1 — Create the Space

1. Go to <https://huggingface.co/new-space>
2. **Owner:** `krish21may` · **Space name:** `customer-churn-intelligence`
3. **SDK: Docker** → *Blank* template (**not** the Streamlit template — your repo supplies its
   own `Dockerfile`)
4. **Visibility: Public** (required for the report link to be openable by an examiner)
5. Create. Leave it empty — the workflow populates it.

Your resulting Space ID is **`krish21may/customer-churn-intelligence`**.

> Your existing `Bank-Customer-Churn` Space is the default template on port 8501. This project's
> `deploy/README.md` declares `app_port: 7860` and matches its own Dockerfile, so leave it as is.

### Step 2 — Create a scoped write token

1. <https://huggingface.co/settings/tokens> → **Create new token** → **Fine-grained**
2. Name: `github-actions-churn-deploy`
3. Under repository permissions, select **only** `krish21may/customer-churn-intelligence`, with
   **Write** access
4. Grant nothing else — no org scope, no inference, no billing
5. Copy the value **once**

**Never paste this token into a file, a chat, a screenshot or a terminal you are recording. I
must not see it either.**

### Step 3 — Configure GitHub

<https://github.com/krish2105/SDAIM-Customer-Churn/settings/secrets/actions>

- **Secrets** tab → *New repository secret* → name `HF_TOKEN`, value = the token
- **Variables** tab → *New repository variable* → name `HF_SPACE_ID`, value
  `krish21may/customer-churn-intelligence`

Or, for the non-secret variable only:

```bash
gh variable set HF_SPACE_ID --repo krish2105/SDAIM-Customer-Churn --body "krish21may/customer-churn-intelligence"
```

Set `HF_TOKEN` through the web UI, not the CLI — it keeps the value out of your shell history.

### Step 4 — Trigger and verify

```bash
gh workflow run "Deploy to Hugging Face Space" --repo krish2105/SDAIM-Customer-Churn --ref main
```

Then verify **in this order**, capturing each:

1. The Actions run turns green (Figure 29)
2. The Space **build logs** show a successful Docker build (Figure 31) — *this is separate from
   the workflow succeeding*
3. The live app loads (Figure 32)
4. A prediction works in the live app (Figure 33)

### Step 5 — Visible-change test (brief Part 4.4)

Follow `docs/DEMONSTRATION_SCRIPT.md` §"Visible-change deployment test": raise `MODEL_VERSION`
to `1.1.0` in `src/config.py`, `make train`, add the caption, run `make test` and
`make secret-scan`, commit, push, then verify a **second** run and 1.1.0 live.

### Hugging Face Pro — what is worth using here

| Capability | Verdict for this project |
|---|---|
| Increased Space quotas / longer uptime | **Use.** Keeps the demo live for marking. |
| **Persistent storage** | **Consider in H2** — the only clean way to retain uploaded batch files or logs across restarts. Not needed for H1. |
| **Dev Mode** (SSH/VS Code into a running Space) | **Use for debugging.** Faster than push-and-wait when diagnosing a build failure. |
| **ZeroGPU** | **Not applicable.** This is a CPU logistic regression; a GPU would idle. |
| **Inference Providers credits** | **Use in H2-4** for the LLM layer, with a token cap. |
| Private Spaces | Optional — keep public for the report link. |

**Verify each of these against current Hugging Face documentation before writing them into your
report.** Platform entitlements change, and I have not confirmed the 2026 Pro terms — treat this
table as a starting point to check, not a citation.

---

## 9. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Space deployment left until the final week | Medium | **Critical** — two brief sections unmarkable | Do H1-5 in week 1. It is 3 hours. |
| R2 | Fairness audit reveals a material disparity | **Medium-high** | Medium | This is a *finding*, not a failure. Document it, quantify the mitigation cost, recommend against group-specific thresholds. |
| R3 | Calibration work invalidates the current metrics | Low | Medium | Recalibration changes probabilities, not ranking — ROC-AUC is threshold-independent and will not move. |
| R4 | LLM layer generates a causal claim | Medium | **High** — undermines the governance story | Prohibited-language filter, structured output, fallback template, off by default. |
| R5 | Scope creep across four parallel workstreams | **High** | High | H1 items are independent and shippable alone. Cut H2 items whole, never partially. |
| R6 | `scikit-learn` version drift breaks the artifact | Low | High | Already mitigated: a test asserts the runtime pin matches the training version. |
| R7 | Token accidentally committed | Low | **Critical** | Scanner runs locally, in CI and pre-deploy. Rotation procedure in `docs/SECURITY.md` §4. |
| R8 | Adding heavy dependencies bloats the image | Medium | Medium | MLflow and drift tooling stay in dev requirements. A test already asserts runtime imports match runtime requirements. |

---

## 10. Success criteria

**Horizon 1 — all met:**

- [x] Space is live and a prediction works on it
- [x] Visible-change test verified with a second workflow run
- [x] Fairness audit published; the model card now states the finding, not its absence
- [x] Calibration measured; decision documented with the benefit quantified
- [x] Threshold sensitivity curve published; the app exposes an adjustable threshold
- [x] Per-prediction contributions render with causal-language guardrails
- [x] All tests green; `make verify` clean

**Horizon 2 — all met:**

- [x] Three MLflow runs comparable; model registered with a documented rollback
- [x] Drift detector demonstrated firing on a shift **and staying quiet on the control**
- [x] 1,000-row batch scores in 0.01 s and exports
- [x] LLM layer passes the prohibited-language test and works correctly when disabled
- [x] Runtime image unchanged — MLflow, scipy and the analysis stack stay in dev only

**Programme-level:**

| Metric | Baseline | Target | **Delivered** |
|---|---|---|---|
| Test count | 52 | 75+ | **106** |
| Documented limitations closed | 0 of 8 | 4 of 8 | **5 of 8** |
| External gates verified | 1 of 6 | 6 of 6 | **6 of 6** |
| Direct cost | £0 | £0 | **£0** |

---

## 11. Open questions for the team and instructor

1. **Cost ratio** — will the instructor accept a parameterised sweep in place of real costs?
   (Recommended framing: yes, because inventing costs would be fabrication.)
2. **Fairness remediation** — if a disparity is found, is documenting it sufficient for the
   assignment, or is remediation expected?
3. **GenAI scope** — is an LLM layer within the brief's scope, or a distraction from the
   assessed criteria? Confirm before spending 12 hours.
4. **Repository visibility** — public is currently set. Confirm this is acceptable, and that no
   group member objects to their name appearing on a public repository.
5. **The fictional-data question (D-01)** — still unconfirmed by the instructor. **Chase this
   now**; it affects how the report's framing is marked.

---

## 12. Recommendation

Execute Horizon 1 in full and in the stated order. It closes the four gaps most likely to be
challenged, it is achievable inside two weeks by four people, and every item converts a
published limitation into a demonstrated capability.

Treat Horizon 2 as a portfolio investment to be cut whole if time compresses — batch scoring
first if only one item survives, because it changes what the product *is*; the GenAI layer last,
because it carries the most risk to the governance narrative that is this project's strongest
asset.

Do not attempt Horizon 3 before submission. Present it as the evidence-based roadmap it is, and
be explicit that no business-value claim can be substantiated until the A/B evaluation in it has
actually been run.
