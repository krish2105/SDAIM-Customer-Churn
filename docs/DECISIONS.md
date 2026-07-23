# Design Decisions

Every material decision, the alternatives considered, and why the choice was made.
Decisions are numbered so the report and the viva can reference them directly.

---

## D-01 — Treating the IBM sample as the project dataset

**Decision.** Use the official IBM Telco Customer Churn sample and describe it accurately
as a **fictional** company sample everywhere it appears.

**Context.** The instructor brief asks for a *"real-world dataset"*. The IBM sample is
officially published, realistically structured, widely used in teaching and industry
demonstrations, and has a verifiable provenance chain (public repository, Apache-2.0
licence, exact Git blob SHA). It is not, however, observed commercial customer data.

**Alternatives considered.**
1. Describe it as real-world data — **rejected**: that would be a false statement about
   provenance, and provenance accuracy is the point of the exercise.
2. Substitute a genuinely observed commercial dataset — **rejected**: no such dataset was
   supplied, and sourcing customer data independently raises data-protection issues that
   an academic project cannot properly discharge.
3. Use the IBM sample and disclose its nature explicitly — **chosen**.

**Consequence.** An instructor confirmation is requested and recorded as pending in
`PROJECT_INPUTS.md`. The disclosure appears in the README, the app, the model card, the
Space README and the report template.

---

## D-02 — Excluding `customerID` from the feature matrix

**Decision.** Drop `customerID` before building `X`; keep it only for traceability.

**Rationale.** It is a unique identifier. It carries no generalisable signal, and a
sufficiently flexible model could memorise it. Tests assert it is absent from both the
feature schema and the fitted pipeline's `feature_names_in_`.

---

## D-03 — `SeniorCitizen` as a category, not a number

**Decision.** Cast `SeniorCitizen` to string and route it through the categorical branch.

**Rationale.** It is encoded 0/1 but is a binary flag, not a magnitude. Scaling it as a
numeric feature would imply an ordering and a distance that do not exist. The IBM data
dictionary classifies it as a binary categorical.

**Alternative.** Passthrough as numeric — works in practice for a 0/1 variable but
misrepresents the variable's type and would be inconsistent with the data dictionary.

---

## D-04 — Cleaning `TotalCharges` inside the pipeline, not in the raw file

**Decision.** Strip whitespace and coerce with `errors="coerce"` when loading; impute the
resulting missing values with a `SimpleImputer(strategy="median")` **inside** the
pipeline. `data/raw/` is never written to.

**Rationale.** Eleven rows have a blank `TotalCharges`, and all eleven have `tenure = 0` —
customers who have not completed a billing cycle. Imputing before the split would fit the
median on data that includes the test set, which is leakage. Keeping the raw file
immutable preserves the audited evidence and the verified blob SHA.

**Alternatives.**
1. Drop the 11 rows — **rejected**: loses a structurally meaningful group (brand-new
   customers) that the application must be able to score.
2. Fill with 0 — **rejected**: an unjustified assumption; median imputation is documented
   and fitted on training data only.

---

## D-05 — One split, before any transformer is fitted

**Decision.** `train_test_split(test_size=0.20, random_state=42, stratify=y)` executed once
on raw features, before any imputer, encoder or scaler exists.

**Rationale.** This is the project's primary leakage control. Every transformer lives
inside the `Pipeline`, so cross-validation refits them per fold and the test set never
influences a fitted parameter. Stratification keeps the 26.54% positive rate stable across
both splits (measured: 0.2654 train, 0.2654 test).

---

## D-06 — Two substantive models plus a reference baseline

**Decision.** Logistic Regression and Random Forest as candidates; a stratified
`DummyClassifier` scored alongside them as a reference that can never be selected.

**Rationale.** The brief requires at least two models. These two are complementary — a
regularised linear model and a bagged tree ensemble — explainable, and available in
scikit-learn without adding a dependency. The baseline exists to make the accuracy
argument concrete rather than rhetorical: it reaches 0.6217 test accuracy with 0.2914
recall and 0.5163 ROC-AUC, which demonstrates that accuracy alone is uninformative here.

**Alternatives.** XGBoost / LightGBM — **rejected**: the master specification explicitly
discourages them, they add a compiled dependency to the deployment image, and the marginal
gain would not change the conclusions of this project.

---

## D-07 — `class_weight="balanced"` on both candidates

**Decision.** Apply balanced class weights rather than resampling.

**Rationale.** The positive class is 26.54% of the data. Balanced weighting raises recall
on the minority class without synthesising records, keeps the pipeline simple, and adds no
dependency. SMOTE would require `imbalanced-learn` and introduces synthetic customers that
are hard to justify in a governance review.

---

## D-08 — Selection rule fixed before the test set was examined

**Decision.** The rule is written into `src/train.py` as executable code: mean CV ROC-AUC
first; if the gap is under 0.01, mean CV F1; if that gap is also under 0.01, mean CV
recall.

**Rationale.** Writing the rule in code before any test result exists is what makes the
held-out evaluation unbiased. A rule chosen after seeing test numbers is not a rule.

**What actually happened.** The two candidates were effectively tied: CV ROC-AUC 0.8460
(Logistic Regression) against 0.8454 (Random Forest), a gap of 0.0005; CV F1 0.6286 against
0.6296, a gap of 0.0010. Both gaps fell inside the tolerance, so the rule fell through to
CV recall, where Logistic Regression led 0.8013 to 0.7632. **Logistic Regression was
selected.**

**Honest note for the viva.** On the held-out test set the Random Forest scored slightly
better on accuracy (0.7608 vs 0.7374), precision (0.5337 vs 0.5034) and F1 (0.6349 vs
0.6130), with identical recall (0.7834) and effectively identical ROC-AUC (0.8417 vs
0.8414). The pre-declared rule was still followed, because changing the selection after
seeing test results would invalidate the held-out evaluation. The two models are close
enough that either is defensible; what matters methodologically is that the choice was not
made with test data in view. This is recorded rather than hidden.

---

## D-09 — Scaling only for the linear model

**Decision.** `StandardScaler` in the Logistic Regression pipeline; omitted from the
Random Forest pipeline.

**Rationale.** Regularised linear models need comparable feature scales for the penalty to
be applied fairly. Tree ensembles are invariant to monotonic rescaling, so scaling would
add a fitted step with no effect. `build_preprocessor(scale_numeric=...)` makes the
difference explicit rather than incidental.

---

## D-10 — `OneHotEncoder(handle_unknown="ignore")`

**Decision.** Ignore unseen categories at inference instead of raising.

**Rationale.** The deployed application must not crash on an unexpected input. An unknown
category becomes an all-zero block, so the prediction degrades gracefully and is still
returned for human review. A test asserts that scoring a record with an invented
`PaymentMethod` and `Contract` still yields a valid probability.

**Trade-off.** A silently ignored category is a silently less-informed prediction. The
form only offers observed categories, so this is a defence-in-depth measure rather than a
routine path.

---

## D-11 — Decision threshold left at 0.50

**Decision.** Keep 0.50 and document it as a default, not an optimum.

**Rationale.** Threshold optimisation requires a cost matrix — the relative cost of a
missed churner against an unnecessary review. No such costs were supplied, and inventing
them would be exactly the kind of fabricated business claim this project avoids. The
threshold is stored in metadata so it can be changed deliberately later.

---

## D-12 — Risk tiers as communication bands

**Decision.** Low < 0.40, Medium 0.40–0.70, High ≥ 0.70, labelled in the app as
communication bands that are **not** validated business thresholds.

**Rationale.** Tiers make a probability actionable for a non-technical reader. Presenting
them as validated operating points would imply an empirical calibration that was never
performed.

---

## D-13 — Complete pipeline in one artifact

**Decision.** Serialise the entire `Pipeline` — preprocessing and estimator — to a single
`model_pipeline.joblib` (8.4 KB).

**Rationale.** The brief permits saving components separately or the whole pipeline. One
artifact means the application cannot apply preprocessing that differs from training; the
transformation is inseparable from the model. It also removes any ordering or versioning
mismatch between separately saved encoders and the estimator.

**Consequence on size.** At 8.4 KB the artifact is far below any GitHub or Hugging Face
threshold, so **Git LFS is not required**. This was checked rather than assumed. Should a
future model exceed 50 MB, `docs/TROUBLESHOOTING.md` documents the LFS route.

---

## D-14 — Runtime versions pinned to the training environment

**Decision.** `deploy/requirements.txt` pins exact versions, including
`scikit-learn==1.9.0`, matching the environment that produced the artifact.

**Rationale.** Unpickling a scikit-learn estimator under a different minor version is
unsupported and can fail or, worse, change behaviour silently. A test asserts that the pin
matches `model_metadata.json["environment"]["scikit_learn"]`, so the two can never drift
apart unnoticed.

---

## D-15 — A separate `reference_rates.json` artifact

**Decision.** Training writes a small descriptive artifact holding training-split churn
rates per category and a histogram of held-out test-set scores. The app uses it only to
draw context charts.

**Rationale.** The executive charts needed genuine reference statistics. Embedding them in
`feature_schema.json` would conflate "how to build the form" with "descriptive context",
and recomputing them at runtime would require shipping the dataset into the container.
This addition is a deviation from the file list in the master specification and is recorded
here for that reason. The file participates in no prediction, and both charts state on
screen that they are descriptive rather than causal.

---

## D-16 — Custom CSS design system instead of native Streamlit theming

**Decision.** Pin Streamlit to its light base theme in `.streamlit/config.toml` and layer a
token-based light/dark palette from `deploy/theme.py`, driven by an in-app toggle.

**Rationale.** Streamlit's native theming follows the operating system and offers no
in-app switch, which the brief for this application required. Tokens are defined once and
consumed by both the CSS and the matplotlib charts, so the charts change with the theme
instead of staying light-mode islands. Without pinning the base theme, Streamlit's own
widget colours fight the toggle — this was observed and fixed during development.

**Constraints honoured.** No external font, stylesheet or script is referenced: a Docker
Space should not depend on a CDN at request time, and a remote asset would be a
supply-chain dependency. Type comes from the platform stack. No CSS rule hides a focus
ring, all interactive targets are at least 44 px, and both palettes were checked against
WCAG AA contrast.

**Trade-off.** Styling another framework's internals is version-sensitive. The overrides
target stable `data-testid` and ARIA attributes rather than generated class names, and the
layout degrades to plain Streamlit rather than breaking if a selector stops matching.

---

## D-17 — Sections built from `st.container(border=True)`

**Decision.** Use Streamlit's bordered container for the form's card sections, with the
section heading rendered inside it.

**Rationale.** An opening `<div>` written via `st.markdown` does not enclose subsequent
widgets: Streamlit closes unbalanced HTML at the end of each markdown block, so the card
rendered empty and the inputs fell outside it. This was observed in the browser and fixed.
The native container is the only construct that genuinely wraps the widgets.

---

## D-18 — Assessment persisted in session state

**Decision.** Store the last assessment in `st.session_state` and re-render it, showing a
notice when the inputs have since changed.

**Rationale.** Any rerun — switching the theme, for example — would otherwise discard the
result on screen, which reads as a fault. Persisting it risks the opposite problem, a stale
result presented as current, so the staleness notice is part of the decision rather than an
afterthought.

---

## D-19 — Action versions verified against their published definitions

**Decision.** `actions/checkout@v6`, `actions/setup-python@v6`,
`huggingface/hub-sync@v0.2.1`.

**Rationale.** The master specification named `huggingface/hub-sync@v0.1.0`; it also allows
a newer version when the published documentation shows one. v0.2.1 is the current release
(published 2026-07-15) and was used. Reading the action definition also revealed a
**required** input the specification did not mention — `github_repo_id` — which is now
supplied as `${{ github.repository }}`. Without it the deployment job would have failed at
run time. `actions/checkout` was kept at v6 as specified; v7 was released three days before
this build and cannot be validated here.

---

## D-20 — CI validates before deployment, and never fabricates success

**Decision.** `deploy.yml` runs a full validation job (dataset, tests, secret scan) that
the sync job depends on, and its summary states that a successful sync is not a successful
deployment.

**Rationale.** A workflow that reports "deployed" when it has only pushed files invites a
false claim in the report. The Space build happens after the sync and must be observed
separately. Configuration is checked first so a missing `HF_SPACE_ID` or `HF_TOKEN` fails
with an actionable message rather than an obscure push error.

---

# Horizon 1 and 2 decisions

Recorded when the improvement plan was implemented. These are the choices an examiner is
most likely to probe, so the reasoning is written down rather than left in commit messages.

---

## D-21 — Equal opportunity as the governing fairness criterion

**Decision.** Report demographic parity, equal opportunity and predictive parity; optimise
for **equal opportunity**.

**Rationale.** The three criteria are mathematically incompatible whenever group base rates
differ, so a choice is forced and must be justified. This model exists to find at-risk
customers so a specialist can review them. The harm that matters is a customer being
**missed** — receiving no review at all — and that harm should not fall more heavily on one
group. Equal opportunity equalises recall, which is exactly that.

**Why not demographic parity.** Senior citizens churn at 44.14% against 23.25% for
non-seniors. Forcing equal flag rates would mean deliberately under-flagging the group that
actually churns more — worse service dressed up as fairness.

**The finding that surprised us.** The equal-opportunity gap runs *in favour* of seniors:
recall 0.9286 against 0.7319. The group under-served on the criterion we optimise for is
**non-seniors**. The genuine burden on seniors is the false-positive gap (0.5000 against
0.2492) — unnecessary contact, not missed reviews. Stating this correctly matters more than
reporting a gap and implying the usual direction.

---

## D-22 — Measure the cost of removing protected attributes before deciding

**Decision.** Retrain without `gender` and `SeniorCitizen`, measure the loss, then decide.

**Rationale.** "Keep or remove a protected attribute" is usually argued from principle. It
is answerable with evidence, and the evidence changed the recommendation: removal costs
**+0.0008 ROC-AUC** and **+0.0027 recall**, well inside the ±0.0124 cross-validation
standard deviation. The attributes buy nothing measurable while creating a disparity that
must then be explained. **Removal is now recommended.**

**Why they are retained in 1.1.0 regardless.** The measured metrics are already published
across the report, the model card and the live application. Silently changing the model
mid-submission would invalidate that evidence trail. Recorded as the first Horizon 3 action
with the cost quantified — documented, not deferred by omission.

---

## D-23 — Log-odds contributions instead of SHAP

**Decision.** Decompose predictions as `coefficient × transformed_value`.

**Rationale.** For a linear model this is not an approximation of the model — it **is** the
model, and it reconstructs the score exactly (asserted by test on 25 customers). SHAP would
estimate, at the cost of a heavy dependency, a quantity available here in closed form.
Under viva questioning, arithmetic the team can derive beats a library it cannot.

**Limitation accepted.** The method does not generalise. If the Random Forest were ever
selected, `explain_prediction` reports that no additive decomposition exists rather than
silently substituting a different method — permutation importance would be the correct
replacement.

---

## D-24 — Publish a cost-ratio curve instead of choosing a threshold

**Decision.** Sweep the ratio of miss-cost to review-cost and publish the optimum against it.

**Rationale.** The cost-optimal threshold depends entirely on a business quantity nobody
supplied. Inventing a currency figure would be fabricating exactly the kind of business
claim this project refuses to make elsewhere. Only the *ratio* matters, not absolute costs,
so the whole answer can be published without inventing anything: 0.73 at 1:1, down to 0.12
at 20:1.

**Consequence.** The deployed 0.50 is cost-optimal at roughly **3:1** — using it was always
an implicit assertion about relative costs. The number was there all along; it simply was
not stated. The application now exposes the threshold so the assumption is visible rather
than buried.

---

## D-25 — PSI alone governs drift status; the chi-squared p-value does not

**Decision.** Report the chi-squared p-value for information, but let PSI decide the status.

**Rationale.** This was not theoretical. An earlier version escalated status when p < 0.05
and produced a **false positive on the unshifted control set** — `OnlineBackup`, PSI 0.0044
(no meaningful drift) with p = 0.047. At n ≈ 1,400 the test flags differences far too small
to matter. A detector that cries wolf on its own holdout would be switched off within a week.

The documentation had already said "PSI, not the p-value, governs the status". The code
contradicted it. The code was wrong and was fixed.

---

## D-26 — Build drift apparatus, and refuse to call it monitoring

**Decision.** Build the detectors and validate them on a simulated shift; state plainly that
no real drift can be observed.

**Rationale.** The dataset is a single cross-section with no time dimension. There is no
"later" distribution. Claiming to monitor drift would be false. What *can* be demonstrated
is that the apparatus works — and it is validated **both ways**: stable on the unshifted
holdout (0 of 19 flagged) and alert on a simulated acquisition campaign (15 of 19, 7 at
alert). A detector never shown to fire is not evidence of anything; nor is one that always
fires.

**Why not Evidently.** Evaluated and rejected. A large transitive dependency tree for two
detectors implemented here in a few dozen lines, whose internals the team could not defend.

---

## D-27 — The LLM is a writing tool, not a reasoning tool

**Decision.** The retention brief receives only pre-computed values, never the customer
record. It is never asked to predict or to explain *why* anyone might leave.

**Rationale.** An LLM asked to explain a churn score will produce fluent causal claims the
model cannot support. That would undermine the governance position that is this project's
strongest feature — every other surface is careful to say association, not causation.
Confining the model to rendering numbers keeps the guarantee intact.

**Guardrails, all enforced rather than assumed:** structured output validated before display;
a prohibited-language filter rejecting causal and deterministic phrasing; provenance
labelling on screen; a kill switch; and a deterministic fallback that must pass the same
language filter it enforces. **Ships disabled by default**, so the Space never depends on an
external provider.

---

## D-28 — MLflow on SQLite, and precision about what it adds

**Decision.** Local SQLite backend. No tracking server, no cloud.

**Rationale.** MLflow 3.x retired the plain filesystem store and raises unless
`MLFLOW_ALLOW_FILE_STORE=true`. Opting out of a deprecation to keep a retired code path
alive would be borrowing trouble. SQLite is the documented migration target and is still a
single local file with no infrastructure, so the scope limit is unaffected.

**Precision that matters.** MLflow does **not** add reproducibility — seeds and pinned
versions already provide that, and claiming otherwise would be wrong. It adds
**comparability across runs**, **rollback**, and **attribution** of a metric to a dataset
revision via the logged blob SHA.

**Rollback is deliberately manual.** Re-pointing production at a different model has
customer-facing consequences and should require a human, consistent with the governance
position taken everywhere else.

---

## D-29 — Feature engineering and tuning were tried, measured, and not adopted

**Decision.** Build seven engineered features and grid-search both models, compare four arms
by cross-validation, and **keep the existing model** because the gain did not clear a
pre-declared bar.

**Why the experiment was run.** Version 1.1.0 used the 19 raw columns with hand-set
hyperparameters. Task 1.2 of the brief lists feature engineering explicitly, and "we did not
try" is a weaker answer than "we tried and here is what happened".

**Design.** Four arms so each change is separable rather than confounded:

| Arm | Features | Hyperparameters | Best CV ROC-AUC |
|---|---|---|---:|
| A | Raw 19 | Hand-set (deployed) | 0.8460 |
| B | Raw 19 | Grid-searched | 0.8481 |
| C | Raw + 7 engineered | Hand-set | 0.8468 |
| D | Raw + 7 engineered | Grid-searched | 0.8476 |

**Adoption rule, fixed before the results were read.** Adopt only on a gain of at least
**0.005** mean CV ROC-AUC. Anything smaller sits inside the ±0.0124 cross-validation standard
deviation already measured for this model.

**Outcome.** Best arm was Random Forest with tuned hyperparameters on raw features, at 0.8481
— a gain of **+0.0022**, less than half the bar. **Not adopted. The deployed model is
unchanged.**

**Why this is a result rather than a failure.** It says something specific: the signal in this
dataset is largely exhausted by the raw features and sensible defaults. The engineered
features — spend trajectory, service counts, tenure bands — encode information the model was
already extracting, and the hyperparameters were already near optimal. Adopting a change this
small would add a permanent second transformation to keep consistent between training and
serving, a larger surface to explain, and more to maintain, all to chase a difference
indistinguishable from noise.

Adopting it anyway because the work had been done is precisely the sunk-cost reasoning the
project's selection rule exists to prevent.

**Methodological cost, stated plainly.** The held-out test set has now been consulted twice
across the project's lifetime. Two things limit the damage: no decision in this experiment
used test data — every arm, hyperparameter and the adoption call came from cross-validated
training performance — and the outcome was not to change the model, so the published v1.1.0
metrics remain a single-use evaluation of the deployed artifact. Had the experiment
recommended adoption, the new metrics would have had to be reported as a mildly optimistic
second-look estimate rather than an untouched holdout result.
