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
