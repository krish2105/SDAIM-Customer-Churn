# Demonstration Script

A five-minute demonstration, plus the visible-change deployment procedure and likely viva
questions with evidence-based answers.

---

## Before you start

| Have open | Why |
|---|---|
| Terminal in the project root, `.venv` activated | Live commands |
| `reports/figures/` in a file browser | Instant chart access |
| The deployed Hugging Face Space | The headline deliverable |
| The GitHub repository, Actions tab | Workflow evidence |
| `reports/executive_model_summary.md` | Every number you may be asked for |

Rehearse once with a timer. The section timings below total 5:00.

---

## 0:00–0:40 — Business problem and decision supported

> "A telecommunications company loses customers every month, and retention specialists have
> limited time. There was no consistent basis for deciding which accounts deserve their
> attention — it came down to individual judgement.
>
> We built a decision-support application. It takes one customer record and returns a churn
> probability, a Low/Medium/High risk band, and a suggested review action.
>
> I want to be precise about what it is: it is decision support. It does not change prices,
> it does not modify contracts, and it does not contact anyone. A person decides. That
> constraint is enforced in the model card, stated in the application, and repeated in the
> report."

**Show.** The application home screen.

---

## 0:40–1:10 — Dataset and provenance disclosure

> "We used IBM's Telco Customer Churn sample — 7,043 customers, 21 columns, published under
> Apache-2.0.
>
> One thing to state clearly: this represents a **fictional** company. It is realistic and
> officially published, but it is not observed commercial customer data, and we describe it
> that way everywhere. The brief asked for a real-world dataset, so we have flagged that for
> confirmation rather than glossing over it.
>
> We verified we have the exact official file. This is the Git blob SHA — it matches the
> published revision precisely. Every number in our report is tied to this file."

**Show.**
```bash
git hash-object data/raw/Telco-Customer-Churn.csv
make validate
```

---

## 1:10–1:50 — One important EDA observation

> "The strongest signal is contract term. Month-to-month customers churn at 42.7%. One-year
> contracts, 11.3%. Two-year, 2.8% — a fifteen-fold difference.
>
> Payment method is close behind: electronic check 45.3% against 15.2% for automatic credit
> card.
>
> Two caveats we state in the report. These are associations within the sample, not causal
> effects — we cannot say that moving someone onto a two-year contract would reduce their
> risk. And the service add-on comparisons are partly confounded, because 'No internet
> service' appears as a level in every add-on column.
>
> The other finding that shaped the modelling: churn is 26.5% of the sample. That imbalance
> is why we do not use accuracy to choose a model."

**Show.** `06_churn_rate_by_contract.png`, then `01_target_distribution.png`.

---

## 1:50–2:40 — Model comparison and selection

> "We compared Logistic Regression and Random Forest, with a stratified dummy classifier as
> a reference point.
>
> The baseline is the important one. It gets 62% accuracy — while finding almost no
> churners: recall 0.29, ROC-AUC 0.52, no better than chance at ranking. That is the concrete
> demonstration that accuracy alone tells you nothing on imbalanced data.
>
> We fixed our selection rule in code *before* looking at the test set: cross-validated
> ROC-AUC first, then F1, then recall, with a 0.01 tie tolerance. The two models tied on
> ROC-AUC — 0.8460 against 0.8454 — and on F1, so the rule fell through to recall, where
> Logistic Regression led 0.80 to 0.76. That selected Logistic Regression.
>
> I will be straight about something. On the held-out set the Random Forest is marginally
> better on accuracy, precision and F1, with identical recall. We did not switch. Reselecting
> after seeing test results would make the held-out evaluation meaningless. That is written
> up in our decisions document rather than hidden."

**Show.** `reports/model_comparison.csv` or the `make train` output.

---

## 2:40–3:10 — Final metrics and what the errors mean

> "On 1,409 held-out customers: ROC-AUC 0.841, recall 0.783, precision 0.503, F1 0.613,
> accuracy 0.737.
>
> The confusion matrix is where the business meaning is. We catch 293 of 374 actual
> churners and miss 81. We flag 289 customers who would have stayed.
>
> Those errors are not equal. A missed churner gets no review at all — the chance to
> intervene is gone silently. A false positive costs a specialist's time and is recoverable.
> That asymmetry is why we weight recall, and it is why we accept precision near 0.50 at
> this threshold."

**Show.** `12_confusion_matrix_logistic_regression.png`.

---

## 3:10–3:50 — Live prediction

> "Here is the deployed application. I will load a documented profile — a new
> month-to-month fibre customer with no support add-ons.
>
> Notice the form enforces logical dependencies: if I set internet service to 'No', every
> internet add-on locks to 'No internet service', because that is the only combination the
> model has ever seen.
>
> Generate. Here is the probability, the risk band, and the suggested action — deliberately
> cautious wording: 'may support prioritising the account for a structured human review.'
>
> And here is the honesty layer. The bands are communication aids, not validated business
> thresholds. This panel lists the limitations, including that we have not carried out a
> fairness audit and one is required before operational use."

**Show.** The live Space. Demonstrate the dependency lock, then generate a prediction.

---

## 3:50–4:20 — Repository and CI

> "Everything is version-controlled. Source, deployment package, 52 automated tests,
> documentation and two workflows.
>
> CI runs on every push and pull request: it validates the dataset against its blob SHA,
> compiles everything, runs the tests, scans for secrets, and verifies the deployment package
> is complete.
>
> The tests are not decorative. One reverses the input column order and requires an identical
> probability — that catches positional instead of name-based column selection, which would
> silently produce wrong answers. Another parses the app's imports and checks each one is in
> the runtime requirements, so a missing dependency fails locally rather than in production."

**Show.** The repository, then a green CI run.

---

## 4:20–5:00 — Automated deployment and the visible-change proof

> "The deployment workflow triggers on pushes to main that touch the deploy directory. It
> runs a full validation job first, and the deploy job depends on it — a failing test means
> nothing ships.
>
> Authentication uses a fine-grained Hugging Face token stored as a GitHub Actions secret,
> and the Space ID as a repository variable. The token is never echoed, never put in a URL,
> never in the summary. There is a test asserting the workflows reference no other secret.
>
> To prove the automation works, we changed the version from 1.0.0 to 1.1.0 and pushed. Here
> is the second workflow run, and here is the live application showing 1.1.0 — no manual step
> anywhere.
>
> Finally, the constraint we started with: every output requires human review. The model
> prioritises accounts for a person. It does not decide anything about a customer."

**Show.** The workflow file, the second run, then 1.1.0 live in the Space.

---

# Visible-change deployment test — procedure

Run this **after** the first deployment has succeeded and been observed.

```bash
# 1. Raise the version
#    Edit src/config.py:  MODEL_VERSION: Final[str] = "1.1.0"
```

```bash
# 2. Retrain so the artifact metadata carries the new version
make train
python -c "import json;print(json.load(open('deploy/artifacts/model_metadata.json'))['model_version'])"
# must print 1.1.0
```

```bash
# 3. Add the visible caption in deploy/app.py, inside the header block:
#    <span class="cci-badge">Application version 1.1.0 — automated deployment verification</span>
```

```bash
# 4. Verify before committing
make test
make secret-scan
git diff
git diff --cached --name-only
```

```bash
# 5. Commit and push
git add -A
git commit -m "Release version 1.1.0 with automated deployment verification caption"
git push origin main
```

Then verify, in order — and **capture each as you go**:

1. A **second** Actions run starts (Figure 35).
2. It completes successfully.
3. The Space rebuilds — watch the build logs.
4. The live app shows **version 1.1.0** and the new caption (Figure 36).
5. A prediction still works (Figure 37).

Do not claim this test passed until you have seen all five with your own eyes.

---

# Likely viva questions

**Why this dataset when the brief says "real-world"?**
It is officially published by IBM under Apache-2.0, realistically structured, and fully
traceable — we verified the exact Git blob SHA. It does represent a fictional company, so we
say that everywhere rather than overstating it, and we flagged it for instructor
confirmation in `PROJECT_INPUTS.md`.

**How do you know there is no data leakage?**
Structurally rather than by inspection. The split happens once, before any transformer
exists. Every imputer, encoder and scaler lives inside the `Pipeline`, so `cross_validate`
refits them from scratch on each fold. The test set is touched exactly once, after
selection. If we had imputed `TotalCharges` before splitting, the median would have been
computed using test rows — that is the specific mistake the design prevents.

**Why Logistic Regression when Random Forest scored better on the test set?**
Because the selection rule was fixed in code before any test result existed. The models tied
on cross-validated ROC-AUC and F1, so the rule fell through to recall, where Logistic
Regression led. Switching after seeing test numbers would turn the held-out set into a
selection set and invalidate the evaluation. The difference is small and either model is
defensible; what is not defensible is choosing after peeking.

**Why is precision only 0.50?**
It is the cost of high recall on an imbalanced target at a 0.50 threshold. Churners are 26.5%
of the data. To catch 78% of them we accept that about half the flagged accounts would have
stayed. That is a deliberate trade: a missed churner is unrecoverable, an unnecessary review
is not. Raising the threshold would raise precision and lower recall — we did not tune it
because no cost matrix was supplied and inventing one would be a fabricated business claim.

**Why not XGBoost?**
It would add a compiled dependency to a container we need to keep small and reproducible,
and the two models we compared are already within 0.001 ROC-AUC of each other — the ceiling
here is the data, not the algorithm. Explainability also matters for a governance review.

**What happens if a user enters a category the model never saw?**
`OneHotEncoder(handle_unknown="ignore")` turns it into an all-zero block, so the prediction
degrades rather than crashing. There is a test that scores a record with an invented payment
method and contract and asserts a valid probability comes back. The form only offers observed
categories, so this is defence in depth.

**How is the Hugging Face token protected?**
It is a fine-grained token scoped to write to one Space, stored as a GitHub Actions secret. It
is read only as `${{ secrets.HF_TOKEN }}`, never echoed, never interpolated into a URL, never
written to the job summary. A test asserts the workflows reference no secret other than
`HF_TOKEN`. A secret scanner runs locally, in CI, and again before deployment — and it prints
only file paths and pattern categories, never the matched text, because a scanner that echoed
findings would leak them into CI logs.

**What are the biggest limitations?**
Three. It is trained on a fictional sample, so live performance is unknown. It is a single
cross-section with no time dimension, so we cannot assess drift and have no monitoring
baseline. And `gender` and `SeniorCitizen` are predictors with **no fairness audit performed**
— that is a real gap, documented in the model card, and it would need closing before
operational use.

**Could this be automated end to end?**
Technically yes, and it should not be. A churn score is a statistical association, not a
judgement about a person. Automating retention offers from it creates differential treatment
between customers, which needs model-risk and governance review. The model card lists
automated pricing, contract changes and service denial as explicitly out of scope.

**What would you do next?**
Four things, in order: a fairness audit across the demographic attributes; threshold
optimisation once real intervention costs are available; calibration analysis, since we report
probabilities and have not verified they are calibrated; and dependency and container
vulnerability scanning in CI, which is a known gap in `docs/SECURITY.md`.

**How do you know the deployed model is the one you trained?**
The artifact is a single file containing preprocessing and estimator together, so the
deployed transformation cannot differ from the trained one. The metadata records the training
timestamp, seed and dataset blob SHA. A test asserts the runtime `scikit-learn` pin matches
the training version, because unpickling across versions is unsupported. And the container
returned the identical probability to the local environment when we tested it.
