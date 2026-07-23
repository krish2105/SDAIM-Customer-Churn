---
title: Customer Churn Intelligence
emoji: 📉
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Customer Churn Intelligence and Retention Decision-Support Platform

## Business purpose

A telecommunications organisation needs a consistent way to estimate whether a customer
is at elevated risk of churn, so that retention specialists can prioritise accounts for a
structured human review. This application scores one customer record at a time and returns
a risk assessment for a person to act on or overrule.

It is **decision support**. It does not decide anything.

## Inputs and outputs

**Inputs** — 19 predictors grouped into four sections:

| Group | Fields |
|---|---|
| Customer profile | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Contract and billing | Contract, PaperlessBilling, PaymentMethod |
| Charges and tenure | tenure, MonthlyCharges, TotalCharges |

Logical dependencies are enforced in the form: with no phone service `MultipleLines` is
locked to `No phone service`, and with no internet service every internet add-on is locked
to `No internet service` — the only combinations present in the training data.

**Outputs** — predicted churn class, churn probability, a Low / Medium / High risk band, a
cautious suggested review action, the model version, and three context charts.

## Model summary

| Property | Value |
|---|---|
| Model | Logistic Regression (`class_weight="balanced"`) inside a complete scikit-learn `Pipeline` |
| Compared against | Random Forest (400 trees), plus a stratified `DummyClassifier` reference |
| Preprocessing | Median imputation + standard scaling (numeric); most-frequent imputation + one-hot encoding with `handle_unknown="ignore"` (categorical) |
| Split | 80 / 20 stratified, `random_state=42`, split once before any transformer was fitted |
| Selection | Stratified 5-fold cross-validation on the training split only |
| Decision threshold | 0.50 (documented default, not cost-optimised) |
| Model version | 1.1.0 |

## Measured performance

Held-out test set (1,409 customers), evaluated once after selection:

| Metric | Value |
|---|---:|
| Accuracy | 0.7374 |
| Precision (churn) | 0.5034 |
| Recall (churn) | 0.7834 |
| F1 (churn) | 0.6130 |
| ROC-AUC | 0.8414 |

Confusion matrix: 746 true negatives, 289 false positives, 81 false negatives, 293 true
positives. The model identifies 293 of the 374 actual churners in the test set and misses 81.

Recall is weighted heavily because a missed churner receives no retention review at all,
whereas an unnecessary review costs specialist time and is recoverable.

## Dataset source

- IBM Telco Customer Churn sample — <https://github.com/IBM/telco-customer-churn-on-icp4d>
- File: `Telco-Customer-Churn.csv`, 7,043 rows × 21 columns
- Verified Git blob SHA: `3de7a612d1609f25f21a455bda77948729369002`
- Repository licence: Apache-2.0

> **The data represents a fictional telecommunications company.** It is not actual observed
> commercial customer data and must never be described as such.

## Limitations

- Trained on a fictional sample; performance on any live population is unknown.
- A single cross-section with no time dimension, so drift cannot be assessed.
- The positive class is a minority (26.54%), which limits precision at high recall.
- Risk bands are communication aids for triage, not validated business thresholds.
- `gender` and `SeniorCitizen` are predictors and **no formal fairness audit has been carried
  out**. One is required before any operational use.
- Predictions describe association within the sample, never causation.
- No explanation of an individual prediction is produced.

## Governance notice

This model must not autonomously change prices, terminate or modify contracts, deny
service, target customers unfairly, or make any financial or customer-treatment decision.
Every output requires human review before a customer is contacted. No revenue,
churn-reduction or return-on-investment effect has been measured, and none is claimed.

## Deployment

Docker SDK Space listening on port 7860. Files are synchronised from the `deploy/`
directory of the GitHub repository by a GitHub Actions workflow, authenticated with a
fine-grained Hugging Face token held as the repository secret `HF_TOKEN`. The target Space
is configured as the repository variable `HF_SPACE_ID`. No credential appears in this
repository.

GitHub repository: `<<UNRESOLVED: add the verified repository URL once it exists>>`
