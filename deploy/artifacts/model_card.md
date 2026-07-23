# Model Card — Customer Churn Intelligence

**Model name:** Logistic Regression  
**Model version:** 1.1.0  
**Trained (UTC):** 2026-07-23T15:58:50+00:00  
**Random seed:** 42  
**Decision threshold:** 0.5  
**Artifact:** `deploy/artifacts/model_pipeline.joblib` (complete preprocessing + estimator pipeline)

## Intended use

Estimating whether a single customer record shows elevated churn risk, so that a
**human retention specialist** can prioritise accounts for structured review. The
output is decision support. It provides a probability, a communication band and a
cautious suggested action for a person to act on or overrule.

## Out-of-scope uses

This model must **not** be used to:

- change prices or apply differential pricing automatically;
- terminate, downgrade or modify a contract;
- deny, restrict or degrade service;
- target or exclude customers in ways that could be unfair or discriminatory;
- make any financial or customer-treatment decision without human review;
- infer anything about an individual outside this feature set;
- operate on a population materially different from the training sample without
  revalidation.

## Dataset provenance

| Property | Value |
|---|---|
| Dataset | IBM Telco Customer Churn sample (`Telco-Customer-Churn.csv`) |
| Publisher | IBM |
| Official repository | https://github.com/IBM/telco-customer-churn-on-icp4d |
| Repository licence | Apache-2.0 |
| Verified Git blob SHA | `3de7a612d1609f25f21a455bda77948729369002` |
| Rows / columns | 7,043 / 21 |
| Positive class rate | 26.54% |

> **The data represents a fictional telecommunications company.** It is not actual
> observed commercial customer data and must never be described as such.

## Features

- 16 categorical predictors and 3 numeric predictors (19 total).
- `customerID` is excluded from the feature matrix; it carries no signal and
  would risk memorisation.
- `SeniorCitizen` is treated as a binary category rather than a continuous magnitude.
- `TotalCharges` is stripped of whitespace and coerced to numeric; the resulting missing
  values are median-imputed **inside** the pipeline, fitted on training data only.

## Training procedure

- Single stratified split: 80% train / 20% test, `random_state=42`.
- Model selection by stratified 5-fold cross-validation on the training
  split only.
- All imputation, encoding and scaling occur inside the `Pipeline`, so every
  cross-validation fold refits them from scratch and no test information leaks.
- Unseen categories at inference are absorbed by `OneHotEncoder(handle_unknown="ignore")`.

## Selection

Cross-validated ROC-AUC (gap 0.0005) and F1 (gap 0.0010) were both within the 0.01 tolerance, so the rule fell through to mean CV recall. Logistic Regression was selected with 0.8013, because a missed churner costs more than an unnecessary review.

## Metrics (held-out test set, evaluated once after selection)

| Metric | Value |
|---|---:|
| Accuracy | 0.7374 |
| Precision (churn) | 0.5034 |
| Recall (churn) | 0.7834 |
| F1 (churn) | 0.6130 |
| ROC-AUC | 0.8414 |

Full comparison including the reference baseline:

| Model | Role | CV ROC-AUC | Test ROC-AUC | Test recall | Test F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | candidate | 0.8460 | 0.8414 | 0.7834 | 0.6130 |
| Random Forest | candidate | 0.8454 | 0.8417 | 0.7834 | 0.6349 |
| Dummy (stratified) baseline | baseline | 0.5065 | 0.5163 | 0.2914 | 0.2903 |

## Limitations

- Trained on a fictional sample; performance on any live population is unknown and
  must be revalidated before operational use.
- A single cross-section with no time dimension, so no drift or seasonality behaviour
  can be assessed and no monitoring baseline exists.
- Class imbalance (26.54% positive) limits the
  precision attainable at high recall.
- The 0.5 threshold is a documented default, not a cost-optimised operating point. No
  cost matrix was supplied, so none was fitted.
- Risk tiers are communication bands and have not been validated as business thresholds.
- Predictions describe association within the sample, never causation. The model cannot
  say what would happen if a customer's contract or service were changed.

## Ethical and governance considerations

- `gender` and `SeniorCitizen` are included as predictors. **No formal fairness audit**
  (equalised odds, demographic parity or subgroup error analysis) has been carried out.
  One is required before any operational use, and removing or auditing these attributes
  should be considered explicitly.
- Outputs may support prioritisation. They are not a judgement about any person and must
  not be presented to a customer as a fact about them.
- Retention offers driven by a risk score can create differential treatment between
  customers. Any such use needs review by model-risk and governance functions first.
- The model produces no explanation of an individual prediction. Where a customer-facing
  reason is required, this model alone does not supply one.

## Human review requirement

Every prediction requires human review before any customer-affecting action is taken.
The application states this on screen. The model has no authority to act, and no
downstream system should consume its output as an automated trigger.

## Reproducing this artifact

```bash
make bootstrap
make validate
make train
```

Environment used for this artifact: Python 3.11.15, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.4.6.
