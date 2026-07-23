# Executive Model Summary

**Project:** Customer Churn Intelligence and Retention Decision-Support Platform  
**Model version:** 1.1.0  
**Selected model:** Logistic Regression  
**Decision threshold:** 0.5  
**Random seed:** 42

> The underlying data is IBM's **fictional** telecommunications sample. Results
> demonstrate technical feasibility and may support prioritisation of accounts for
> human review. No revenue, churn-reduction or ROI effect has been measured, and
> none is claimed.

## 1. What this model does

For one customer record it returns a churn class, a churn probability, a Low / Medium /
High communication band, and a cautious suggested review action. It is **decision
support for a human retention specialist**, not an autonomous decision system.

## 2. Selection rule (fixed before the test set was examined)

Candidates were compared by mean ROC-AUC under stratified 5-fold cross-validation on the training split only. Where the two means differed by less than 0.01 the tie was broken first on mean CV F1 and then on mean CV recall, because a missed churner costs more than an unnecessary review. The held-out test set was evaluated once, after selection, and did not influence the choice. Cross-validated ROC-AUC (gap 0.0005) and F1 (gap 0.0010) were both within the 0.01 tolerance, so the rule fell through to mean CV recall. Logistic Regression was selected with 0.8013, because a missed churner costs more than an unnecessary review.

## 3. Model comparison

Cross-validation is computed on the training split only; test columns come from the
held-out 20% that no model saw during fitting or selection.

| Model | Role | CV ROC-AUC | CV F1 | CV recall | Test ROC-AUC | Test F1 | Test recall | Test precision | Test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | candidate | 0.8460 ± 0.0124 | 0.6286 | 0.8013 | 0.8414 | 0.6130 | 0.7834 | 0.5034 | 0.7374 |
| Random Forest | candidate | 0.8454 ± 0.0091 | 0.6296 | 0.7632 | 0.8417 | 0.6349 | 0.7834 | 0.5337 | 0.7608 |
| Dummy (stratified) baseline | baseline | 0.5065 ± 0.0171 | 0.2762 | 0.2776 | 0.5163 | 0.2903 | 0.2914 | 0.2891 | 0.6217 |

## 4. Held-out performance of the selected model

| Metric | Value |
|---|---:|
| Accuracy | 0.7374 |
| Precision (churn) | 0.5034 |
| Recall (churn) | 0.7834 |
| F1 (churn) | 0.6130 |
| ROC-AUC | 0.8414 |

Confusion matrix on the held-out test set:

| | Predicted: retained | Predicted: churn |
|---|---:|---:|
| **Actual: retained** | 746 (true negative) | 289 (false positive) |
| **Actual: churn** | 81 (false negative) | 293 (true positive) |

## 5. What the errors mean for the business

- **False negatives (81 of 374 actual churners, 21.7%)** — customers who churned but were not flagged. These are
  the costly errors: the account receives no retention review at all, so the opportunity
  to intervene is lost silently. This is why recall is weighted heavily in selection.
- **False positives (289 of 582 flagged accounts, 49.7%)** — customers flagged for review who would have stayed. The cost
  is retention-specialist time and the risk of an unnecessary contact, which is recoverable.
- The model flags 582 of 1,409 test customers for review (41.3%), giving retention teams a bounded workload.

## 6. Why accuracy alone is insufficient

The sample contains 26.54% churners. A model that
always predicted "retained" would score 73.46%
accuracy while finding no churn risk whatsoever and providing zero business value.
The `Dummy (stratified) baseline` baseline confirms this: it reaches 0.6217 test accuracy with 0.2914 recall and 0.5163 ROC-AUC.

Recall, F1 and ROC-AUC are therefore the metrics that govern selection, because they
measure whether at-risk customers are actually identified.

## 7. Risk tiers

| Tier | Probability | Suggested handling |
|---|---|---|
| Low | p < 0.40 | No immediate action indicated; monitor in routine reporting. |
| Medium | 0.40 ≤ p < 0.70 | May support inclusion in a periodic review queue. |
| High | p ≥ 0.70 | May support prioritisation for structured human review. |

These bands are **communication aids for triage only**. They have not been independently
validated as business thresholds, and no cost-sensitive optimisation was performed.

## 8. Governance

The model must not autonomously change prices, terminate or modify contracts, deny
service, target customers unfairly, or make any financial or customer-treatment decision.
Every output requires human review before any customer is contacted.

## 9. Limitations

- The dataset is a fictional IBM sample; performance on any live population is unknown.
- It is a single cross-section with no time dimension, so no drift behaviour can be assessed.
- The class balance (26.54% positive) limits precision
  attainable at high recall.
- `gender` and `SeniorCitizen` are included as predictors; no formal fairness audit across
  these attributes has been carried out, and one is recommended before any operational use.
- The 0.5 decision threshold is a documented default, not a business-optimised operating point.
